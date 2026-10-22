import argparse
import math
import os

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from dreamcan.common import load_config, load_stats, set_seed
from dreamcan.policy import GoalPolicy, flow_matching_loss
from dreamcan.policy_data import PolicyDataset


def lr_at(step, cfg):
    if step < cfg["warmup_steps"]:
        return cfg["lr"] * (step + 1) / cfg["warmup_steps"]
    progress = min(1.0, (step - cfg["warmup_steps"]) / max(1, cfg["cosine_steps"] - cfg["warmup_steps"]))
    return 0.5 * cfg["lr"] * (1.0 + math.cos(math.pi * progress))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank, local_rank, world = 0, 0, 1
    if distributed:
        dist.init_process_group("nccl")
        rank, local_rank, world = dist.get_rank(), int(os.environ["LOCAL_RANK"]), dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    set_seed(cfg["seed"] + rank)

    instr = np.load(cfg["instructions"]) if cfg.get("instructions") else None
    texts = [str(s) for s in instr["texts"]] if instr is not None else None
    stats = load_stats(cfg["stats"])
    aug = cfg["augment"]
    dataset = PolicyDataset(
        cfg["data"],
        cfg["radio_features"],
        cfg["goal_features"],
        stats,
        texts,
        views=cfg["views"],
        state_keys=cfg["state_keys"],
        history_len=cfg["model"]["history_len"],
        chunk_len=cfg["model"]["action_chunk_len"],
        crop_prob=aug["crop_prob"],
        crop_scale=tuple(aug["crop_scale"]),
        state_noise=aug["state_noise"],
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = GoalPolicy(instr["embeddings"] if instr is not None else None, **cfg["model"]).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    raw = model.module if distributed else model
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if any(k in name for k in cfg["no_weight_decay"]) else decay).append(param)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}], lr=cfg["lr"]
    )
    step, epoch = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step, epoch = ckpt["step"], ckpt["epoch"]
    os.makedirs(cfg["output_dir"], exist_ok=True)

    pbar = tqdm(total=cfg["max_steps"], initial=step, disable=rank != 0)
    while step < cfg["max_steps"]:
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            if step >= cfg["max_steps"]:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for group in opt.param_groups:
                group["lr"] = lr_at(step, cfg)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = flow_matching_loss(model, batch, cfg["task_aux_weight"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            pbar.update(1)
            if rank == 0 and step % cfg["log_every"] == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_at(step, cfg):.2e}")
            if rank == 0 and step % cfg["save_every"] == 0:
                torch.save(
                    {
                        "model": raw.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": cfg["model"],
                        "stats": {k: v.tolist() for k, v in stats.items()},
                        "instructions": texts,
                        "step": step,
                        "epoch": epoch,
                    },
                    os.path.join(cfg["output_dir"], f"policy_{step:06d}.pt"),
                )
        epoch += 1
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
