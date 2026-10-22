import argparse
import os

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from dreamcan.common import load_config, set_seed
from dreamcan.lt_value import LTFeatures, LTTransitionDataset, LTValueNet
from dreamcan.value import expectile_loss, goal_reward, sample_goals


def td_loss(model, batch, features, ep_lengths, cfg, accelerator):
    goal_ep, goal_t = sample_goals(batch["ep"], batch["t"], batch["n"], ep_lengths, cfg["gamma"])
    goal_local, goal_global = (x.to(batch["local"].device, non_blocking=True) for x in features.goals(goal_ep.cpu(), goal_t.cpu()))
    language = features.language[goal_ep.cpu()].to(batch["local"].device, non_blocking=True)
    with accelerator.autocast():
        v = model(batch["local"], batch["global"], goal_local, goal_global, language)
        with torch.no_grad():
            v_next = model(batch["next_local"], batch["next_global"], goal_local, goal_global, language, target=True)
    y = goal_reward(batch["ep"], batch["t"], goal_ep, goal_t) + cfg["gamma"] * (1.0 - batch["terminal"].float()) * v_next
    return expectile_loss(y - v, cfg["expectile"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    feature_path = cfg["features"].format(task=args.task)
    output_dir = cfg["output_dir"].format(task=args.task)

    accelerator = Accelerator(
        mixed_precision="bf16", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    set_seed(cfg["seed"] + accelerator.process_index)
    device = accelerator.device

    features = LTFeatures(feature_path, cfg["model"]["history_len"])
    model = LTValueNet(features.local_dim, features.global_dim, features.language.shape[1], **cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]

    loader = DataLoader(
        LTTransitionDataset(features),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        prefetch_factor=1 if cfg["num_workers"] > 0 else None,
        persistent_workers=cfg["num_workers"] > 0,
    )
    model, opt, loader = accelerator.prepare(model, opt, loader)
    raw = accelerator.unwrap_model(model)
    ep_lengths = torch.tensor(features.lengths, device=device)
    os.makedirs(output_dir, exist_ok=True)

    def batches():
        while True:
            yield from loader

    it = batches()
    pbar = tqdm(range(step + 1, cfg["steps"] + 1), disable=not accelerator.is_main_process)
    model.train()
    for step in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
        loss = td_loss(model, batch, features, ep_lengths, cfg, accelerator)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        raw.update_target(cfg["target_tau"])
        if accelerator.is_main_process and step % cfg["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            if accelerator.is_main_process:
                torch.save(
                    {"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg},
                    os.path.join(output_dir, f"value_{step:06d}.pt"),
                )
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
