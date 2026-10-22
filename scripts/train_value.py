import argparse
import os

import h5py
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from tqdm import tqdm

from dreamcan.common import load_config, set_seed
from dreamcan.value import (
    GoalValueNet,
    TransitionDataset,
    ValueFeatures,
    collate_transitions,
    expectile_loss,
    goal_reward,
    sample_goals,
    split_episodes,
)


def td_loss(model, batch, features, ep_lengths, cfg, accelerator):
    goal_ep, goal_t = sample_goals(batch["ep"], batch["t"], batch["n"], ep_lengths, cfg["gamma"])
    goal = features.goals(goal_ep.cpu(), goal_t.cpu()).to(batch["state"].device, non_blocking=True)
    with accelerator.autocast():
        v = model(batch["state"], goal, batch["lang_ids"], batch["lang_lengths"])
        with torch.no_grad():
            v_next = model(batch["next_state"], goal, batch["lang_ids"], batch["lang_lengths"], target=True)
    y = goal_reward(batch["ep"], batch["t"], goal_ep, goal_t) + cfg["gamma"] * (1.0 - batch["terminal"]) * v_next
    return expectile_loss(y - v, cfg["expectile"])


def make_loader(features, cfg, shuffle):
    return DataLoader(
        TransitionDataset(features),
        batch_size=cfg["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["num_workers"],
        collate_fn=collate_transitions,
        prefetch_factor=1 if cfg["num_workers"] > 0 else None,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)

    accelerator = Accelerator(
        mixed_precision="bf16", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    set_seed(cfg["seed"] + accelerator.process_index)
    device = accelerator.device

    train_keys, val_keys = split_episodes(cfg["data"], cfg["val_fraction"], cfg["seed"])
    history_len = cfg["model"]["history_len"]
    train_features = ValueFeatures(cfg["data"], cfg["features"], train_keys, history_len)
    val_features = ValueFeatures(cfg["data"], cfg["features"], val_keys, history_len)
    with h5py.File(cfg["features"], "r") as f:
        g = f["episodes"][train_keys[0]]
        frame_dim = g["local"].shape[2] * g["local"].shape[3] + g["global"].shape[2]

    model = GoalValueNet(frame_dim, **cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]

    loader = make_loader(train_features, cfg, shuffle=True)
    val_loader = make_loader(val_features, cfg, shuffle=False)
    model, opt, loader = accelerator.prepare(model, opt, loader)
    raw = accelerator.unwrap_model(model)
    train_lengths = torch.tensor(train_features.lengths, device=device)
    val_lengths = torch.tensor(val_features.lengths, device=device)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    def batches():
        while True:
            yield from loader

    it = batches()
    pbar = tqdm(range(step + 1, cfg["steps"] + 1), disable=not accelerator.is_main_process)
    model.train()
    for step in pbar:
        batch = {k: v.to(device, non_blocking=True) for k, v in next(it).items()}
        loss = td_loss(model, batch, train_features, train_lengths, cfg, accelerator)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        opt.zero_grad(set_to_none=True)
        raw.update_target(cfg["target_tau"])

        if accelerator.is_main_process and step % cfg["log_every"] == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    val = [
                        td_loss(raw, {k: v.to(device) for k, v in b.items()}, val_features, val_lengths, cfg, accelerator).item()
                        for b in val_loader
                    ]
                model.train()
                print(f"step {step} train_loss {loss.item():.4f} val_loss {sum(val) / max(1, len(val)):.4f}", flush=True)
                torch.save(
                    {"model": raw.state_dict(), "optimizer": opt.state_dict(), "step": step, "config": cfg},
                    os.path.join(cfg["output_dir"], f"value_{step:06d}.pt"),
                )
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
