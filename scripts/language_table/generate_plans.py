import argparse
import json
import os

import imageio
import imageio.v3 as iio
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image

from dreamcan.wan import TIME_COMPRESSION, generate_video, load_wan

SIZE = (320, 180)


def write_plan(path, frames):
    frames = [f if f.shape[:2] == SIZE[::-1] else np.asarray(Image.fromarray(f).resize(SIZE, Image.BILINEAR)) for f in frames]
    imageio.mimwrite(path, frames, fps=4, quality=10, macro_block_size=1)


def worker(rank, gpus, args, keys, episodes):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    pipe = load_wan(args.wan_root, torch.device("cuda:0"), lora=args.lora, lora_rank=args.lora_rank).pipe
    pipe.dit.eval()
    idx = [TIME_COMPRESSION * i for i in range(1, args.num_subgoals + 1)]
    for key in keys[rank :: len(gpus)]:
        out = os.path.join(args.out, f"{key}.mp4")
        if os.path.exists(out):
            continue
        first = iio.imread(os.path.join(args.data, "videos", f"{key}.mp4"), index=0)
        cond = Image.fromarray(first).resize((args.width, args.height), Image.BICUBIC)
        video = generate_video(pipe, cond, episodes[key], args.seed, args.num_subgoals, args.steps, args.height, args.width)
        video = ((video.float().cpu().clamp(-1, 1) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).numpy()
        write_plan(out, [np.asarray(cond)] + [video[i] for i in idx])
        print(f"[gpu {gpus[rank]}] {key}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=("test", "calibration"))
    p.add_argument("--lora", default=None)
    p.add_argument("--wan_root", default=None)
    p.add_argument("--subgoals", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--num_subgoals", type=int, default=16)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--height", type=int, default=192)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.data, "episodes.json")) as f:
        episodes = json.load(f)
    with open(os.path.join(args.data, "splits.json")) as f:
        keys = json.load(f)[args.split]
    if args.subgoals:
        with open(args.subgoals) as f:
            subgoals = json.load(f)
        for key in keys:
            video = iio.imread(os.path.join(args.data, "videos", f"{key}.mp4"))
            write_plan(os.path.join(args.out, f"{key}.mp4"), [video[0]] + [video[t] for t in subgoals[key]])
        return
    gpus = [int(g) for g in args.gpus.split(",")]
    mp.spawn(worker, args=(gpus, args, keys, episodes), nprocs=len(gpus))


if __name__ == "__main__":
    main()
