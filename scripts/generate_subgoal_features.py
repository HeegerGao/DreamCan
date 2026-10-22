import argparse
import os

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from dreamcan.common import episode_keys, instruction_of, load_radio, radio_encode, resize_uint8
from dreamcan.wan import generate_subgoals, load_wan


def sample_index(data_path):
    with h5py.File(data_path, "r") as f:
        keys = episode_keys(f)
        return [(k, t) for k in keys for t in range(f["episodes"][k]["actions"].shape[0])]


def worker(rank, gpus, args):
    device = torch.device(f"cuda:{gpus[rank]}")
    torch.cuda.set_device(device)
    pipe = load_wan(args.wan_root, device, lora=args.lora).pipe
    pipe.dit.eval()
    radio = load_radio(args.radio, device)
    local = np.load(os.path.join(args.out, "local.npy"), mmap_mode="r+")
    summary = np.load(os.path.join(args.out, "global.npy"), mmap_mode="r+")
    done = np.load(os.path.join(args.out, "done.npy"), mmap_mode="r+")
    index = sample_index(args.data)
    todo = [i for i in range(rank, len(index), len(gpus)) if not done[i]]
    with h5py.File(args.data, "r") as f:
        for count, i in enumerate(tqdm(todo, position=rank)):
            key, t = index[i]
            g = f["episodes"][key]
            cond = Image.fromarray(resize_uint8(g["image"][t : t + 1], (args.size, args.size))[0])
            goals = generate_subgoals(pipe, cond, instruction_of(g), args.seed, args.num_subgoals, args.steps)[: args.goal_len]
            goals = F.interpolate(goals, size=(args.size, args.size), mode="bilinear", align_corners=False)
            lo, gl = radio_encode(radio, goals.to(device), 8)
            local[i] = lo.cpu().numpy().astype(np.float16)
            summary[i] = gl.cpu().numpy().astype(np.float16)
            done[i] = 1
            if count % 50 == 0:
                local.flush()
                summary.flush()
                done.flush()
    local.flush()
    summary.flush()
    done.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--wan_root", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--radio", default="c-radio_v3-h")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--num_subgoals", type=int, default=10)
    p.add_argument("--goal_len", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n = len(sample_index(args.data))
    os.makedirs(args.out, exist_ok=True)
    tokens = (args.size // 16) ** 2
    shapes = {"local": (n, args.goal_len, tokens, 1280), "global": (n, args.goal_len, 3840), "done": (n,)}
    for name, shape in shapes.items():
        path = os.path.join(args.out, f"{name}.npy")
        if not os.path.exists(path):
            dtype = np.uint8 if name == "done" else np.float16
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
    gpus = [int(g) for g in args.gpus.split(",")]
    mp.spawn(worker, args=(gpus, args), nprocs=len(gpus))


if __name__ == "__main__":
    main()
