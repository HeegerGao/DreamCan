import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from dreamcan.common import episode_keys, load_radio, radio_encode, uint8_to_unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--radio", default="c-radio_v3-h")
    p.add_argument("--views", nargs="+", default=["image", "wrist_image"])
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--frame_batch", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    radio = load_radio(args.radio, device)
    size = (args.size, args.size)
    with h5py.File(args.data, "r") as src, h5py.File(args.out, "w") as dst:
        dst.attrs["radio"] = args.radio
        dst.attrs["views"] = ",".join(args.views)
        root = dst.create_group("episodes")
        for key in tqdm(episode_keys(src)):
            g = src["episodes"][key]
            n = g["actions"].shape[0]
            local, summary = [], []
            for s in range(0, n, args.frame_batch):
                e = min(n, s + args.frame_batch)
                frames = np.concatenate([g[v][s:e] for v in args.views])
                lo, gl = radio_encode(radio, uint8_to_unit(frames, size, device), args.microbatch)
                v = len(args.views)
                local.append(lo.reshape(v, e - s, *lo.shape[1:]).transpose(0, 1).cpu().numpy().astype(np.float16))
                summary.append(gl.reshape(v, e - s, -1).transpose(0, 1).cpu().numpy().astype(np.float16))
            local = np.concatenate(local)
            summary = np.concatenate(summary)
            out = root.create_group(key)
            out.create_dataset("local", data=local, chunks=(1, *local.shape[1:]))
            out.create_dataset("global", data=summary, chunks=(1, *summary.shape[1:]))


if __name__ == "__main__":
    main()
