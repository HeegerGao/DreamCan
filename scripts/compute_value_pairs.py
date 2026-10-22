import argparse
import os

import h5py
import numpy as np
import torch
from tqdm import tqdm

from dreamcan.common import episode_keys, language_tokens
from dreamcan.value import GoalValueNet, ValueFeatures


@torch.no_grad()
def value_pairs(model, features, ep, batch_size, device):
    n = features.lengths[ep]
    hist = torch.stack([features.history(ep, t) for t in range(n)]).to(device)
    goals = features.frames(ep, np.arange(n)).to(device)
    ii, jj = np.triu_indices(n, k=1)
    ids, lengths = language_tokens([features.instructions[ep]])
    out = []
    for s in range(0, len(ii), batch_size):
        i = torch.as_tensor(ii[s : s + batch_size], device=device)
        j = torch.as_tensor(jj[s : s + batch_size], device=device)
        b = i.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(hist[i], goals[j], ids.expand(b, -1).to(device), lengths.expand(b).to(device))
        out.append(v.float().cpu().numpy())
    pairs = np.full((n, n), np.nan, dtype=np.float32)
    pairs[ii, jj] = np.concatenate(out)
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    with h5py.File(args.data, "r") as f:
        keys = episode_keys(f)
    model_cfg = ckpt["config"]["model"]
    features = ValueFeatures(args.data, args.features, keys, model_cfg["history_len"])
    frame_dim = ckpt["model"]["state_in_proj.weight"].shape[1]
    model = GoalValueNet(frame_dim, **model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    os.makedirs(args.out, exist_ok=True)
    for ep, key in enumerate(tqdm(keys)):
        np.savez_compressed(os.path.join(args.out, f"{key}.npz"), v_pair=value_pairs(model, features, ep, args.batch_size, device))


if __name__ == "__main__":
    main()
