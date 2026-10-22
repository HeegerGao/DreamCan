import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from dreamcan.lt_value import LTFeatures, LTValueNet


@torch.inference_mode()
def value_pairs(model, features, ep, device, block=192):
    n = features.lengths[ep]
    local, global_ = (x.to(device) for x in features.frames(ep, np.arange(n)))
    frames = model.frame(local, global_)
    hist = torch.stack([(torch.arange(n) - 1).clamp(min=0), torch.arange(n)], dim=1).to(device)
    tok = frames[hist] + model.temporal_pos[:, : hist.shape[1]]
    h_s = model.state_out(model.temporal_encoder(tok).mean(dim=1))
    h_g = model.goal_out(frames)
    h_l = model.encode_language(features.language[ep : ep + 1].to(device))
    out = np.full((n, n), np.nan, dtype=np.float32)
    for i in range(0, n, block):
        hs = h_s[i : i + block]
        for j in range(0, n, block):
            hg = h_g[j : j + block]
            a, b = hs.shape[0], hg.shape[0]
            v = model.head(
                hs[:, None].expand(a, b, -1).reshape(a * b, -1),
                hg[None].expand(a, b, -1).reshape(a * b, -1),
                h_l.expand(a * b, -1),
            )
            out[i : i + a, j : j + b] = v.reshape(a, b).float().cpu().numpy()
    out[np.tril_indices(n)] = np.nan
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt["config"]["model"]
    features = LTFeatures(args.features, model_cfg["history_len"])
    model = LTValueNet(features.local_dim, features.global_dim, features.language.shape[1], **model_cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])
    os.makedirs(args.out, exist_ok=True)
    for ep, key in enumerate(tqdm(features.keys)):
        np.savez_compressed(os.path.join(args.out, f"{key}.npz"), v_pair=value_pairs(model, features, ep, device))


if __name__ == "__main__":
    main()
