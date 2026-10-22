import argparse

import clip
import h5py
import numpy as np
import torch
from clip.model import build_model

from dreamcan.common import episode_keys, instruction_of


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with h5py.File(args.data, "r") as f:
        texts = list(dict.fromkeys(instruction_of(f["episodes"][k]) for k in episode_keys(f)))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    state = {k[len("model.") :] if k.startswith("model.") else k: v for k, v in state.items()}
    model = build_model(state).float().to(args.device).eval()
    feats = model.encode_text(clip.tokenize(texts, truncate=True).to(args.device)).float()
    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    np.savez(args.out, texts=np.asarray(texts), embeddings=feats.cpu().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
