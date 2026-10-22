import argparse
import os

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dreamcan.common import load_radio, radio_encode
from dreamcan.lt_value import load_episodes


@torch.inference_mode()
def embed_instructions(texts, model_path, device, max_length=256):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()
    out = {}
    for text in tqdm(texts, desc="instructions"):
        tok = tokenizer([text], truncation=True, max_length=max_length, return_tensors="pt").to(device)
        hidden = model(**tok).last_hidden_state.float()
        mask = tok["attention_mask"].float().unsqueeze(-1)
        out[text] = ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0))[0].cpu().numpy().astype(np.float16)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--qwen", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--radio", default="c-radio_v3-h")
    p.add_argument("--frame_batch", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    episodes = load_episodes(os.path.join(args.data, "episodes.json"))
    language = embed_instructions(sorted(set(episodes.values())), args.qwen, device)
    radio = load_radio(args.radio, device)
    with h5py.File(args.out, "w") as dst:
        dst.attrs["radio"] = args.radio
        root = dst.create_group("episodes")
        for key, instruction in tqdm(sorted(episodes.items()), desc="episodes"):
            video = iio.imread(os.path.join(args.data, "videos", f"{key}.mp4"))
            local, summary = [], []
            for s in range(0, len(video), args.frame_batch):
                x = torch.from_numpy(video[s : s + args.frame_batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
                lo, gl = radio_encode(radio, x, args.microbatch)
                local.append(lo.mean(dim=1).cpu().numpy().astype(np.float16))
                summary.append(gl.cpu().numpy().astype(np.float16))
            g = root.create_group(key)
            g.create_dataset("local", data=np.concatenate(local))
            g.create_dataset("global", data=np.concatenate(summary))
            g.create_dataset("language", data=language[instruction])


if __name__ == "__main__":
    main()
