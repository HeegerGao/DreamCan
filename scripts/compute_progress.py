import argparse
import json
import sys

import h5py
import numpy as np
import torch
from tqdm import tqdm

from dreamcan.common import episode_keys, instruction_of


def sample_indices(n, stride, max_frames):
    m = max(1, min(int(round(n / stride)), n, max_frames))
    return np.arange(n) if m == n else np.linspace(0, n - 1, m, dtype=int)


@torch.no_grad()
def predict_progress(rm, frames, task, batch_size, prefix_frames, device):
    model, tokenizer, collator, exp_config = rm
    loss_cfg = getattr(exp_config, "loss", None)
    discrete = loss_cfg is not None and getattr(loss_cfg, "progress_loss_type", "l2").lower() == "discrete"
    bins = getattr(loss_cfg, "progress_discrete_bins", None) or getattr(exp_config.model, "progress_discrete_bins", 10)
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs

    samples = []
    for end in range(1, len(frames) + 1):
        sub = frames[np.linspace(0, end - 1, prefix_frames, dtype=int)]
        traj = Trajectory(
            frames=sub,
            frames_shape=tuple(sub.shape),
            task=task,
            id=f"prefix_{end:04d}",
            metadata={"subsequence_length": int(sub.shape[0])},
            video_embeddings=None,
        )
        samples.append(ProgressSample(trajectory=traj, sample_type="progress"))
    out = []
    for s in range(0, len(samples), batch_size):
        inputs = collator(samples[s : s + batch_size])["progress_inputs"]
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        res = compute_batch_outputs(model, tokenizer, inputs, sample_type="progress", is_discrete_mode=discrete, num_bins=bins)
        out += [float(p[-1]) for p in res["progress_pred"]]
    return np.asarray(out, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--robometer_repo", required=True)
    p.add_argument("--model", default="robometer/Robometer-4B")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--max_frames", type=int, default=512)
    p.add_argument("--prefix_frames", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    sys.path.insert(0, args.robometer_repo)
    from robometer.utils.save import load_model_from_hf
    from robometer.utils.setup_utils import setup_batch_collator

    device = torch.device(args.device)
    exp_config, tokenizer, processor, model = load_model_from_hf(args.model, device=device)
    model.eval()
    rm = (model, tokenizer, setup_batch_collator(processor, tokenizer, exp_config, is_eval=True), exp_config)

    progress = {}
    with h5py.File(args.data, "r") as f:
        for key in tqdm(episode_keys(f)):
            g = f["episodes"][key]
            n = g["image"].shape[0]
            idx = sample_indices(n, args.stride, args.max_frames)
            frames = g["image"][:][idx]
            pred = predict_progress(rm, frames, instruction_of(g), args.batch_size, args.prefix_frames, device)
            progress[key] = np.interp(np.arange(n), idx, pred).tolist()
            torch.cuda.empty_cache()
    with open(args.out, "w") as f:
        json.dump(progress, f)


if __name__ == "__main__":
    main()
