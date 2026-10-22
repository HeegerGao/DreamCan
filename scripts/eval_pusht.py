import argparse
import json
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image

from dreamcan.common import IMAGENET_MEAN, IMAGENET_STD, denormalize, load_radio, normalize, radio_encode
from dreamcan.policy import GoalPolicy
from dreamcan.pusht_env import PushTEnv
from dreamcan.wan import generate_subgoals, load_wan

INSTRUCTION = "push the T-shaped block to match the green target"


class Planner:
    def __init__(self, args, device):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt["stats"].items()}
        self.policy = GoalPolicy(None, resnet_pretrained=False, **ckpt["config"]).to(device).eval()
        self.policy.load_state_dict(ckpt["model"])
        self.radio = load_radio(args.radio, device)
        self.pipe = load_wan(args.wan_root, device, lora=args.lora).pipe
        self.pipe.dit.eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
        self.size = (args.radio_size, args.radio_size)
        self.args = args
        self.device = device

    @torch.no_grad()
    def observe(self, frame):
        x = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).to(self.device)
        small = F.interpolate(x.unsqueeze(0), size=self.size, mode="bilinear", align_corners=False)
        radio_local, _ = radio_encode(self.radio, small, 8)
        return (x - self.mean) / self.std, small[0], radio_local[0]

    @torch.no_grad()
    def plan(self, small, seed):
        cond = Image.fromarray((small.clamp(0.0, 1.0) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy())
        goals = generate_subgoals(self.pipe, cond, INSTRUCTION, seed, self.args.num_subgoals, self.args.wan_steps)
        goals = F.interpolate(goals[: self.policy.goal_sequence_len], size=self.size, mode="bilinear", align_corners=False)
        return radio_encode(self.radio, goals.to(self.device), 8)

    @torch.no_grad()
    def act(self, image, radio_local, state, goals):
        l, v = self.policy.history_len, self.policy.num_views
        images = image.expand(1, l, v, *image.shape)
        radio_local = radio_local.expand(1, l, v, *radio_local.shape)
        s = normalize(state, self.stats["state_min"], self.stats["state_max"])
        state = torch.from_numpy(np.concatenate([s, s])).unsqueeze(0).to(self.device)
        task = torch.zeros(1, dtype=torch.long, device=self.device)
        goal_local, goal_global = goals
        chunk = self.policy.sample(
            images, radio_local, state, task, goal_local.unsqueeze(0), goal_global.unsqueeze(0), steps=self.args.flow_steps
        )
        return denormalize(chunk[0].cpu().numpy(), self.stats["action_min"], self.stats["action_max"])


def run_episode(planner, env, seed, args, record):
    env.seed(seed)
    frame = env.reset()
    t, peak, goals = 0, 0.0, None
    while t < args.max_steps:
        image, small, radio_local = planner.observe(frame)
        if t % args.replan_every == 0:
            goals = planner.plan(small, seed)
        actions = planner.act(image, radio_local, env.state(), goals)
        for s in range(min(args.exec_steps, args.max_steps - t)):
            record(frame)
            frame, coverage, done = env.step(actions[s].astype(np.float64).tolist())
            peak = max(peak, coverage)
            t += 1
            if done:
                return True, peak, t
    return False, peak, t


def worker(rank, gpus, args, trials):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    device = torch.device("cuda:0")
    planner = Planner(args, device)
    results = []
    for trial in trials[rank :: len(gpus)]:
        seed = args.seed + trial
        env = PushTEnv(render_size=args.render_size, success_threshold=args.success_threshold)
        frames = []
        record = frames.append if args.video_dir else (lambda frame: None)
        success, peak, steps = run_episode(planner, env, seed, args, record)
        results.append({"trial": trial, "success": success, "coverage": peak, "steps": steps})
        print(f"[gpu {gpus[rank]}] trial {trial} success {success} coverage {peak:.3f}", flush=True)
        if args.video_dir:
            imageio.mimwrite(os.path.join(args.video_dir, f"trial{trial:03d}_{'success' if success else 'fail'}.mp4"), frames, fps=10)
    with open(os.path.join(args.out_dir, f"part_{rank}.json"), "w") as f:
        json.dump(results, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--wan_root", required=True)
    p.add_argument("--radio", default="c-radio_v3-h")
    p.add_argument("--out_dir", default="eval/pusht")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--replan_every", type=int, default=16)
    p.add_argument("--exec_steps", type=int, default=8)
    p.add_argument("--num_subgoals", type=int, default=4)
    p.add_argument("--flow_steps", type=int, default=20)
    p.add_argument("--wan_steps", type=int, default=30)
    p.add_argument("--render_size", type=int, default=224)
    p.add_argument("--radio_size", type=int, default=96)
    p.add_argument("--success_threshold", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=100000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    mp.spawn(worker, args=(gpus, args, list(range(args.trials))), nprocs=len(gpus))

    results = []
    for rank in range(len(gpus)):
        with open(os.path.join(args.out_dir, f"part_{rank}.json")) as f:
            results += json.load(f)
        os.remove(os.path.join(args.out_dir, f"part_{rank}.json"))
    results.sort(key=lambda r: r["trial"])
    summary = {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_coverage": float(np.mean([r["coverage"] for r in results])),
        "episodes": results,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("success_rate", "mean_coverage")}, indent=1))


if __name__ == "__main__":
    main()
