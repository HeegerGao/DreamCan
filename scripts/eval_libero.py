import argparse
import json
import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image

from dreamcan.common import IMAGENET_MEAN, IMAGENET_STD, denormalize, load_radio, normalize, radio_encode
from dreamcan.policy import GoalPolicy
from dreamcan.wan import generate_subgoals, load_wan

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def quat2axisangle(q):
    w = max(-1.0, min(1.0, float(q[3])))
    den = math.sqrt(max(0.0, 1.0 - w * w))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return np.asarray(q[:3]) * 2.0 * math.acos(w) / den


def proprio(obs, stats):
    state = np.concatenate(
        [obs["robot0_eef_pos"], quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float64)), obs["robot0_gripper_qpos"][:2]]
    ).astype(np.float32)
    joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    return np.concatenate(
        [normalize(state, stats["state_min"], stats["state_max"]), normalize(joint, stats["joint_state_min"], stats["joint_state_max"])]
    )


class Planner:
    def __init__(self, args, device):
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        self.instructions = ckpt["instructions"]
        self.stats = {k: np.asarray(v, dtype=np.float32) for k, v in ckpt["stats"].items()}
        table = torch.zeros(len(self.instructions), ckpt["model"]["language_encoder.table"].shape[1])
        self.policy = GoalPolicy(table, resnet_pretrained=False, **ckpt["config"]).to(device).eval()
        self.policy.load_state_dict(ckpt["model"])
        self.radio = load_radio(args.radio, device)
        self.pipe = load_wan(args.wan_root, device, lora=args.lora).pipe
        self.pipe.dit.eval()
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
        self.args = args
        self.device = device

    def preprocess(self, img):
        x = torch.from_numpy(np.ascontiguousarray(img[::-1, ::-1])).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)[0]
        return (x / 255.0 - self.mean) / self.std

    def unit(self, x):
        return (x * self.std + self.mean).clamp(0.0, 1.0)

    @torch.no_grad()
    def act(self, hist_main, hist_wrist, obs, instruction, seed):
        cond = (self.unit(hist_main[-1]) * 255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        goals = generate_subgoals(self.pipe, Image.fromarray(cond), instruction, seed, steps=self.args.wan_steps)
        goals = F.interpolate(goals, size=(224, 224), mode="bilinear", align_corners=False)
        goal_local, goal_global = radio_encode(self.radio, goals.to(self.device), 8)

        l = self.policy.history_len
        views = [torch.stack(([h[0]] * max(0, l - len(h)) + h)[-l:]) for h in (hist_main, hist_wrist)]
        images = torch.stack(views, dim=1).unsqueeze(0)
        radio_local, _ = radio_encode(self.radio, self.unit(images.reshape(-1, *images.shape[3:])), 8)
        state = torch.from_numpy(proprio(obs, self.stats)).unsqueeze(0).to(self.device)
        task = torch.tensor([self.instructions.index(instruction)], device=self.device)
        chunk = self.policy.sample(
            images,
            radio_local.reshape(1, l, len(views), *radio_local.shape[1:]),
            state,
            task,
            goal_local.unsqueeze(0),
            goal_global.unsqueeze(0),
            steps=self.args.flow_steps,
        )
        return denormalize(chunk[0].cpu().numpy(), self.stats["action_min"], self.stats["action_max"])


def run_episode(planner, env, init_state, instruction, seed, args, record):
    env.reset()
    obs = env.set_init_state(init_state)
    hist_main, hist_wrist = [], []
    t, horizon = 0, args.max_steps + args.wait_steps
    while t < horizon:
        if t < args.wait_steps:
            record(obs["agentview_image"][::-1, ::-1])
            obs, _, _, _ = env.step(DUMMY_ACTION)
            t += 1
            continue
        hist_main.append(planner.preprocess(obs["agentview_image"]))
        hist_wrist.append(planner.preprocess(obs["robot0_eye_in_hand_image"]))
        actions = planner.act(hist_main, hist_wrist, obs, instruction, seed)
        for s in range(min(args.exec_steps, horizon - t)):
            if s > 0:
                hist_main.append(planner.preprocess(obs["agentview_image"]))
                hist_wrist.append(planner.preprocess(obs["robot0_eye_in_hand_image"]))
            record(obs["agentview_image"][::-1, ::-1])
            obs, _, _, _ = env.step(actions[s].tolist())
            t += 1
            if env.check_success():
                return True, t
    return False, t


def worker(rank, gpus, args, episodes):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    device = torch.device("cuda:0")
    planner = Planner(args, device)
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    results, env, env_task = [], None, None
    for task_id, trial in episodes[rank :: len(gpus)]:
        task = suite.get_task(task_id)
        if env_task != task_id:
            if env is not None:
                env.close()
            env = OffScreenRenderEnv(
                bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
                camera_heights=256,
                camera_widths=256,
                horizon=args.max_steps + args.wait_steps + 50,
            )
            env.seed(args.seed + task_id)
            env_task = task_id
        init_states = suite.get_task_init_states(task_id)
        seed = args.seed + 1000 * task_id + trial
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        frames = []
        record = frames.append if args.video_dir else (lambda frame: None)
        success, steps = run_episode(planner, env, init_states[trial % len(init_states)], task.language, seed, args, record)
        results.append({"task_id": task_id, "trial": trial, "success": success, "steps": steps})
        print(f"[gpu {gpus[rank]}] task {task_id} trial {trial} success {success}", flush=True)
        if args.video_dir:
            imageio.mimwrite(os.path.join(args.video_dir, f"task{task_id:02d}_trial{trial:02d}_{'success' if success else 'fail'}.mp4"), frames, fps=20)
    if env is not None:
        env.close()
    with open(os.path.join(args.out_dir, f"part_{rank}.json"), "w") as f:
        json.dump(results, f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--wan_root", required=True)
    p.add_argument("--radio", default="c-radio_v3-h")
    p.add_argument("--out_dir", default="eval")
    p.add_argument("--video_dir", default=None)
    p.add_argument("--gpus", default="0")
    p.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--wait_steps", type=int, default=5)
    p.add_argument("--exec_steps", type=int, default=8)
    p.add_argument("--flow_steps", type=int, default=20)
    p.add_argument("--wan_steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.video_dir:
        os.makedirs(args.video_dir, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    episodes = [(int(task), trial) for task in args.tasks.split(",") for trial in range(args.trials)]
    mp.spawn(worker, args=(gpus, args, episodes), nprocs=len(gpus))

    results = []
    for rank in range(len(gpus)):
        with open(os.path.join(args.out_dir, f"part_{rank}.json")) as f:
            results += json.load(f)
        os.remove(os.path.join(args.out_dir, f"part_{rank}.json"))
    results.sort(key=lambda r: (r["task_id"], r["trial"]))
    per_task = {}
    for r in results:
        per_task.setdefault(r["task_id"], []).append(r["success"])
    summary = {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "per_task": {t: float(np.mean(s)) for t, s in per_task.items()},
        "episodes": results,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"success_rate": summary["success_rate"], "per_task": summary["per_task"]}, indent=1))


if __name__ == "__main__":
    main()
