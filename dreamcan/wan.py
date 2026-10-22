import glob
import importlib.util
import json
import os

import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import Dataset

from dreamcan.common import episode_keys, instruction_of, resize_uint8

TIME_COMPRESSION = 4
NUM_SUBGOALS = 10


def anchored_forward(self, data, inputs=None):
    pipe = self.pipe
    shared, posi, nega = self.get_pipeline_inputs(data)
    shared, posi, nega = self.transfer_data_to_device((shared, posi, nega), pipe.device, pipe.torch_dtype)
    for unit in pipe.units:
        shared, posi, nega = pipe.unit_runner(unit, pipe, shared, posi, nega)
    latents = shared["input_latents"]
    n = latents.shape[2]
    prefix = 0 if torch.rand(()) < self.anchor_p_plain else int(torch.randint(1, min(self.anchor_max_prefix, n - 2) + 1, ()).item())
    clean = torch.zeros(n, dtype=torch.bool, device=pipe.device)
    clean[: prefix + 1] = True
    timestep = pipe.scheduler.timesteps[torch.randint(0, len(pipe.scheduler.timesteps), (1,))].to(dtype=pipe.torch_dtype, device=pipe.device)
    noise = torch.randn_like(latents)
    noised = pipe.scheduler.add_noise(latents, noise, timestep)
    target = pipe.scheduler.training_target(latents, noise, timestep)
    noised[:, :, : prefix + 1] = latents[:, :, : prefix + 1]
    pred = pipe.model_fn(
        **{name: getattr(pipe, name) for name in pipe.in_iteration_models},
        latents=noised,
        timestep=timestep,
        context=posi["context"],
        clean_frame_mask=clean,
        fuse_vae_embedding_in_latents=True,
        use_gradient_checkpointing=self.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload,
    )
    loss = ((pred[:, :, ~clean].float() - target[:, :, ~clean].float()) ** 2).mean()
    return loss * pipe.scheduler.training_weight(timestep)


def load_wan(
    model_root,
    device,
    lora=None,
    tokenizer=None,
    lora_rank=32,
    lora_targets="q,k,v,o,ffn.0,ffn.2",
    anchor_p_plain=None,
    anchor_max_prefix=12,
):
    train_py = os.path.join(os.environ["DIFFSYNTH_ROOT"], "examples", "wanvideo", "model_training", "train.py")
    spec = importlib.util.spec_from_file_location("diffsynth_wan_train", train_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.WanTrainingModule
    if anchor_p_plain is not None:
        attrs = {"forward": anchored_forward, "anchor_p_plain": anchor_p_plain, "anchor_max_prefix": anchor_max_prefix}
        cls = type("AnchoredWanTrainingModule", (cls,), attrs)
    dit = sorted(glob.glob(os.path.join(model_root, "diffusion_pytorch_model*.safetensors")))
    paths = [dit, os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"), os.path.join(model_root, "Wan2.2_VAE.pth")]
    return cls(
        model_paths=json.dumps(paths),
        tokenizer_path=tokenizer or os.path.join(model_root, "google", "umt5-xxl"),
        trainable_models=None,
        lora_base_model="dit",
        lora_target_modules=lora_targets,
        lora_rank=lora_rank,
        lora_checkpoint=lora,
        use_gradient_checkpointing=True,
        extra_inputs=None if anchor_p_plain is not None else "input_image",
        task="sft",
        device=str(device),
    )


def save_lora(accelerator, model, path):
    accelerator.wait_for_everyone()
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        state = accelerator.unwrap_model(model).export_trainable_state_dict(state, remove_prefix="pipe.dit.")
        save_file({k: v.contiguous() for k, v in state.items()}, path)


@torch.no_grad()
def generate_video(pipe, image, prompt, seed, num_subgoals, steps, height, width):
    video = pipe(
        prompt=prompt,
        negative_prompt="",
        input_image=image,
        height=height,
        width=width,
        num_frames=1 + TIME_COMPRESSION * num_subgoals,
        cfg_scale=1.0,
        num_inference_steps=steps,
        seed=seed,
        tiled=False,
        output_type="floatpoint",
    )
    return video[0] if video.dim() == 5 else video


def generate_subgoals(pipe, image, prompt, seed, num_subgoals=NUM_SUBGOALS, steps=30, size=256):
    video = generate_video(pipe, image, prompt, seed, num_subgoals, steps, size, size)
    idx = [TIME_COMPRESSION * i for i in range(1, num_subgoals + 1)]
    return (video[:, idx].permute(1, 0, 2, 3).float().clamp(-1.0, 1.0) + 1.0) * 0.5


class SubgoalVideoDataset(Dataset):
    def __init__(self, data_path, subgoals_path, num_subgoals=NUM_SUBGOALS, size=256, prompt_dropout=0.0):
        self.data_path = data_path
        self.num_subgoals = num_subgoals
        self.size = (size, size)
        self.prompt_dropout = prompt_dropout
        with open(subgoals_path) as f:
            subgoals = json.load(f)
        with h5py.File(data_path, "r") as f:
            self.keys = episode_keys(f)
            self.instructions = [instruction_of(f["episodes"][k]) for k in self.keys]
            lengths = [f["episodes"][k]["actions"].shape[0] for k in self.keys]
        self.subgoals = [subgoals[k] for k in self.keys]
        self.index = [(e, t) for e, n in enumerate(lengths) for t in range(n)]
        self._h5 = None

    def __len__(self):
        return len(self.index)

    def future(self, ep, t):
        remaining = [s for s in self.subgoals[ep] if s > t] or [self.subgoals[ep][-1]]
        remaining = remaining[: self.num_subgoals]
        return remaining + [remaining[-1]] * (self.num_subgoals - len(remaining))

    def __getitem__(self, i):
        if self._h5 is None:
            self._h5 = h5py.File(self.data_path, "r")
        ep, t = self.index[i]
        images = self._h5["episodes"][self.keys[ep]]["image"]
        future = self.future(ep, t)
        imgs = resize_uint8(np.stack([images[j] for j in [t] + future]), self.size)
        frames = [Image.fromarray(imgs[0])]
        for img in imgs[1:]:
            frames += [Image.fromarray(img)] * TIME_COMPRESSION
        prompt = "" if np.random.rand() < self.prompt_dropout else self.instructions[ep]
        return {"video": frames, "prompt": prompt}


class LTSubgoalVideoDataset(Dataset):
    def __init__(self, data_dir, subgoals_path, num_subgoals=16, height=192, width=320):
        self.data_dir = data_dir
        self.num_subgoals = num_subgoals
        self.size = (width, height)
        with open(os.path.join(data_dir, "episodes.json")) as f:
            episodes = json.load(f)
        with open(subgoals_path) as f:
            subgoals = json.load(f)
        self.keys = sorted(subgoals)
        self.instructions = [episodes[k] for k in self.keys]
        self.subgoals = [subgoals[k] for k in self.keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        video = iio.imread(os.path.join(self.data_dir, "videos", f"{self.keys[i]}.mp4"))
        ts = [min(max(t, 0), len(video) - 1) for t in self.subgoals[i]][: self.num_subgoals]
        ts += [ts[-1]] * (self.num_subgoals - len(ts))
        imgs = [Image.fromarray(video[t]).resize(self.size, Image.BICUBIC) for t in [0] + ts]
        frames = [imgs[0]]
        for img in imgs[1:]:
            frames += [img] * TIME_COMPRESSION
        return {"video": frames, "prompt": self.instructions[i]}
