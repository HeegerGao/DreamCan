import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dreamcan.common import episode_keys, imagenet_normalize, instruction_of, normalize, resize_uint8

def read_rows(ds, idx):
    lo, hi = int(idx.min()), int(idx.max())
    return ds[lo : hi + 1][idx - lo]


def crop_images(imgs, box):
    if box is None:
        return imgs
    h, w = imgs.shape[1:3]
    top, left, bh, bw = box
    ch, cw = max(1, int(round(bh * h))), max(1, int(round(bw * w)))
    t0, l0 = min(int(round(top * h)), h - ch), min(int(round(left * w)), w - cw)
    x = torch.from_numpy(np.ascontiguousarray(imgs[:, t0 : t0 + ch, l0 : l0 + cw])).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return np.clip(x.permute(0, 2, 3, 1).numpy(), 0, 255).astype(np.uint8)


def crop_patches(feat, box):
    if box is None:
        return feat
    lead, (p, c) = feat.shape[:-2], feat.shape[-2:]
    g = int(round(p**0.5))
    x = torch.from_numpy(np.ascontiguousarray(feat)).float().reshape(-1, g, g, c).permute(0, 3, 1, 2)
    top, left, bh, bw = box
    ch, cw = max(1, int(round(bh * g))), max(1, int(round(bw * g)))
    t0, l0 = min(int(round(top * g)), g - ch), min(int(round(left * g)), g - cw)
    x = F.interpolate(x[:, :, t0 : t0 + ch, l0 : l0 + cw], size=(g, g), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(*lead, p, c).numpy().astype(feat.dtype)


class PolicyDataset(Dataset):
    def __init__(
        self,
        data_path,
        radio_path,
        goal_dir,
        stats,
        instructions=None,
        views=("image", "wrist_image"),
        state_keys=("state", "joint_state"),
        history_len=2,
        chunk_len=16,
        image_size=224,
        crop_prob=0.8,
        crop_scale=(0.7, 1.0),
        state_noise=0.02,
    ):
        self.data_path = data_path
        self.radio_path = radio_path
        self.goal_dir = goal_dir
        self.stats = stats
        self.views = tuple(views)
        self.state_keys = tuple(state_keys)
        self.history_len = history_len
        self.chunk_len = chunk_len
        self.image_size = (image_size, image_size)
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.state_noise = state_noise
        task_of = {s: i for i, s in enumerate(instructions or [])}
        with h5py.File(data_path, "r") as f:
            self.keys = episode_keys(f)
            self.lengths = [f["episodes"][k]["actions"].shape[0] for k in self.keys]
            self.tasks = [task_of.get(instruction_of(f["episodes"][k]), 0) for k in self.keys]
        self.index = [(e, t) for e, n in enumerate(self.lengths) for t in range(n)]
        self._files = None

    def __len__(self):
        return len(self.index)

    def open(self):
        self._files = (
            h5py.File(self.data_path, "r"),
            h5py.File(self.radio_path, "r"),
            np.load(os.path.join(self.goal_dir, "local.npy"), mmap_mode="r"),
            np.load(os.path.join(self.goal_dir, "global.npy"), mmap_mode="r"),
        )

    def crop_box(self):
        if self.crop_prob <= 0.0 or np.random.rand() >= self.crop_prob:
            return None
        side = float(np.sqrt(np.random.uniform(*self.crop_scale)))
        return float(np.random.uniform(0.0, 1.0 - side)), float(np.random.uniform(0.0, 1.0 - side)), side, side

    def __getitem__(self, i):
        if self._files is None:
            self.open()
        data, radio, goal_local, goal_global = self._files
        ep, t = self.index[i]
        n = self.lengths[ep]
        g = data["episodes"][self.keys[ep]]
        hist = np.clip(np.arange(t - self.history_len + 1, t + 1), 0, n - 1)
        box = self.crop_box()

        images = []
        for view in self.views:
            frames = crop_images(resize_uint8(read_rows(g[view], hist), self.image_size), box)
            images.append(imagenet_normalize(frames))
        state = []
        for key in self.state_keys:
            x = normalize(read_rows(g[key], hist), self.stats[f"{key}_min"], self.stats[f"{key}_max"])
            if self.state_noise > 0.0:
                x = x + np.random.normal(0.0, self.state_noise, size=x.shape).astype(x.dtype)
            state.append(x[-1])
        chunk = np.clip(np.arange(t, t + self.chunk_len), 0, n - 1)
        actions = normalize(read_rows(g["actions"], chunk), self.stats["action_min"], self.stats["action_max"])

        return {
            "images": torch.from_numpy(np.stack(images, axis=1)),
            "radio_local": torch.from_numpy(crop_patches(read_rows(radio["episodes"][self.keys[ep]]["local"], hist), box)),
            "state": torch.from_numpy(np.concatenate(state).astype(np.float32)),
            "task": self.tasks[ep],
            "goal_local": torch.from_numpy(crop_patches(np.array(goal_local[i]), box)),
            "goal_global": torch.from_numpy(np.array(goal_global[i])),
            "actions": torch.from_numpy(actions),
        }
