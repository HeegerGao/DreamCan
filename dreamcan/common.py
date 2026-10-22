import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LANG_PAD = 256
LANG_VOCAB = 257


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def episode_keys(h5):
    return sorted(k for k in h5["episodes"].keys() if k.startswith("ep_"))


def instruction_of(group):
    raw = group.attrs["language_instruction"]
    return raw.decode() if isinstance(raw, bytes) else str(raw)


def load_stats(path):
    with open(path) as f:
        return {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}


def compute_stats(h5):
    keys = episode_keys(h5)
    state = np.concatenate([h5["episodes"][k]["state"][:] for k in keys])
    joint = np.concatenate([h5["episodes"][k]["joint_state"][:] for k in keys])
    action = np.concatenate([h5["episodes"][k]["actions"][:] for k in keys])
    return {
        "state_min": state.min(0).astype(np.float32),
        "state_max": state.max(0).astype(np.float32),
        "joint_state_min": joint.min(0).astype(np.float32),
        "joint_state_max": joint.max(0).astype(np.float32),
        "action_min": np.percentile(action, 1.0, axis=0).astype(np.float32),
        "action_max": np.percentile(action, 99.0, axis=0).astype(np.float32),
    }


def normalize(x, lo, hi):
    r = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    return (2.0 * (x - lo) / r - 1.0).astype(np.float32)


def denormalize(x, lo, hi):
    r = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    return ((x + 1.0) / 2.0 * r + lo).astype(np.float32)


def resize_uint8(imgs, size):
    if tuple(imgs.shape[1:3]) == tuple(size):
        return imgs
    t = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=tuple(size), mode="bilinear", align_corners=False)
    return np.clip(t.permute(0, 2, 3, 1).numpy(), 0, 255).astype(np.uint8)


def imagenet_normalize(imgs):
    x = (imgs.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def language_tokens(texts, max_chars=256):
    ids = torch.full((len(texts), max_chars), LANG_PAD, dtype=torch.long)
    lengths = torch.zeros(len(texts), dtype=torch.long)
    for i, s in enumerate(texts):
        raw = s.encode("utf-8", errors="ignore")[:max_chars] or b" "
        ids[i, : len(raw)] = torch.tensor(list(raw), dtype=torch.long)
        lengths[i] = len(raw)
    return ids, lengths


def load_radio(version, device):
    model = torch.hub.load(
        "NVlabs/RADIO", "radio_model", version=version, progress=True, skip_validation=True, trust_repo=True
    )
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.inference_mode()
def radio_encode(radio, x, microbatch=8):
    local, summary = [], []
    for s in range(0, x.shape[0], microbatch):
        chunk = x[s : s + microbatch].clamp(0.0, 1.0)
        res = radio.get_nearest_supported_resolution(int(chunk.shape[-2]), int(chunk.shape[-1]))
        if tuple(chunk.shape[-2:]) != (int(res.height), int(res.width)):
            chunk = F.interpolate(chunk, size=(int(res.height), int(res.width)), mode="bilinear", align_corners=False)
        out = radio(chunk, feature_fmt="NLC")
        if isinstance(out, dict):
            out = out["backbone"]
        local.append(out.features.float())
        summary.append(out.summary.float())
    return torch.cat(local), torch.cat(summary)


def uint8_to_unit(imgs, size, device):
    x = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2).float().div_(255.0).to(device)
    if tuple(x.shape[-2:]) != tuple(size):
        x = F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)
    return x.clamp_(0.0, 1.0)
