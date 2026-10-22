import json
import os

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

TASKS = ("line", "all_to_region", "canonical_shape")


def load_episodes(path):
    with open(path) as f:
        return json.load(f)


def _value_head(in_dim, mult):
    return nn.Sequential(nn.Linear(in_dim, in_dim * mult), nn.GELU(), nn.Linear(in_dim * mult, 1))


class LTValueNet(nn.Module):
    def __init__(
        self,
        local_dim,
        global_dim,
        language_dim,
        history_len=2,
        hidden_dim=256,
        token_dim=512,
        temporal_nhead=8,
        temporal_layers=2,
        temporal_dropout=0.1,
        language_hidden_mult=2,
        value_mlp_hidden_mult=2,
    ):
        super().__init__()
        self.history_len = history_len
        self.local_proj = nn.Linear(local_dim, token_dim)
        self.global_proj = nn.Linear(global_dim, token_dim)
        self.frame_norm = nn.LayerNorm(token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=temporal_nhead,
            dim_feedforward=token_dim * 4,
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers, enable_nested_tensor=False)
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, token_dim))
        self.state_out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        self.goal_out = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden_dim))
        mid = language_dim * language_hidden_mult
        self.language_mlp = nn.Sequential(
            nn.LayerNorm(language_dim), nn.Linear(language_dim, mid), nn.GELU(), nn.Linear(mid, hidden_dim)
        )
        self.value_mlp = _value_head(5 * hidden_dim, value_mlp_hidden_mult)
        self.value_mlp_target = _value_head(5 * hidden_dim, value_mlp_hidden_mult)
        self.value_mlp_target.load_state_dict(self.value_mlp.state_dict())

    def frame(self, local, global_):
        return self.frame_norm(self.local_proj(local) + self.global_proj(global_))

    def encode_state(self, local, global_):
        tok = self.frame(local, global_) + self.temporal_pos[:, : local.shape[1]]
        return self.state_out(self.temporal_encoder(tok).mean(dim=1))

    def encode_goal(self, local, global_):
        return self.goal_out(self.frame(local, global_))

    def encode_language(self, language):
        return self.language_mlp(language.float())

    def head(self, h_s, h_g, h_l, target=False):
        fused = torch.cat([h_s, h_g, h_g - h_s, h_s * h_g, h_l], dim=-1)
        return (self.value_mlp_target if target else self.value_mlp)(fused).squeeze(-1)

    def forward(self, state_local, state_global, goal_local, goal_global, language, target=False):
        h_s = self.encode_state(state_local, state_global)
        h_g = self.encode_goal(goal_local, goal_global)
        return self.head(h_s, h_g, self.encode_language(language), target)

    @torch.no_grad()
    def update_target(self, tau):
        for tp, sp in zip(self.value_mlp_target.parameters(), self.value_mlp.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


class LTFeatures:
    def __init__(self, feature_path, history_len=2):
        self.feature_path = feature_path
        self.history_len = history_len
        with h5py.File(feature_path, "r") as f:
            self.keys = sorted(f["episodes"].keys())
            self.lengths = [f["episodes"][k]["local"].shape[0] for k in self.keys]
            self.language = torch.from_numpy(np.stack([f["episodes"][k]["language"][:] for k in self.keys]).astype(np.float32))
            g = f["episodes"][self.keys[0]]
            self.local_dim = g["local"].shape[1]
            self.global_dim = g["global"].shape[1]
        self._h5 = None
        self._pid = None

    def _group(self, ep):
        if self._pid != os.getpid():
            self._h5 = h5py.File(self.feature_path, "r")
            self._pid = os.getpid()
        return self._h5["episodes"][self.keys[ep]]

    def frames(self, ep, ts):
        g = self._group(ep)
        idx = np.asarray(ts, dtype=np.int64)
        lo, hi = int(idx.min()), int(idx.max()) + 1
        local = g["local"][lo:hi][idx - lo]
        global_ = g["global"][lo:hi][idx - lo]
        return torch.from_numpy(local.astype(np.float32)), torch.from_numpy(global_.astype(np.float32))

    def history(self, ep, t):
        return self.frames(ep, np.clip(np.arange(t - self.history_len + 1, t + 1), 0, self.lengths[ep] - 1))

    def goals(self, eps, ts):
        local, global_ = zip(*(self.frames(int(e), [int(t)]) for e, t in zip(eps.tolist(), ts.tolist())))
        return torch.cat(local), torch.cat(global_)


class LTTransitionDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.index = [(e, t) for e, n in enumerate(features.lengths) for t in range(n - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, t = self.index[i]
        n = self.features.lengths[ep]
        local, global_ = self.features.history(ep, t)
        next_local, next_global = self.features.history(ep, t + 1)
        return {
            "local": local,
            "global": global_,
            "next_local": next_local,
            "next_global": next_global,
            "language": self.features.language[ep],
            "ep": ep,
            "t": t,
            "n": n,
            "terminal": float(t == n - 2),
        }
