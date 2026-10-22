import os
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from dreamcan.common import LANG_PAD, LANG_VOCAB, episode_keys, instruction_of, language_tokens


def _value_head(in_dim, mult):
    return nn.Sequential(nn.Linear(in_dim, in_dim * mult), nn.GELU(), nn.Linear(in_dim * mult, 1))


class GoalValueNet(nn.Module):
    def __init__(
        self,
        frame_dim,
        history_len=2,
        hidden_dim=256,
        state_token_dim=512,
        temporal_nhead=8,
        temporal_layers=2,
        temporal_dropout=0.1,
        goal_bottleneck_dim=1024,
        goal_mlp_hidden_mult=2,
        lang_embed_dim=128,
        lang_gru_hidden=256,
        value_mlp_hidden_mult=2,
    ):
        super().__init__()
        self.history_len = history_len
        self.state_in_proj = nn.Linear(frame_dim, state_token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=state_token_dim,
            nhead=temporal_nhead,
            dim_feedforward=state_token_dim * 4,
            dropout=temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=temporal_layers, enable_nested_tensor=False)
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, state_token_dim))
        self.state_out_proj = nn.Linear(state_token_dim, hidden_dim)
        mid = goal_bottleneck_dim * goal_mlp_hidden_mult
        self.goal_mlp = nn.Sequential(
            nn.Linear(frame_dim, goal_bottleneck_dim),
            nn.GELU(),
            nn.Linear(goal_bottleneck_dim, mid),
            nn.GELU(),
            nn.Linear(mid, hidden_dim),
        )
        self.lang_embed = nn.Embedding(LANG_VOCAB, lang_embed_dim, padding_idx=LANG_PAD)
        self.lang_gru = nn.GRU(lang_embed_dim, lang_gru_hidden, num_layers=1, batch_first=True)
        self.lang_out = nn.Linear(lang_gru_hidden, hidden_dim)
        self.value_mlp = _value_head(5 * hidden_dim, value_mlp_hidden_mult)
        self.value_mlp_target = _value_head(5 * hidden_dim, value_mlp_hidden_mult)
        self.value_mlp_target.load_state_dict(self.value_mlp.state_dict())

    def encode_language(self, ids, lengths):
        emb = self.lang_embed(ids)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.lang_gru(packed)
        return self.lang_out(h[-1])

    def forward(self, state, goal, lang_ids, lang_lengths, target=False):
        h_l = self.encode_language(lang_ids, lang_lengths)
        tok = self.state_in_proj(state) + self.temporal_pos[:, : state.shape[1]]
        h_s = self.state_out_proj(self.temporal_encoder(tok).mean(dim=1))
        h_g = self.goal_mlp(goal)
        fused = torch.cat([h_s, h_g, h_g - h_s, h_s * h_g, h_l], dim=-1)
        head = self.value_mlp_target if target else self.value_mlp
        return head(fused).squeeze(-1)

    @torch.no_grad()
    def update_target(self, tau):
        for tp, sp in zip(self.value_mlp_target.parameters(), self.value_mlp.parameters()):
            tp.mul_(1.0 - tau).add_(sp, alpha=tau)


def expectile_loss(diff, tau):
    return (torch.where(diff < 0, 1.0 - tau, tau) * diff * diff).mean()


def sample_goals(ep, t, n, ep_lengths, gamma):
    b = ep.shape[0]
    device = ep.device
    u = torch.rand(b, device=device)
    is_future = u < 0.5
    is_random = (u >= 0.5) & (u < 0.8)
    k = torch.distributions.Geometric(probs=torch.full((b,), 1.0 - gamma, device=device)).sample().long()
    t_future = torch.minimum(t + (k - 1).clamp(min=0), n - 1)
    rand_ep = torch.randint(0, ep_lengths.shape[0], (b,), device=device)
    rand_n = ep_lengths[rand_ep].clamp(min=1)
    rand_t = torch.minimum(torch.floor(torch.rand(b, device=device) * rand_n.float()).long(), rand_n - 1)
    goal_ep = torch.where(is_random, rand_ep, ep)
    goal_t = torch.where(is_future, t_future, torch.where(is_random, rand_t, t))
    return goal_ep, goal_t


def goal_reward(ep, t, goal_ep, goal_t):
    reached = (goal_ep == ep) & ((goal_t - t).abs() <= 1)
    return torch.where(reached, 0.0, -1.0)


def split_episodes(data_path, val_fraction, seed):
    with h5py.File(data_path, "r") as f:
        keys = episode_keys(f)
        buckets = defaultdict(list)
        for k in keys:
            buckets[instruction_of(f["episodes"][k]).strip()].append(k)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for instr in sorted(buckets, key=lambda s: s[:120]):
        eps = list(buckets[instr])
        rng.shuffle(eps)
        n_val = min(max(0, int(round(len(eps) * val_fraction))), len(eps) - 1)
        val += eps[:n_val]
        train += eps[n_val:]
    return sorted(train), sorted(val)


class ValueFeatures:
    def __init__(self, data_path, feature_path, keys, history_len=2):
        self.feature_path = feature_path
        self.history_len = history_len
        self.keys = list(keys)
        with h5py.File(data_path, "r") as f:
            self.lengths = [f["episodes"][k]["actions"].shape[0] for k in self.keys]
            self.instructions = [instruction_of(f["episodes"][k]) for k in self.keys]
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
        uniq, inv = np.unique(idx, return_inverse=True)
        local = g["local"][uniq][:, 0]
        summary = g["global"][uniq][:, 0]
        flat = np.concatenate([local.reshape(len(uniq), -1), summary], axis=1)
        return torch.from_numpy(flat[inv].astype(np.float32))

    def history(self, ep, t):
        idx = np.clip(np.arange(t - self.history_len + 1, t + 1), 0, self.lengths[ep] - 1)
        return self.frames(ep, idx)

    def goals(self, eps, ts):
        return torch.stack([self.frames(int(e), [int(t)])[0] for e, t in zip(eps.tolist(), ts.tolist())])


class TransitionDataset(Dataset):
    def __init__(self, features):
        self.features = features
        self.index = [(e, t) for e, n in enumerate(features.lengths) for t in range(n - 1)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ep, t = self.index[i]
        n = self.features.lengths[ep]
        return {
            "state": self.features.history(ep, t),
            "next_state": self.features.history(ep, t + 1),
            "ep": ep,
            "t": t,
            "n": n,
            "terminal": float(t == n - 2),
            "instruction": self.features.instructions[ep],
        }


def collate_transitions(batch):
    ids, lengths = language_tokens([b["instruction"] for b in batch])
    return {
        "state": torch.stack([b["state"] for b in batch]),
        "next_state": torch.stack([b["next_state"] for b in batch]),
        "ep": torch.tensor([b["ep"] for b in batch]),
        "t": torch.tensor([b["t"] for b in batch]),
        "n": torch.tensor([b["n"] for b in batch]),
        "terminal": torch.tensor([b["terminal"] for b in batch]),
        "lang_ids": ids,
        "lang_lengths": lengths,
    }
