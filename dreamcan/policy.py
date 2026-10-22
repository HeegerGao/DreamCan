import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


def rope_sections(head_dim):
    a = (head_dim // 3) // 2 * 2
    b = ((head_dim - a) // 2) // 2 * 2
    return a, b, head_dim - a - b


def rope_1d(x, pos, base=10000.0):
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
    angle = pos.float().unsqueeze(-1) * inv_freq
    emb = torch.cat([angle, angle], dim=-1)
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * emb.cos().to(x.dtype) + rotated * emb.sin().to(x.dtype)


def rope_3d(x, pos, sections):
    parts = torch.split(x, sections, dim=-1)
    return torch.cat([rope_1d(p, q) for p, q in zip(parts, pos)], dim=-1)


class TimestepMLP(nn.Module):
    def __init__(self, d_model, hidden_mult=4, max_period=10000.0):
        super().__init__()
        half = d_model // 2
        self.register_buffer("freq", torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half))
        self.net = nn.Sequential(nn.Linear(d_model, d_model * hidden_mult), nn.SiLU(), nn.Linear(d_model * hidden_mult, d_model))

    def forward(self, t):
        arg = t.float().reshape(-1, 1) * self.freq
        return self.net(torch.cat([arg.sin(), arg.cos()], dim=-1))


class FiLM(nn.Module):
    def __init__(self, cond_dim, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim * 2))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x, cond):
        scale, shift = self.net(cond).chunk(2, dim=-1)
        return x * (1.0 + scale.view(-1, x.shape[1], 1, 1)) + shift.view(-1, x.shape[1], 1, 1)


class FilmResNet34(nn.Module):
    output_dim = 512

    def __init__(self, cond_dim, pretrained=True):
        super().__init__()
        backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.stages = nn.ModuleList([backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4])
        self.avgpool = backbone.avgpool
        self.film = nn.ModuleList([FiLM(cond_dim, block.conv2.out_channels) for stage in self.stages for block in stage])

    def forward(self, x, cond):
        h = self.stem(x)
        i = 0
        for stage in self.stages:
            for block in stage:
                h = self.film[i](block(h), cond)
                i += 1
        return torch.flatten(self.avgpool(h), 1)


class PerceiverBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents, tokens):
        kv = self.norm_kv(tokens)
        out, _ = self.attn(self.norm_q(latents), kv, kv, need_weights=False)
        latents = latents + self.dropout(out)
        return latents + self.dropout(self.ffn(self.norm_ffn(latents)))


class GoalPerceiver(nn.Module):
    def __init__(self, d_model, num_latents, nhead, dim_feedforward, num_layers, dropout):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(num_latents, d_model))
        nn.init.trunc_normal_(self.latents, std=0.02)
        self.blocks = nn.ModuleList([PerceiverBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        lat = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        for blk in self.blocks:
            lat = blk(lat, tokens)
        return self.out_norm(lat)


class Attention(nn.Module):
    def __init__(self, d_model, nhead, sections, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.sections = sections
        self.dropout_p = dropout
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def heads(self, x):
        return x.view(x.shape[0], x.shape[1], self.nhead, self.head_dim).transpose(1, 2)

    def project_kv(self, x, pos):
        return rope_3d(self.heads(self.k_proj(x)), pos, self.sections), self.heads(self.v_proj(x))

    def forward(self, x, pos, k, v):
        q = rope_3d(self.heads(self.q_proj(x)), pos, self.sections)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p if self.training else 0.0)
        return self.out_proj(out.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model))


class Block(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, sections, prefix_ffn=True):
        super().__init__()
        self.attn = Attention(d_model, nhead, sections, dropout)
        self.norm_attn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.prefix_attn_affine = nn.Parameter(torch.ones(2, d_model))
        self.prefix_ffn_affine = nn.Parameter(torch.ones(2, d_model)) if prefix_ffn else None
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        with torch.no_grad():
            self.prefix_attn_affine[1].zero_()
            if self.prefix_ffn_affine is not None:
                self.prefix_ffn_affine[1].zero_()

    @staticmethod
    def affine(x, norm, a):
        return norm(x) * a[0] + a[1]

    def prefix_kv(self, prefix, pos):
        return self.attn.project_kv(self.affine(prefix, self.norm_attn, self.prefix_attn_affine), pos)

    def forward_prefix(self, prefix, pos, k, v):
        h = self.affine(prefix, self.norm_attn, self.prefix_attn_affine)
        prefix = prefix + self.dropout(self.attn(h, pos, k, v))
        return prefix + self.dropout(self.ffn(self.affine(prefix, self.norm_ffn, self.prefix_ffn_affine)))

    def forward_action(self, a, pos, k_p, v_p, t_emb):
        sa, ba, ga, sf, bf, gf = (m.unsqueeze(1) for m in self.ada(t_emb).chunk(6, dim=-1))
        h = self.norm_attn(a) * (1 + sa) + ba
        k_a, v_a = self.attn.project_kv(h, pos)
        a = a + ga * self.dropout(self.attn(h, pos, torch.cat([k_p, k_a], dim=2), torch.cat([v_p, v_a], dim=2)))
        h = self.norm_ffn(a) * (1 + sf) + bf
        return a + gf * self.dropout(self.ffn(h))


class InstructionEmbedding(nn.Module):
    def __init__(self, d_model, table, hidden_mult=2):
        super().__init__()
        self.register_buffer("table", torch.as_tensor(table, dtype=torch.float32))
        dim = self.table.shape[1]
        self.proj = nn.Sequential(nn.Linear(dim, d_model * hidden_mult), nn.GELU(), nn.Linear(d_model * hidden_mult, d_model))

    def forward(self, task):
        return self.proj(self.table.index_select(0, task).to(self.proj[0].weight.dtype)).unsqueeze(1)


class GoalPolicy(nn.Module):
    SEG_OBS_SUMMARY, SEG_OBS_PATCH, SEG_GOAL, SEG_STATE, SEG_LANG = range(5)

    def __init__(
        self,
        instruction_table,
        state_dim=15,
        action_dim=7,
        history_len=2,
        action_chunk_len=16,
        num_views=2,
        d_model=512,
        nhead=8,
        num_layers=10,
        dim_feedforward=2048,
        dropout=0.1,
        radio_local_dim=1280,
        radio_summary_dim=3840,
        radio_patch_tokens=196,
        goal_sequence_len=10,
        goal_latent_tokens=4,
        goal_perceiver_layers=2,
        goal_drop_prob=0.3,
        language_drop_prob=0.3,
        resnet_pretrained=True,
    ):
        super().__init__()
        self.d_model = d_model
        self.history_len = history_len
        self.action_chunk_len = action_chunk_len
        self.action_dim = action_dim
        self.num_views = num_views
        self.goal_sequence_len = goal_sequence_len
        self.goal_latent_tokens = goal_latent_tokens
        self.goal_drop_prob = goal_drop_prob
        self.language_drop_prob = language_drop_prob
        self.patch_grid = int(round(math.sqrt(radio_patch_tokens)))

        self.vision_encoder = FilmResNet34(d_model, pretrained=resnet_pretrained)
        self.resnet_proj = nn.Linear(FilmResNet34.output_dim, d_model)
        self.resnet_norm = nn.LayerNorm(d_model)
        self.patch_proj = nn.Sequential(nn.Linear(radio_local_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.patch_norm = nn.LayerNorm(d_model)
        self.goal_summary_proj = nn.Sequential(nn.Linear(radio_summary_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.goal_summary_norm = nn.LayerNorm(d_model)
        self.goal_perceiver = GoalPerceiver(d_model, goal_latent_tokens, nhead, dim_feedforward, goal_perceiver_layers, dropout)
        self.segment_emb = nn.Embedding(5, d_model)
        self.view_emb = nn.Parameter(torch.zeros(num_views, d_model))
        self.hist_emb = nn.Parameter(torch.zeros(history_len, d_model))
        self.goal_step_emb = nn.Parameter(torch.zeros(goal_sequence_len, d_model))
        self.null_goal_latents = nn.Parameter(torch.zeros(goal_latent_tokens, d_model))
        self.state_norm = nn.LayerNorm(d_model)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.use_language = instruction_table is not None
        if self.use_language:
            self.null_language = nn.Parameter(torch.zeros(d_model))
            self.lang_norm = nn.LayerNorm(d_model)
            self.language_encoder = InstructionEmbedding(d_model, instruction_table)
            self.task_head = nn.Linear(d_model, self.language_encoder.table.shape[0])
            nn.init.trunc_normal_(self.null_language, std=0.02)
        else:
            self.film_cond_fallback = nn.Parameter(torch.zeros(d_model))
            self.lang_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        sections = rope_sections(d_model // nhead)
        self.time_mlp = TimestepMLP(d_model)
        self.action_in = nn.Linear(action_dim, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, nhead, dim_feedforward, dropout, sections, prefix_ffn=i < num_layers - 1) for i in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        nn.init.zeros_(self.final_ada[1].weight)
        nn.init.zeros_(self.final_ada[1].bias)
        self.velocity_head = nn.Linear(d_model, action_dim)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        for p in (self.view_emb, self.hist_emb, self.goal_step_emb, self.null_goal_latents):
            nn.init.trunc_normal_(p, std=0.02)
        nn.init.trunc_normal_(self.segment_emb.weight, std=0.02)

    def seg(self, i, dtype):
        return self.segment_emb.weight[i].to(dtype)

    def drop_masks(self, b, device):
        if not self.training:
            zeros = torch.zeros(b, 1, 1, device=device, dtype=torch.bool)
            return zeros, zeros
        g_drop = (torch.rand(b, device=device) < self.goal_drop_prob).view(b, 1, 1)
        l_drop = (torch.rand(b, device=device) < self.language_drop_prob).view(b, 1, 1)
        both = g_drop & l_drop
        if bool(both.any()):
            keep_goal = (torch.rand(b, device=device) < 0.5).view(b, 1, 1)
            g_drop = torch.where(both, ~keep_goal, g_drop)
            l_drop = torch.where(both, keep_goal, l_drop)
        return g_drop, l_drop

    def build_prefix(self, images, radio_local, state, task, goal_local, goal_global):
        dtype = self.patch_proj[0].weight.dtype
        dev = radio_local.device
        b, l, v = radio_local.shape[:3]
        radio_local = radio_local.to(dtype)
        goal_local = goal_local.to(dtype)
        goal_global = goal_global.to(dtype)

        g_drop, l_drop = self.drop_masks(b, dev)
        if self.use_language:
            lang = self.lang_norm(self.language_encoder(task).to(dtype))
            lang_undropped = lang
            lang = torch.where(l_drop, self.null_language.to(dtype).view(1, 1, -1).expand_as(lang), lang)
            film = lang.squeeze(1)
        else:
            lang = self.lang_token.to(dtype).expand(b, -1, -1)
            lang_undropped = None
            film = self.film_cond_fallback.to(dtype).expand(b, -1)

        cond = film.unsqueeze(1).expand(b, l * v, self.d_model).reshape(b * l * v, -1)
        feat = self.vision_encoder(images.reshape(b * l * v, *images.shape[3:]).to(dtype), cond)
        obs_summary = self.resnet_norm(self.resnet_proj(feat)).reshape(b, l, v, -1)

        g = self.goal_sequence_len
        goal_patch = self.patch_norm(self.patch_proj(goal_local))
        goal_sum = self.goal_summary_norm(self.goal_summary_proj(goal_global))
        frames = torch.cat([goal_sum.unsqueeze(2), goal_patch], dim=2)
        goal_latents = self.goal_perceiver(frames.reshape(b * g, frames.shape[2], self.d_model))
        goal_latents = goal_latents.reshape(b, g, self.goal_latent_tokens, self.d_model)
        null = self.null_goal_latents.to(dtype).view(1, 1, self.goal_latent_tokens, self.d_model)
        goal_latents = torch.where(g_drop.unsqueeze(-1), null.expand_as(goal_latents), goal_latents)

        tokens, p_hist, p_row, p_col = [], [], [], []

        def add(tok, hist, row, col):
            tokens.append(tok)
            p_hist.append(torch.full((tok.shape[1],), hist, dtype=torch.long, device=dev))
            p_row.append(row)
            p_col.append(col)

        grid = self.patch_grid
        rows = torch.arange(grid, device=dev).repeat_interleave(grid) + 1
        cols = torch.arange(grid, device=dev).repeat(grid)
        zero = torch.zeros(1, dtype=torch.long, device=dev)
        obs_patch = self.patch_norm(self.patch_proj(radio_local))
        for hi in range(l):
            for vi in range(v):
                base = self.hist_emb[hi] + self.view_emb[vi]
                add(
                    (obs_summary[:, hi, vi] + base + self.seg(self.SEG_OBS_SUMMARY, dtype)).unsqueeze(1),
                    hi,
                    zero,
                    torch.full((1,), vi, dtype=torch.long, device=dev),
                )
                add(obs_patch[:, hi, vi] + (base + self.seg(self.SEG_OBS_PATCH, dtype)).view(1, 1, -1), hi, rows, cols)
        lat_rows = torch.zeros(self.goal_latent_tokens, dtype=torch.long, device=dev)
        lat_cols = torch.arange(self.goal_latent_tokens, dtype=torch.long, device=dev)
        for gi in range(g):
            add(
                goal_latents[:, gi] + (self.goal_step_emb[gi] + self.seg(self.SEG_GOAL, dtype)).view(1, 1, -1),
                l + gi,
                lat_rows,
                lat_cols,
            )
        st = self.state_norm(self.state_mlp(state)).unsqueeze(1).to(dtype)
        add(st + self.seg(self.SEG_STATE, dtype).view(1, 1, -1), l + g, zero, zero)
        add(lang + self.seg(self.SEG_LANG, dtype).view(1, 1, -1), l + g, zero, torch.ones(1, dtype=torch.long, device=dev))
        pos = (torch.cat(p_hist), torch.cat(p_row), torch.cat(p_col))
        return torch.cat(tokens, dim=1), pos, lang_undropped

    def action_pos(self, device):
        h = self.action_chunk_len
        return (
            torch.full((h,), self.history_len + self.goal_sequence_len + 1, dtype=torch.long, device=device),
            torch.zeros(h, dtype=torch.long, device=device),
            torch.arange(h, dtype=torch.long, device=device),
        )

    def head(self, a, t_emb):
        shift, scale = self.final_ada(t_emb).chunk(2, dim=-1)
        return self.velocity_head(self.final_norm(a) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))

    def forward(self, x_t, t, images, radio_local, state, task, goal_local, goal_global):
        prefix, pos_p, lang = self.build_prefix(images, radio_local, state, task, goal_local, goal_global)
        t_emb = self.time_mlp(t)
        pos_a = self.action_pos(x_t.device)
        a = self.action_in(x_t)
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            k_p, v_p = blk.prefix_kv(prefix, pos_p)
            a = blk.forward_action(a, pos_a, k_p, v_p, t_emb)
            if i < last:
                prefix = blk.forward_prefix(prefix, pos_p, k_p, v_p)
        logits = self.task_head(lang.squeeze(1).float()) if self.use_language else None
        return self.head(a, t_emb), logits

    @torch.no_grad()
    def encode_prefix(self, images, radio_local, state, task, goal_local, goal_global):
        prefix, pos_p, _ = self.build_prefix(images, radio_local, state, task, goal_local, goal_global)
        cache = []
        last = len(self.blocks) - 1
        for i, blk in enumerate(self.blocks):
            k_p, v_p = blk.prefix_kv(prefix, pos_p)
            cache.append((k_p, v_p))
            if i < last:
                prefix = blk.forward_prefix(prefix, pos_p, k_p, v_p)
        return cache

    @torch.no_grad()
    def denoise(self, cache, x_t, t):
        t_emb = self.time_mlp(t)
        pos_a = self.action_pos(x_t.device)
        a = self.action_in(x_t)
        for blk, (k_p, v_p) in zip(self.blocks, cache):
            a = blk.forward_action(a, pos_a, k_p, v_p, t_emb)
        return self.head(a, t_emb)

    @torch.no_grad()
    def sample(self, images, radio_local, state, task, goal_local, goal_global, steps=20):
        x = torch.randn(images.shape[0], self.action_chunk_len, self.action_dim, device=images.device)
        cache = self.encode_prefix(images, radio_local, state, task, goal_local, goal_global)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((x.shape[0],), (k + 0.5) * dt, device=x.device)
            x = x + self.denoise(cache, x, t) * dt
        return x


def flow_matching_loss(model, batch, aux_weight):
    actions = batch["actions"]
    t = torch.rand(actions.shape[0], device=actions.device, dtype=actions.dtype)
    noise = torch.randn_like(actions)
    x_t = (1.0 - t.view(-1, 1, 1)) * noise + t.view(-1, 1, 1) * actions
    velocity, logits = model(
        x_t, t, batch["images"], batch["radio_local"], batch["state"], batch["task"], batch["goal_local"], batch["goal_global"]
    )
    loss = F.mse_loss(velocity, actions - noise)
    if logits is not None and aux_weight > 0.0:
        loss = loss + aux_weight * F.cross_entropy(logits, batch["task"])
    return loss
