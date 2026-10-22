import math

import numpy as np

VARIANTS = (
    dict(wr=0.65, wp=0.35, ws=0.75, wg=1.35, sigma=0.16, gap_frac=0.055),
    dict(wr=0.90, wp=0.25, ws=0.85, wg=1.10, sigma=0.18, gap_frac=0.055),
    dict(wr=0.50, wp=0.55, ws=0.65, wg=1.80, sigma=0.14, gap_frac=0.05),
    dict(wr=0.55, wp=0.0, ws=1.00, wg=1.85, sigma=0.16, gap_frac=0.06),
    dict(wr=0.60, wp=0.80, ws=0.45, wg=0.90, sigma=0.13, gap_frac=0.05),
)
FINAL_EDGE_WEIGHT = 0.3


def moving_average(x, width):
    x = np.asarray(x, dtype=np.float64)
    if width <= 1 or x.size <= 1:
        return x.copy()
    left = width // 2
    xp = np.pad(x, (left, width - 1 - left), mode="edge")
    return np.convolve(xp, np.ones(width) / width, mode="valid")[: x.shape[0]]


def normalize_progress(raw, window=5):
    p = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(p)
    lo = float(np.nanmin(p[finite])) if finite.any() else 0.0
    hi = float(np.nanmax(p[finite])) if finite.any() else 0.0
    q = (p - lo) / (hi - lo) if hi - lo >= 1e-9 else np.linspace(0.0, 1.0, p.shape[0])
    q = np.maximum.accumulate(np.clip(moving_average(np.clip(q, 0.0, 1.0), window), 0.0, 1.0))
    span = q[-1] - q[0]
    if abs(span) >= 1e-9:
        q = (q - q[0]) / span
    else:
        q = (q - q.min()) / (q.max() - q.min() + 1e-9) if q.max() > q.min() else np.linspace(0.0, 1.0, q.size)
    return np.clip(q, 0.0, 1.0)


def reachability_cost(v_pair, gamma=0.95, eps=0.02, window=5):
    v = np.asarray(v_pair, dtype=np.float64)
    cost = -np.log(np.clip(1.0 + (1.0 - gamma) * v, eps, 1.0))
    cost[~np.isfinite(v)] = np.inf
    cost[np.tril_indices(len(cost))] = np.inf
    finite = np.isfinite(cost)
    values = np.where(finite, cost, 0.0)
    weights = finite.astype(np.float64)
    for axis in (0, 1):
        values = np.apply_along_axis(moving_average, axis, values, window)
        weights = np.apply_along_axis(moving_average, axis, weights, window)
    out = np.full_like(values, np.inf)
    ok = weights > 1e-9
    out[ok] = values[ok] / weights[ok]
    out[np.tril_indices(len(out))] = np.inf
    return out


def label_dp(cost, progress, k, min_gap, v):
    n = progress.shape[0]
    if n <= (k + 1) * min_gap:
        return [], math.inf
    dp = np.full((n, k + 1), np.inf)
    nxt = np.full((n, k + 1), -1, dtype=np.int64)
    targets = np.linspace(0.10, 0.92, k)
    last = n - 1
    for i in range(n):
        if last - i >= min_gap and np.isfinite(cost[i, last]):
            delta = max(progress[last] - progress[i], 0.0)
            dp[i, 0] = FINAL_EDGE_WEIGHT * v["wr"] * cost[i, last] - FINAL_EDGE_WEIGHT * v["wg"] * delta
    dp[last, 0] = 0.0
    for m in range(1, k + 1):
        q = targets[k - m]
        for i in range(n - 1, -1, -1):
            lo, hi = i + min_gap, n - 1 - m * min_gap
            if lo > hi:
                continue
            js = np.arange(lo, hi + 1)
            mu = (last - i) / (m + 1)
            edge = (
                v["wr"] * cost[i, js]
                + v["wp"] * ((progress[js] - q) / v["sigma"]) ** 2
                + v["ws"] * (((js - i) - mu) / (mu + 1e-6)) ** 2
                - v["wg"] * np.maximum(progress[js] - progress[i], 0.0)
            )
            total = edge + dp[js, m - 1]
            best = int(np.argmin(total))
            if np.isfinite(total[best]):
                dp[i, m] = total[best]
                nxt[i, m] = js[best]
    out, cur = [], 0
    for m in range(k, 0, -1):
        cur = int(nxt[cur, m])
        if cur < 0:
            return [], math.inf
        out.append(cur)
    return out, float(dp[0, k])


def quality(timesteps, progress, cost, k):
    ts = np.asarray(timesteps)
    starts = np.concatenate([[0], ts[:-1]])
    c = np.asarray([cost[i, j] for i, j in zip(starts, ts)])
    span = max(1, progress.shape[0] - 1)
    score = 0.80 * np.nanmean(c) + 0.18 * np.nanmax(c)
    score -= 0.85 * np.sum(np.maximum(progress[ts] - progress[starts], 0.0))
    score += 1.20 * max(0.0, 0.78 - ts[-1] / span)
    score += 0.80 * max(0.0, ts[0] / span - 0.32)
    score += 0.80 * max(0.0, 0.045 - np.min(np.diff(np.r_[0, ts])) / span)
    score += 0.30 * abs(progress[ts[-1]] - min(0.96, progress[-1]))
    return float(score + 0.02 * abs(k - 6))


def fallback(n, k):
    if n <= 1:
        return [0]
    if k >= n - 1:
        return list(range(1, n))
    out = []
    for t in np.rint(np.linspace(1, n - 1, k)).astype(np.int64).tolist():
        if not out or t > out[-1]:
            out.append(int(t))
    t = 1
    while len(out) < k and t < n:
        if t not in out:
            out = sorted(out + [t])
        t += 1
    out[-1] = n - 1
    return out


def annotate(v_pair, raw_progress, num_subgoals=16, gamma=0.95, eps=0.02, window=5, min_gap=4):
    n = len(raw_progress)
    progress = normalize_progress(raw_progress, window)
    cost = reachability_cost(v_pair, gamma, eps, window)
    k_total = min(num_subgoals, max(1, n - 1))
    k = k_total - 1
    best, best_q = None, math.inf
    for v in VARIANTS:
        if k > 0:
            gap = min(max(min_gap, int(round(v["gap_frac"] * n))), max(1, (n - 1) // max(k + 1, 2)))
            pred, _ = label_dp(cost, progress, k, gap, v)
            if not pred:
                continue
        else:
            pred = []
        ts = pred if pred and pred[-1] == n - 1 else pred + [n - 1]
        q = quality(ts, progress, cost, k_total)
        if q < best_q:
            best, best_q = ts, q
    return best if best is not None else fallback(n, k_total)
