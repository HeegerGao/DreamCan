import numpy as np

NUM_SUBGOALS = {
    "put both the alphabet soup and the tomato sauce in the basket": 8,
    "put both the cream cheese box and the butter in the basket": 8,
    "turn on the stove and put the moka pot on it": 7,
    "put the black bowl in the bottom drawer of the cabinet and close it": 6,
    "put the white mug on the left plate and put the yellow and white mug on the right plate": 8,
    "pick up the book and place it in the back compartment of the caddy": 5,
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": 8,
    "put both the alphabet soup and the cream cheese box in the basket": 8,
    "put both moka pots on the stove": 8,
    "put the yellow and white mug in the microwave and close it": 6,
}


def monotone_progress(progress, width=7):
    p = np.clip(np.asarray(progress, dtype=np.float64), 0.0, 1.0)
    pad = width // 2
    p = np.convolve(np.pad(p, (pad, pad), mode="edge"), np.ones(width) / width, mode="valid")[: len(p)]
    span = p[-1] - p[0]
    if abs(span) < 0.02:
        q = np.clip(p - p.min(), 0.0, None)
        q = q / (q.max() + 1e-6) if q.max() > 1e-6 else np.linspace(0.0, 1.0, len(p))
    else:
        q = np.clip((p - p[0]) / (span + 1e-6), 0.0, 1.0)
    q = np.maximum.accumulate(q)
    if q[-1] > 1e-6:
        q = q / q[-1]
    return np.clip(q, 0.0, 1.0)


def reachability_cost(v_pair, gamma, eps):
    c = -np.log(np.clip(1.0 + (1.0 - gamma) * v_pair.astype(np.float64), eps, 1.0))
    c[~np.isfinite(v_pair)] = np.inf
    c[np.tril_indices(len(c))] = np.inf
    return c


def select_subgoals(cost, progress, k, min_gap, w_reach=0.5, w_spacing=1.0, w_progress=2.0):
    n = len(progress)
    if n <= k * min_gap:
        return []
    dp = np.full((n, k + 1), np.inf)
    nxt = np.full((n, k + 1), -1, dtype=np.int64)
    dp[:, 0] = 0.0
    for m in range(1, k + 1):
        for i in range(n - 1, -1, -1):
            lo, hi = i + min_gap, n - 1 - (m - 1) * min_gap
            if lo > hi:
                continue
            js = np.arange(lo, hi + 1)
            mu = (n - 1 - i) / (m + 1)
            edge = (
                w_reach * cost[i, js]
                + w_spacing * (((js - i) - mu) / (mu + 1e-6)) ** 2
                - w_progress * np.maximum(progress[js] - progress[i], 0.0)
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
            return []
        out.append(cur)
    return out


def annotate(v_pair, progress, k, gamma=0.95, eps=0.02, gap_frac=0.06, min_gap=8):
    n = len(progress)
    gap = max(min_gap, int(round(gap_frac * n)))
    subgoals = select_subgoals(reachability_cost(v_pair, gamma, eps), monotone_progress(progress), k, gap)
    if not subgoals:
        return []
    if subgoals[-1] != n - 1:
        subgoals.append(n - 1)
    return subgoals
