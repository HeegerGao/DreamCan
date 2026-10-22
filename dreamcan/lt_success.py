import numpy as np

from dreamcan.lt_progress import frame_progress

GATES = {
    "line": {
        "goal": ("end_soft_inliers", "end_separation"),
        "integrity": ("collapse_frac", "max_track_gap"),
        "transition": ("max_progress_drop", "p90_track_disp"),
    },
    "default": {
        "goal": ("end_progress",),
        "integrity": ("collapse_frac", "max_track_gap"),
        "transition": ("max_progress_drop", "p90_track_disp"),
    },
}
HIGHER_IS_BETTER = {"end_soft_inliers", "end_separation", "end_progress"}
REF_FRAMES = 5


def gates_for(task):
    return GATES.get(task, GATES["default"])


def plan_metrics(task, detections, instruction):
    frames = detections["frames"]
    roi = detections["table_roi"]
    comps = [
        frame_progress(task, np.asarray([d["centroid"] for d in f], dtype=np.float32).reshape(-1, 2), instruction, roi)
        for f in frames
    ]
    progress = np.asarray([c["progress"] for c in comps])
    n_total = len(frames)
    n_sub = n_total - 1
    tracks, real_counts = {}, []
    for fi, f in enumerate(frames):
        for d in f:
            tracks.setdefault(d["track_id"], {})[fi] = d
        real_counts.append(sum(not d["interpolated"] for d in f))

    gaps, disps = [], []
    for per_frame in tracks.values():
        real = [fi in per_frame and not per_frame[fi]["interpolated"] for fi in range(n_total)]
        run = best = 0
        for fi in range(1, n_total):
            run = 0 if real[fi] else run + 1
            best = max(best, run)
        gaps.append(best)
        for fi in range(1, n_total):
            if real[fi] and real[fi - 1]:
                a = np.asarray(per_frame[fi]["centroid"], dtype=np.float64)
                b = np.asarray(per_frame[fi - 1]["centroid"], dtype=np.float64)
                disps.append(float(np.linalg.norm(a - b)))

    tail = [-2, -1] if n_total >= 3 else [-1]
    key = "soft_inliers" if task == "line" else "progress"
    end = max(tail, key=lambda i: comps[i][key])
    run_max, max_drop = progress[0], 0.0
    for p in progress[1:]:
        max_drop = max(max_drop, run_max - p)
        run_max = max(run_max, p)
    out = {
        "end_progress": float(progress[end]),
        "max_track_gap": int(max(gaps)) if gaps else n_sub,
        "collapse_frac": float(np.mean([c < 6 for c in real_counts[1:]])) if n_sub else 1.0,
        "max_progress_drop": float(max_drop),
        "p90_track_disp": float(np.nanquantile(np.asarray(disps), 0.90)) if disps else float("nan"),
    }
    if task == "line":
        out["end_soft_inliers"] = float(comps[end]["soft_inliers"])
        out["end_separation"] = float(comps[end]["separation"])
    return out


def passes(metrics, thresholds, task):
    checks = {}
    for name, thr in thresholds.items():
        value = metrics.get(name)
        if value is None or np.isnan(value):
            checks[name] = False
        else:
            checks[name] = value >= thr if name in HIGHER_IS_BETTER else value <= thr
    gates = {g: all(checks[m] for m in ms) for g, ms in gates_for(task).items()}
    return all(gates.values()), gates


def calibrate(rows, task, q=0.95):
    thresholds = {}
    for name in [m for ms in gates_for(task).values() for m in ms]:
        values = np.asarray([r[name] for r in rows if not np.isnan(r[name])], dtype=np.float64)
        thresholds[name] = float(np.quantile(values, 1.0 - q if name in HIGHER_IS_BETTER else q))
    return thresholds
