import json
import math
import re

import numpy as np

TASKS = ("line", "all_to_region", "canonical_shape")
COLORS = ("red", "green", "blue", "yellow")
WIDTH, HEIGHT = 320, 180
NUM_BLOCKS = 8
BOARD_POLY = np.asarray([(55, 22), (260, 22), (312, 168), (14, 168)], dtype=np.float32)

REGION_ALIASES = {
    "top of the board": "top center",
    "bottom of the board": "bottom center",
    "left of the board": "center left",
    "right of the board": "center right",
    "top": "top center",
    "bottom": "bottom center",
    "left": "center left",
    "right": "center right",
}
LINE_REGIONS = {
    "top left": (0.30, 0.28),
    "top center": (0.50, 0.27),
    "top right": (0.70, 0.28),
    "center left": (0.34, 0.50),
    "center": (0.50, 0.50),
    "center right": (0.66, 0.50),
    "bottom left": (0.25, 0.73),
    "bottom center": (0.50, 0.74),
    "bottom right": (0.75, 0.73),
}
PILE_REGIONS = {
    "top left": (0.17, 0.06),
    "top center": (0.50, 0.06),
    "top right": (0.84, 0.06),
    "center left": (0.17, 0.32),
    "center": (0.50, 0.32),
    "center right": (0.84, 0.32),
    "bottom left": (0.17, 0.58),
    "bottom center": (0.50, 0.58),
    "bottom right": (0.84, 0.58),
}

BLOCK_DIAMETER = 0.12 * min(WIDTH, HEIGHT)
LINE_TAU = 1.00 * BLOCK_DIAMETER
LINE_BETA = 0.15 * BLOCK_DIAMETER
SEPARATION_BETA = 0.25 * BLOCK_DIAMETER
REGION_SEARCH = 0.30
DIAGONAL_BAND = 0.15
ANCHOR_THRESHOLD = 0.85
THETAS = {
    "horizontal": 0.0,
    "vertical": math.pi / 2.0,
    "diagonal_pos": math.pi / 6.0,
    "diagonal_neg": -math.pi / 6.0,
}

PILE_TAU = 0.34 * min(WIDTH, HEIGHT)
PILE_BETA = 0.08 * min(WIDTH, HEIGHT)
COMPACT_TAU = 0.19 * min(WIDTH, HEIGHT)
COMPACT_BETA = 0.07 * min(WIDTH, HEIGHT)
COMPACT_WEIGHT = 0.4

BOARD_W, BOARD_H = 300.0, 170.0
SHAPE_TAU = 0.088 * min(BOARD_W, BOARD_H)
SHAPE_BETA = 0.04 * min(BOARD_W, BOARD_H)
COMPLETE_SAMPLES = 12
COMPLETE_SIGMA_MULT = 0.6
SIGMA_C_MIN, SIGMA_C_MAX = 7.0, 14.0
COMPLETE_WEIGHT = 0.5
SCALE_MIN, SCALE_MAX = 20.0, 150.0
PERIMETER_MIN = 120.0
Y_SQUASH = (0.35, 0.45, 0.55, 0.65, 0.75, 0.9)
ROTATIONS = tuple(float(np.deg2rad(d)) for d in (-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0))
ICP_ITERS = 5
TEMPLATE_SAMPLES = 64


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_detections(path):
    with open(path) as f:
        return json.load(f)


def frame_blocks(frame):
    return [d for d in frame if d.get("color") in COLORS]


def frame_points(frame):
    pts = [d["centroid"] for d in frame_blocks(frame)]
    return np.asarray(pts, dtype=np.float32).reshape(-1, 2)


def table_roi_of(detections):
    raw = detections.get("table_roi")
    full = (0.0, 0.0, float(WIDTH - 1), float(HEIGHT - 1))
    if raw is None:
        return full
    x1, y1, x2, y2 = [float(v) for v in raw]
    x1, x2 = float(np.clip(x1, 0.0, WIDTH - 1)), float(np.clip(x2, 0.0, WIDTH - 1))
    y1, y2 = float(np.clip(y1, 0.0, HEIGHT - 1)), float(np.clip(y2, 0.0, HEIGHT - 1))
    return full if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)


def board_point(anchors, region):
    p0, p1, p2, p3 = BOARD_POLY
    x, y = anchors[REGION_ALIASES.get(region, region)]
    top = p0 * (1.0 - x) + p1 * x
    bottom = p3 * (1.0 - x) + p2 * x
    return top * (1.0 - y) + bottom * y


def line_region(text):
    for name in LINE_REGIONS:
        if name in text:
            return name
    for alias, region in REGION_ALIASES.items():
        if alias in text:
            return region
    return None


def pile_region(text):
    for name in sorted(PILE_REGIONS, key=len, reverse=True):
        if name in text:
            return name
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if alias in text:
            return REGION_ALIASES[alias]
    return None


def parse_line_goal(instruction):
    text = instruction.lower().strip()
    if "horizontal line" in text:
        return "horizontal", line_region(text), None
    if "vertical line" in text:
        return "vertical", line_region(text), None
    if "diagonal line" in text:
        direction = "neg" if "top right to bottom left" in text else "pos"
        return "diagonal", None if " from " in text else line_region(text), direction
    if "blocks in a line" in text or re.search(r"\bin a line\b", text):
        return "any", line_region(text), "neg" if "top right to bottom left" in text else "pos"
    return "any", line_region(text), None


def line_thetas(goal):
    orientation, _, direction = goal
    if orientation in ("horizontal", "vertical"):
        return [orientation]
    if orientation == "diagonal":
        return ["diagonal_neg" if direction == "neg" else "diagonal_pos"]
    return list(THETAS)


def roi_corners(roi):
    x1, y1, x2, y2 = roi
    return np.asarray([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], dtype=np.float32)


def region_subroi(region, roi):
    region = REGION_ALIASES.get(region, region)
    x1, y1, x2, y2 = roi
    w, h = x2 - x1, y2 - y1
    if "left" in region:
        rx1, rx2 = x1, x1 + REGION_SEARCH * w
    elif "right" in region:
        rx1, rx2 = x2 - REGION_SEARCH * w, x2
    elif "center" in region:
        cx = 0.5 * (x1 + x2)
        rx1, rx2 = cx - 0.5 * REGION_SEARCH * w, cx + 0.5 * REGION_SEARCH * w
    else:
        rx1, rx2 = x1, x2
    if "top" in region:
        ry1, ry2 = y1, y1 + REGION_SEARCH * h
    elif "bottom" in region:
        ry1, ry2 = y2 - REGION_SEARCH * h, y2
    elif "center" in region:
        cy = 0.5 * (y1 + y2)
        ry1, ry2 = cy - 0.5 * REGION_SEARCH * h, cy + 0.5 * REGION_SEARCH * h
    else:
        ry1, ry2 = y1, y2
    return (float(rx1), float(ry1), float(rx2), float(ry2))


def offset_range(goal, theta_name, roi):
    theta = THETAS[theta_name]
    n = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    if goal[1]:
        q = roi_corners(region_subroi(goal[1], roi)) @ n
        return float(np.min(q)), float(np.max(q))
    if goal[0] == "diagonal" and theta_name in ("diagonal_pos", "diagonal_neg"):
        q = roi_corners(roi) @ n
        center = np.asarray([0.5 * (roi[0] + roi[2]), 0.5 * (roi[1] + roi[3])], dtype=np.float32)
        half = DIAGONAL_BAND * float(np.max(q) - np.min(q))
        return float(center @ n) - half, float(center @ n) + half
    return None


def best_offset(q, span):
    lo, hi = (float(np.min(q)), float(np.max(q))) if span is None else sorted((float(span[0]), float(span[1])))
    if abs(hi - lo) < 1e-6:
        return float(0.5 * (lo + hi))
    grid = np.linspace(lo, hi, num=257, dtype=np.float32)
    cands = np.unique(np.concatenate([q[(q >= lo) & (q <= hi)].astype(np.float32), grid]))
    scores = sigmoid((LINE_TAU - np.abs(q[None, :] - cands[:, None])) / LINE_BETA).sum(axis=1)
    return float(cands[int(np.argmax(scores))])


def line_fit(points, goal, theta_name, roi, fixed_point=None):
    theta = THETAS[theta_name]
    u = np.asarray([math.cos(theta), math.sin(theta)], dtype=np.float32)
    n = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    proj = points @ n
    b = best_offset(proj, offset_range(goal, theta_name, roi)) if fixed_point is None else float(np.asarray(fixed_point, dtype=np.float32) @ n)
    inliers = sigmoid((LINE_TAU - np.abs(proj - b)) / LINE_BETA)
    soft = float(np.sum(inliers))
    count = float(np.clip((soft / float(NUM_BLOCKS)) ** 0.5, 0.0, 1.0))
    deltas = np.diff(np.sort(points @ u))
    separation = float(np.mean(sigmoid((deltas - BLOCK_DIAMETER) / SEPARATION_BETA))) if deltas.size else 0.0
    return {
        "progress": float(np.clip(count * (0.8 + 0.2 * separation), 0.0, 1.0)),
        "soft_inliers": soft,
        "separation": separation,
        "count": count,
        "offset": b,
        "theta": theta_name,
    }


def line_frame(points, goal, roi, thetas=None, fixed_point=None):
    if points.shape[0] < 1:
        return {"progress": 0.0, "soft_inliers": 0.0, "separation": 0.0}
    fits = [line_fit(points, goal, t, roi, fixed_point) for t in (thetas or line_thetas(goal))]
    return max(fits, key=lambda r: r["progress"])


def line_anchor(frames, goal, roi):
    best = None
    for frame in frames[max(0, int(math.floor(0.75 * len(frames))) - 1):]:
        points = frame_points(frame)
        if points.shape[0] < 1:
            continue
        for t in line_thetas(goal):
            row = line_fit(points, goal, t, roi)
            if best is None or row["progress"] > best["progress"]:
                best = row
    theta_name = best["theta"] if best is not None else line_thetas(goal)[0]
    b = best["offset"] if best is not None else 0.0
    theta = THETAS[theta_name]
    n = np.asarray([-math.sin(theta), math.cos(theta)], dtype=np.float32)
    by_track, best_seen = {}, None
    for idx, frame in enumerate(frames):
        for block in frame_blocks(frame):
            p = np.asarray(block["centroid"], dtype=np.float32)
            score = float(sigmoid(np.asarray([(LINE_TAU - abs(float(p @ n - b))) / LINE_BETA], dtype=np.float32))[0])
            by_track.setdefault(block.get("track_id", ""), []).append((idx, score))
            if best_seen is None or score > best_seen[0]:
                best_seen = (score, idx, block)
    for idx, frame in enumerate(frames):
        for block in frame_blocks(frame):
            scores = [s for j, s in by_track[block.get("track_id", "")] if idx <= j < idx + 3]
            if len(scores) == 3 and min(scores) >= ANCHOR_THRESHOLD:
                return theta_name, block.get("track_id", ""), idx, block["centroid"]
    if best_seen is None:
        return theta_name, None, 0, [0.0, 0.0]
    return theta_name, best_seen[2].get("track_id", ""), best_seen[1], best_seen[2]["centroid"]


def line_episode(detections, instruction):
    frames = detections["frames"]
    goal = parse_line_goal(instruction)
    roi = table_roi_of(detections)
    theta_name, track, start, xy = line_anchor(frames, goal, roi)
    out = []
    for idx, frame in enumerate(frames):
        point = np.asarray(xy, dtype=np.float32)
        if idx >= start and track is not None:
            for block in frame_blocks(frame):
                if block.get("track_id", "") == track:
                    point = np.asarray(block["centroid"], dtype=np.float32)
                    break
        out.append(line_frame(frame_points(frame), goal, roi, [theta_name], point)["progress"])
    return np.asarray(out, dtype=np.float32)


def pile_frame(points, instruction):
    region = pile_region(instruction.lower())
    if region is None:
        raise ValueError(f"no region in instruction: {instruction!r}")
    if points.shape[0] < 1:
        return {"progress": 0.0, "soft_inliers": 0.0}
    g = board_point(PILE_REGIONS, region)
    membership = sigmoid((PILE_TAU - np.linalg.norm(points - g[None, :], axis=1).astype(np.float32)) / PILE_BETA)
    spread = float(np.linalg.norm(points - points.mean(axis=0)[None, :], axis=1).mean())
    compact = float(sigmoid(np.asarray([(COMPACT_TAU - spread) / COMPACT_BETA], dtype=np.float32))[0])
    coverage = float(np.mean(membership))
    return {
        "progress": float(np.clip(coverage * (1.0 - COMPACT_WEIGHT + COMPACT_WEIGHT * compact), 0.0, 1.0)),
        "soft_inliers": float(np.sum(membership)),
    }


def arc(cx, cy, r, a0, a1, n=48):
    ang = np.deg2rad(np.linspace(a0, a1, n))
    return np.stack([cx + r * np.cos(ang), cy + r * np.sin(ang)], axis=1)


def shape_polylines():
    def P(*pts):
        return np.asarray(pts, dtype=np.float32)

    return {
        "a": [P((-0.8, -1), (0, 1), (0.8, -1)), P((-0.45, -0.15), (0.45, -0.15))],
        "c": [arc(0, 0, 0.9, 45, 315)],
        "e": [P((0.6, 1), (-0.6, 1), (-0.6, -1), (0.6, -1)), P((-0.6, 0), (0.45, 0))],
        "g": [arc(0, 0, 0.9, 30, 330), P((0.78, -0.45), (0.1, -0.45))],
        "j": [P((-0.5, 1), (0.7, 1)), P((0.35, 1), (0.35, -0.4)), arc(-0.05, -0.4, 0.4, 0, -180, 24)],
        "l": [P((-0.5, 1), (-0.5, -1), (0.7, -1))],
        "n": [P((-0.7, -1), (-0.7, 1), (0.7, -1), (0.7, 1))],
        "o": [arc(0, 0, 0.9, 0, 360)],
        "s": [
            P(
                (0.6, 0.8), (0.2, 1.0), (-0.35, 0.9), (-0.55, 0.55), (-0.3, 0.15),
                (0.3, -0.15), (0.55, -0.55), (0.35, -0.9), (-0.2, -1.0), (-0.6, -0.8),
            )
        ],
        "t": [P((-0.8, 1), (0.8, 1)), P((0, 1), (0, -1))],
        "u": [P((-0.7, 1), (-0.7, -0.35)), arc(0, -0.35, 0.7, 180, 360, 24), P((0.7, -0.35), (0.7, 1))],
        "v": [P((-0.75, 1), (0, -1), (0.75, 1))],
        "x": [P((-0.75, 1), (0.75, -1)), P((0.75, 1), (-0.75, -1))],
        "y": [P((-0.7, 1), (0, 0.05)), P((0.7, 1), (0, 0.05)), P((0, 0.05), (0, -1))],
        "z": [P((-0.7, 1), (0.7, 1), (-0.7, -1), (0.7, -1))],
        "square": [P((-0.85, -0.85), (0.85, -0.85), (0.85, 0.85), (-0.85, 0.85), (-0.85, -0.85))],
        "circle": [arc(0, 0, 0.9, 0, 360)],
        "triangle": [P((-0.9, -0.75), (0.9, -0.75), (0, 0.85), (-0.9, -0.75))],
        "diamond": [P((0, -1), (0.85, 0), (0, 1), (-0.85, 0), (0, -1))],
        "parallelogram": [P((-1, -0.55), (0.45, -0.55), (1, 0.55), (-0.45, 0.55), (-1, -0.55))],
        "smiley": [arc(0, 0.05, 0.85, 195, 345, 32), P((-0.4, 0.45), (-0.4, 0.5)), P((0.4, 0.45), (0.4, 0.5))],
        "rainbow": [arc(0, -0.25, 0.95, 0, 180, 32)],
    }


def sample_template(polylines, n_dense, n_complete, y_squash):
    segs, total = [], 0.0
    for poly in polylines:
        poly = np.asarray(poly, dtype=np.float64) * np.asarray([1.0, -float(y_squash)])
        for a, b in zip(poly[:-1], poly[1:]):
            length = float(np.linalg.norm(b - a))
            if length > 1e-9:
                segs.append((a, b, length))
                total += length

    def sample(n):
        pts = np.empty((n, 2), dtype=np.float64)
        acc, si = 0.0, 0
        for i, s in enumerate((np.arange(n) + 0.5) / n * total):
            while si < len(segs) - 1 and acc + segs[si][2] < s:
                acc += segs[si][2]
                si += 1
            a, b, length = segs[si]
            pts[i] = a + (b - a) * min(max((s - acc) / length, 0.0), 1.0)
        return pts

    return sample(n_dense).astype(np.float32), sample(n_complete).astype(np.float32), total


TEMPLATES = {
    name: {sq: sample_template(polys, TEMPLATE_SAMPLES, COMPLETE_SAMPLES, sq) for sq in Y_SQUASH}
    for name, polys in shape_polylines().items()
}


def parse_shape(instruction):
    text = instruction.lower()
    if "rainbow" in text:
        return "rainbow"
    match = re.search(r"make\s+an?\s+[\"']*([a-z]+)[\"']*\s+shape", text)
    if match and match.group(1) in TEMPLATES:
        return match.group(1)
    for word in sorted(TEMPLATES, key=len, reverse=True):
        if len(word) > 1 and word in text:
            return word
    match = re.search(r"[\"']([a-z])[\"']", text)
    if match and match.group(1) in TEMPLATES:
        return match.group(1)
    raise ValueError(f"no shape in instruction: {instruction!r}")


def rectify(points):
    if points.size == 0:
        return points.reshape(0, 2)
    p0, p1, p2, p3 = BOARD_POLY.astype(np.float64)
    v = (points[:, 1] - p0[1]) / max(p3[1] - p0[1], 1e-6)
    x_left = p0[0] + (p3[0] - p0[0]) * v
    x_right = p1[0] + (p2[0] - p1[0]) * v
    u = (points[:, 0] - x_left) / np.maximum(x_right - x_left, 1e-6)
    return np.stack([u * BOARD_W, v * BOARD_H], axis=1).astype(np.float32)


def fit_similarity(blocks, dense, rot, scale_min, init_scale, init_center):
    c, s = math.cos(rot), math.sin(rot)
    U = dense @ np.asarray([[c, -s], [s, c]], dtype=np.float32).T
    u_mean = U.mean(axis=0)
    scale = float(np.clip(init_scale, scale_min, SCALE_MAX))
    trans = init_center - scale * u_mean
    for _ in range(ICP_ITERS):
        world = U * scale + trans
        d2 = np.sum((blocks[:, None, :] - world[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(blocks.shape[0]), nearest])
        w = sigmoid((SHAPE_TAU - d) / SHAPE_BETA) + 0.05
        w = w / float(np.sum(w))
        matched = U[nearest]
        m_mean = np.sum(matched * w[:, None], axis=0)
        b_mean = np.sum(blocks * w[:, None], axis=0)
        var = float(np.sum(w * np.sum((matched - m_mean) ** 2, axis=1)))
        if var < 1e-9:
            break
        cov = float(np.sum(w * np.sum((matched - m_mean) * (blocks - b_mean), axis=1)))
        scale = float(np.clip(cov / var, scale_min, SCALE_MAX))
        trans = b_mean - scale * m_mean
    return scale, U * scale + trans


def shape_frame(points, instruction):
    blocks = rectify(points)
    if blocks.shape[0] < 3:
        return {"progress": 0.0}
    b_mean = blocks.mean(axis=0)
    b_rms = float(np.sqrt(np.mean(np.sum((blocks - b_mean) ** 2, axis=1))))
    b_med = np.median(blocks, axis=0).astype(np.float32)
    mad_rms = float(np.median(np.linalg.norm(blocks - b_med[None, :], axis=1))) * 1.4826
    best = 0.0
    for dense, complete_pts, arc_len in TEMPLATES[parse_shape(instruction)].values():
        n_complete = complete_pts.shape[0]
        complete_idx = np.clip((((np.arange(n_complete) + 0.5) / n_complete) * dense.shape[0]).astype(int), 0, dense.shape[0] - 1)
        scale_min = max(SCALE_MIN, PERIMETER_MIN / max(arc_len, 1e-6))
        u_rms = float(np.sqrt(np.mean(np.sum((dense - dense.mean(axis=0)) ** 2, axis=1))) + 1e-9)
        inits = ((b_rms / u_rms, b_mean), (mad_rms / u_rms, b_med), (0.5 * (scale_min + SCALE_MAX), b_med))
        for rot in ROTATIONS:
            for init_scale, init_center in inits:
                scale, world = fit_similarity(blocks, dense, rot, scale_min, float(init_scale), init_center)
                d = np.sqrt(np.sum((blocks[:, None, :] - world[None, :, :]) ** 2, axis=2)).min(axis=1)
                m = sigmoid((SHAPE_TAU - d) / SHAPE_BETA)
                on_shape = float(np.sum(m) / float(NUM_BLOCKS))
                sigma_c = float(np.clip(COMPLETE_SIGMA_MULT * arc_len * scale / float(NUM_BLOCKS), SIGMA_C_MIN, SIGMA_C_MAX))
                d_sb = np.sqrt(np.sum((world[complete_idx][:, None, :] - blocks[None, :, :]) ** 2, axis=2))
                cover = np.max(m[None, :] * np.exp(-0.5 * (d_sb / sigma_c) ** 2), axis=1)
                complete = float(np.mean(np.clip(cover, 0.0, 1.0)))
                best = max(best, float(np.clip(on_shape * (1.0 - COMPLETE_WEIGHT + COMPLETE_WEIGHT * complete), 0.0, 1.0)))
    return {"progress": best}


def frame_progress(task, points, instruction, table_roi=None):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if task == "line":
        roi = table_roi_of({"table_roi": table_roi})
        return line_frame(points, parse_line_goal(instruction), roi)
    if task == "all_to_region":
        return pile_frame(points, instruction)
    if task == "canonical_shape":
        return shape_frame(points, instruction)
    raise ValueError(task)


def episode_progress(task, detections, instruction):
    if task == "line":
        return line_episode(detections, instruction)
    return np.asarray(
        [frame_progress(task, frame_points(f), instruction)["progress"] for f in detections["frames"]], dtype=np.float32
    )
