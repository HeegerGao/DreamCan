import copy
import math
from dataclasses import dataclass

import cv2
import imageio.v3 as iio
import numpy as np
import torch

COLORS = ("red", "green", "blue", "yellow")
COLOR_ORDER = {color: i for i, color in enumerate(COLORS)}
HSV_RANGES = {
    "red": [((0, 70, 40), (12, 255, 255)), ((168, 70, 40), (180, 255, 255))],
    "green": [((35, 50, 40), (90, 255, 255))],
    "blue": [((90, 50, 40), (135, 255, 255))],
    "yellow": [((18, 50, 60), (38, 255, 255))],
    "orange": [((8, 70, 60), (25, 255, 255))],
}
MIN_AREA = 18.0
MAX_AREA = 2600.0
MASK_MIN_AREA = 28.0
MASK_MAX_AREA = 1900.0
PROMPT_PAD = 1.5
MAX_PROMPTS_PER_OBJECT = 3
MAX_CENTER_JUMP = 42.0
SNAP_DIST = 30.0
RESCUE_DIST = 62.0
MAX_ONE_SIDED_FILL = 10
MAX_INTERP_GAP = 14
MAX_INTERP_DIST = 34.0
TABLE_TOP_EXTENSION = 12


@dataclass(frozen=True)
class Prompt:
    obj_id: int
    frame_index: int
    bbox: list
    point: list
    score: float


@dataclass(frozen=True)
class TrackPlan:
    obj_id: int
    track_id: str
    color: str
    prompts: list


def _clip_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, max(width - 1, 0)))
    y1 = float(np.clip(y1, 0, max(height - 1, 0)))
    x2 = float(np.clip(x2, x1 + 1, width))
    y2 = float(np.clip(y2, y1 + 1, height))
    return [x1, y1, x2, y2]


def _expand_bbox(bbox, pad, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return _clip_bbox([x1 - pad, y1 - pad, x2 + pad, y2 + pad], width, height)


def _bbox_wh(bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return np.asarray([max(3.0, x2 - x1), max(3.0, y2 - y1)], dtype=np.float32)


def _center(det):
    return np.asarray(det["centroid"], dtype=np.float32)


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    return 0.0 if union <= 0.0 else float(inter / union)


def _roi_mask(shape, roi):
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if roi is None:
        mask[:, :] = 255
        return mask
    x1, y1, x2, y2 = [int(v) for v in roi]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _sanitize_roi(roi, width, height):
    x1, y1, x2, y2 = [int(round(v)) for v in roi]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return [int(width * 0.12), int(height * 0.14), int(width * 0.90), int(height * 0.90)]
    return [x1, y1, x2, y2]


def table_mask(image, roi=None):
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    not_blue = ~((hue >= 88) & (hue <= 140) & (sat > 45))
    wood = (sat < 125) & (val > 70) & (lab[..., 0] > 75) & not_blue
    wood[: int(h * 0.08), :] = False
    wood[int(h * 0.97) :, :] = False
    if roi is not None:
        wood &= _roi_mask((h, w), roi) > 0
    mask = wood.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scale = np.asarray([w / 320.0, h / 180.0], dtype=np.float32)
    poly = np.round(np.asarray([(65, 20), (258, 20), (309, 166), (10, 166)], dtype=np.float32) * scale).astype(np.int32)
    fixed = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(fixed, poly, 255)
    if roi is not None:
        fixed = cv2.bitwise_and(fixed, _roi_mask((h, w), roi))
    if contours:
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) > 0.10 * w * h:
            filled = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(filled, [cv2.convexHull(contour)], -1, 255, thickness=cv2.FILLED)
            inter = cv2.bitwise_and(fixed, filled)
            if inter.sum() > 0.45 * fixed.sum():
                return inter
    return fixed


def table_roi(image):
    h, w = image.shape[:2]
    contours, _ = cv2.findContours(table_mask(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    default = [int(w * 0.10), int(h * 0.12), int(w * 0.92), int(h * 0.92)]
    if not contours:
        return default
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.10 * w * h:
        return default
    x, y, bw, bh = cv2.boundingRect(contour)
    px, py = int(0.015 * w), int(0.015 * h)
    return _sanitize_roi([x - px, y - py, x + bw + px, y + bh + py], w, h)


def _color_masks(image, roi, tmask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    region = cv2.bitwise_and(_roi_mask(image.shape[:2], roi), tmask.astype(np.uint8))
    out = {}
    for name, ranges in HSV_RANGES.items():
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.asarray(lo), np.asarray(hi)))
        mask = cv2.bitwise_and(mask, region)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        out[name] = mask
    return out


def _dedup(candidates, min_dist=6.0):
    kept = []
    for cand in sorted(candidates, key=lambda d: float(d["confidence"]), reverse=True):
        c = _center(cand)
        if not any(float(np.linalg.norm(c - _center(o))) < min_dist or _bbox_iou(cand["bbox"], o["bbox"]) > 0.45 for o in kept):
            kept.append(cand)
    kept.sort(key=lambda d: (d["centroid"][1], d["centroid"][0]))
    return kept


def _is_rod(color, box, area, extent, mean_hsv):
    if color != "blue":
        return False
    _, y, w, h = box
    ratio = h / max(float(w), 1.0)
    s, v = float(mean_hsv[1]), float(mean_hsv[2])
    return (
        (area >= 1800 and y <= 35)
        or (area >= 900 and y <= 32 and v <= 150 and s <= 190)
        or (area >= 650 and h >= 42 and v <= 155 and ratio >= 0.95)
        or (ratio >= 1.75 and h >= 38 and area >= 360 and v <= 160)
        or (ratio >= 2.25 and h >= 34 and area >= 260)
        or (area >= 900 and y <= 35 and extent < 0.55 and ratio > 1.05)
    )


def _contour_candidate(hsv, color_mask, contour, color, roi, min_area, max_area):
    area = float(cv2.contourArea(contour))
    if area < min_area or area > max_area:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return None
    aspect = float(w / max(h, 1))
    if aspect < 0.18 or aspect > 5.0:
        return None
    m = cv2.moments(contour)
    if abs(m["m00"]) < 1e-6:
        return None
    cx, cy = float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
    x1, y1, x2, y2 = roi
    if not (x1 <= cx <= x2 and y1 <= cy <= y2):
        return None
    inst = np.zeros(color_mask.shape, dtype=np.uint8)
    cv2.drawContours(inst, [contour], -1, 255, thickness=cv2.FILLED)
    pixels = cv2.bitwise_and(inst, color_mask) > 0
    if int(pixels.sum()) < max(8, int(0.15 * area)):
        return None
    mean_hsv = hsv[pixels].mean(axis=0)
    if float(mean_hsv[1]) < 45 or (color == "blue" and float(mean_hsv[1]) < 90):
        return None
    extent = float(area / float(max(w * h, 1)))
    if extent < 0.14 or _is_rod(color, (x, y, w, h), area, extent, mean_hsv):
        return None
    sat_score = float(np.clip((float(mean_hsv[1]) - 45.0) / 150.0, 0.0, 1.0))
    area_score = float(np.clip((area - min_area) / 420.0, 0.0, 1.0))
    extent_score = float(np.clip(extent / 0.70, 0.0, 1.0))
    return {
        "color": color,
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "centroid": [cx, cy],
        "area": area,
        "confidence": float(np.clip(0.34 + 0.28 * sat_score + 0.24 * area_score + 0.14 * extent_score, 0.30, 0.99)),
        "mean_hsv": [float(v) for v in mean_hsv],
        "extent": extent,
    }


def _split_component(color_mask, contour):
    comp = np.zeros(color_mask.shape, dtype=np.uint8)
    cv2.drawContours(comp, [contour], -1, 255, thickness=cv2.FILLED)
    comp = cv2.bitwise_and(comp, color_mask)
    ys, xs = np.nonzero(comp > 0)
    if xs.size < 36:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        major = np.linalg.svd(centered, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        _, _, w, h = cv2.boundingRect(contour)
        major = np.asarray([1.0, 0.0] if w >= h else [0.0, 1.0], dtype=np.float32)
    proj = centered @ major
    sp = proj[np.argsort(proj)]
    gaps = np.diff(sp)
    if gaps.size and float(gaps.max()) > 2.0:
        i = int(np.argmax(gaps))
        split = float(0.5 * (sp[i] + sp[i + 1]))
    else:
        split = float(np.median(proj))
    out = []
    for side in (proj <= split, proj > split):
        if int(side.sum()) < 12:
            continue
        sub = np.zeros_like(comp)
        sub[pts[side, 1].astype(np.int32), pts[side, 0].astype(np.int32)] = 255
        sub = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        sub_contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if sub_contours:
            out.append(max(sub_contours, key=cv2.contourArea))
    return out


def _should_split(det):
    x1, y1, x2, y2 = det["bbox"]
    w, h = x2 - x1, y2 - y1
    area = float(det["area"])
    aspect = w / max(float(h), 1.0)
    return (area >= 650 and max(w, h) >= 26) or (area >= 520 and max(w, h) >= 32 and (aspect >= 1.25 or aspect <= 0.82))


def frame_candidates(image, roi):
    masks = _color_masks(image, roi, table_mask(image, roi))
    masks["yellow"] = cv2.bitwise_or(masks["yellow"], masks["orange"])
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    out = {}
    for color in COLORS:
        mask = masks[color]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        primaries = []
        for contour in contours:
            det = _contour_candidate(hsv, mask, contour, color, roi, MIN_AREA, MAX_AREA)
            if det is not None:
                primaries.append((contour, det))
        cands = _dedup([d for _, d in primaries])
        if len(cands) < 2:
            for contour, det in sorted(primaries, key=lambda x: float(x[1]["area"]), reverse=True):
                if not _should_split(det):
                    continue
                children = []
                for child_contour in _split_component(mask, contour):
                    child = _contour_candidate(hsv, mask, child_contour, color, roi, max(12.0, MIN_AREA * 0.8), MAX_AREA)
                    if child is not None:
                        children.append(child)
                if len(children) >= 2 and sum(float(c["area"]) for c in children) >= 0.40 * float(det["area"]):
                    cands = [c for c in cands if c is not det] + children
                    break
        out[color] = _dedup(cands, min_dist=5.0)
    return out


def choose_two(candidates):
    cands = _dedup(candidates, min_dist=5.0)
    if len(cands) <= 2:
        return copy.deepcopy(cands)
    best, best_score = None, -1e9
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            dist = float(np.linalg.norm(_center(a) - _center(b)))
            if dist < 5.5:
                continue
            area_a, area_b = float(a["area"]), float(b["area"])
            balance = 1.0 - abs(area_a - area_b) / max(area_a, area_b, 1.0)
            score = float(a["confidence"]) + float(b["confidence"]) + 0.15 * min(dist / 28.0, 1.0) + 0.08 * balance
            if score > best_score:
                best_score, best = score, (a, b)
    if best is None:
        return copy.deepcopy(sorted(cands, key=lambda d: float(d["confidence"]), reverse=True)[:2])
    return copy.deepcopy(list(best))


def _assign_tracks(choices_per_frame):
    n = len(choices_per_frame)
    tracks = [[None] * n, [None] * n]
    prev = [None, None]
    for t, choices in enumerate(choices_per_frame):
        if len(choices) >= 2:
            c0, c1 = sorted(choices[:2], key=lambda d: (float(d["centroid"][0]), float(d["centroid"][1])))
            if prev[0] is not None and prev[1] is not None:
                a, b = choices[:2]
                ab = float(np.linalg.norm(_center(a) - prev[0]) + np.linalg.norm(_center(b) - prev[1]))
                ba = float(np.linalg.norm(_center(b) - prev[0]) + np.linalg.norm(_center(a) - prev[1]))
                c0, c1 = (a, b) if ab <= ba else (b, a)
            elif prev[0] is not None or prev[1] is not None:
                k = 0 if prev[0] is not None else 1
                a, b = choices[:2]
                if float(np.linalg.norm(_center(b) - prev[k])) < float(np.linalg.norm(_center(a) - prev[k])):
                    a, b = b, a
                pair = [None, None]
                pair[k], pair[1 - k] = a, b
                c0, c1 = pair
            tracks[0][t], tracks[1][t] = c0, c1
            prev = [_center(c0), _center(c1)]
        elif len(choices) == 1:
            cand = choices[0]
            if prev[0] is None and prev[1] is None:
                k = 0
            elif prev[0] is None:
                k = 1
            elif prev[1] is None:
                k = 0
            else:
                k = 0 if float(np.linalg.norm(_center(cand) - prev[0])) <= float(np.linalg.norm(_center(cand) - prev[1])) else 1
            tracks[k][t] = cand
            prev[k] = _center(cand)
    return tracks


def _heuristic_fill(color, k, t, left, right, fallback):
    if left is not None and right is not None and right[0] != left[0]:
        a = (t - left[0]) / float(right[0] - left[0])
        center = (1.0 - a) * _center(left[1]) + a * _center(right[1])
        size = (1.0 - a) * _bbox_wh(left[1]["bbox"]) + a * _bbox_wh(right[1]["bbox"])
        area = (1.0 - a) * float(left[1]["area"]) + a * float(right[1]["area"])
        conf = 0.55 * min(float(left[1]["confidence"]), float(right[1]["confidence"]))
    elif left is not None or right is not None:
        src = left if left is not None else right
        center, size, area = _center(src[1]), _bbox_wh(src[1]["bbox"]), float(src[1]["area"])
        conf = 0.45 * float(src[1]["confidence"])
    else:
        center, size = _center(fallback), _bbox_wh(fallback["bbox"])
        area = float(fallback.get("area", size[0] * size[1] * 0.6))
        conf = 0.25
    return {
        "color": color,
        "bbox": [
            int(round(center[0] - size[0] / 2.0)),
            int(round(center[1] - size[1] / 2.0)),
            int(round(center[0] + size[0] / 2.0)),
            int(round(center[1] + size[1] / 2.0)),
        ],
        "centroid": [float(center[0]), float(center[1])],
        "area": float(area),
        "confidence": float(np.clip(conf, 0.20, 0.70)),
        "track_id": f"{color}_{k}",
        "interpolated": True,
    }


def heuristic_tracks(candidates_per_frame):
    out = {}
    for color in COLORS:
        tracks = _assign_tracks([choose_two(c[color]) for c in candidates_per_frame])
        known_all = [d for tr in tracks for d in tr if d is not None]
        if known_all:
            fallback = sorted(known_all, key=lambda d: float(d["confidence"]), reverse=True)[0]
        else:
            fallback = {"bbox": [145, 75, 169, 99], "centroid": [157.0, 87.0], "area": 250.0, "confidence": 0.20}
        filled = []
        for k, track in enumerate(tracks):
            known = [i for i, d in enumerate(track) if d is not None]
            row = []
            for t, det in enumerate(track):
                if det is not None:
                    obs = copy.deepcopy(det)
                    obs["track_id"] = f"{color}_{k}"
                    obs["interpolated"] = False
                    row.append(obs)
                    continue
                li = max((i for i in known if i < t), default=None)
                ri = min((i for i in known if i > t), default=None)
                left = (li, track[li]) if li is not None else None
                right = (ri, track[ri]) if ri is not None else None
                row.append(_heuristic_fill(color, k, t, left, right, fallback))
            filled.append(row)
        out[color] = filled
    return out


def _expected_color_mask(image, color):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ranges = HSV_RANGES[color] + (HSV_RANGES["orange"] if color == "yellow" else [])
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.asarray(lo), np.asarray(hi)))
    return mask > 0


def _color_ratio(image, mask, color):
    pixels = mask > 0
    count = int(pixels.sum())
    if count <= 0:
        return 0.0
    return float(np.logical_and(_expected_color_mask(image, color), pixels).sum() / count)


def _largest_component(mask):
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _mask_stats(mask):
    ys, xs = np.nonzero(mask > 0)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], [float(xs.mean()), float(ys.mean())], float(xs.size)


def _robot_like_blue(bbox, area, mask, image):
    x1, y1, x2, y2 = bbox
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    aspect = w / float(h)
    extent = area / float(max(w * h, 1))
    if int(mask.sum()) > 0:
        mean_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[mask > 0].mean(axis=0)
        s, v = float(mean_hsv[1]), float(mean_hsv[2])
    else:
        s, v = 255.0, 255.0
    ratio = h / float(max(w, 1))
    return (
        (y1 <= 30 and ratio >= 1.55 and h >= 34 and area >= 250)
        or (ratio >= 2.1 and h >= 34 and area >= 220)
        or (area >= 700 and y1 <= 36 and extent < 0.55)
        or (ratio >= 1.35 and h >= 38 and s <= 185 and v <= 160)
        or (aspect < 0.45 and h >= 34 and y1 <= 42)
    )


def _mask_det(logits, image, tmask, color, track_id):
    mask = np.logical_and(logits > 0.0, tmask > 0)
    if int(mask.sum()) <= 0:
        return None
    mask = _largest_component(mask)
    stats = _mask_stats(mask)
    if stats is None:
        return None
    bbox, centroid, area = stats
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    aspect = w / float(max(h, 1))
    extent = area / float(max(w * h, 1))
    color_frac = _color_ratio(image, mask, color)
    valid = not (
        area < MASK_MIN_AREA
        or area > MASK_MAX_AREA
        or w < 4
        or h < 4
        or w > 72
        or h > 72
        or aspect < 0.22
        or aspect > 4.2
        or extent < 0.16
        or color_frac < 0.075
    )
    compact = float(np.clip((extent - 0.14) / 0.58, 0.0, 1.0))
    size_score = float(np.clip(1.0 - abs(math.log(max(area, 1.0) / math.sqrt(MASK_MIN_AREA * MASK_MAX_AREA))) / 2.5, 0.0, 1.0))
    color_score = float(np.clip(color_frac / 0.45, 0.0, 1.0))
    if color == "blue" and _robot_like_blue(bbox, area, mask, image):
        valid = False
    return {
        "color": color,
        "bbox": bbox,
        "centroid": centroid,
        "area": float(area),
        "confidence": float(np.clip(0.18 + 0.42 * color_score + 0.25 * compact + 0.15 * size_score, 0.0, 0.99)),
        "track_id": track_id,
        "interpolated": False,
        "valid": valid,
    }


def _bbox_mask(bbox, shape):
    h, w = shape
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _prompt_quality(det, image, tmask):
    if det.get("interpolated"):
        return -1.0
    w, h = _bbox_wh(det["bbox"])
    area = float(det.get("area", w * h))
    aspect = float(w / max(h, 1.0))
    extent = area / max(float(w * h), 1.0)
    color_frac = _color_ratio(image, _bbox_mask(det["bbox"], image.shape[:2]), det["color"])
    compact = float(np.clip((extent - 0.12) / 0.62, 0.0, 1.0))
    size_ok = float(np.clip(1.0 - abs(math.log(max(area, 1.0) / 320.0)) / 2.4, 0.0, 1.0))
    aspect_ok = float(np.clip(1.0 - abs(math.log(max(aspect, 1e-3))) / 1.8, 0.0, 1.0))
    color_ok = float(np.clip(color_frac / 0.35, 0.0, 1.0))
    cx, cy = _center(det)
    inside = 1.0 if tmask[int(round(np.clip(cy, 0, tmask.shape[0] - 1))), int(round(np.clip(cx, 0, tmask.shape[1] - 1)))] else 0.0
    return 0.36 * float(det.get("confidence", 0.0)) + 0.24 * color_ok + 0.16 * compact + 0.14 * size_ok + 0.06 * aspect_ok + 0.04 * inside


def _prompt_det(mask, color, image):
    stats = _mask_stats(mask)
    if stats is None:
        return None
    bbox, centroid, area = stats
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if area < 18 or area > 900 or w < 4 or h < 4 or w > 48 or h > 48:
        return None
    aspect = w / float(max(h, 1))
    extent = area / float(max(w * h, 1))
    if aspect < 0.28 or aspect > 3.6 or extent < 0.16:
        return None
    color_frac = _color_ratio(image, mask, color)
    if color_frac < 0.12:
        return None
    size_score = float(np.clip(1.0 - abs(math.log(max(area, 1.0) / 260.0)) / 1.8, 0.0, 1.0))
    aspect_score = float(np.clip(1.0 - abs(math.log(max(aspect, 1e-3))) / 1.4, 0.0, 1.0))
    extent_score = float(np.clip(extent / 0.72, 0.0, 1.0))
    color_score = float(np.clip(color_frac / 0.55, 0.0, 1.0))
    conf = float(np.clip(0.22 + 0.34 * color_score + 0.20 * size_score + 0.14 * aspect_score + 0.10 * extent_score, 0.0, 0.99))
    return {"color": color, "bbox": bbox, "centroid": centroid, "area": float(area), "confidence": conf}


def _split_prompt_component(comp):
    ys, xs = np.nonzero(comp > 0)
    if xs.size < 36:
        return []
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    centered = pts - pts.mean(axis=0, keepdims=True)
    try:
        major = np.linalg.svd(centered, full_matrices=False)[2][0]
    except np.linalg.LinAlgError:
        major = np.asarray([1.0, 0.0], dtype=np.float32)
    proj = centered @ major
    split = float(np.median(proj))
    out = []
    for side in (proj <= split, proj > split):
        if int(side.sum()) < 16:
            continue
        sub = np.zeros_like(comp, dtype=np.uint8)
        sub[pts[side, 1].astype(np.int32), pts[side, 0].astype(np.int32)] = 255
        out.append(cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0)
    return out if len(out) == 2 else []


def _dedup_prompts(dets, image, tmask):
    kept = []
    for det in sorted(dets, key=lambda d: _prompt_quality(d, image, tmask), reverse=True):
        dup = False
        for old in kept:
            if float(np.linalg.norm(_center(det) - _center(old))) < 4.5:
                dup = True
                break
            a, b = _bbox_mask(det["bbox"], image.shape[:2]), _bbox_mask(old["bbox"], image.shape[:2])
            union = float(np.logical_or(a, b).sum())
            if union > 0 and float(np.logical_and(a, b).sum()) / union > 0.55:
                dup = True
                break
        if not dup:
            kept.append(det)
    return kept


def _color_prompt_pair(image, tmask, color):
    mask = np.logical_and(_expected_color_mask(image, color), tmask > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands, bonus = [], {}
    for contour in contours:
        comp = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(comp, [contour], -1, 255, thickness=cv2.FILLED)
        comp = cv2.bitwise_and(comp, mask)
        raw = _prompt_det(comp > 0, color, image)
        if raw is not None:
            cands.append(raw)
        _, _, w, h = cv2.boundingRect(contour)
        area = float((comp > 0).sum())
        if area >= 360 or max(w, h) >= 28 or (area >= 250 and (w / float(max(h, 1)) >= 1.25 or h / float(max(w, 1)) >= 1.25)):
            children = [c for c in (_prompt_det(m, color, image) for m in _split_prompt_component(comp)) if c is not None]
            if len(children) == 2:
                cands.extend(children)
                for child in children:
                    bonus[id(child)] = 0.10
    cands = _dedup_prompts(cands, image, tmask)
    if len(cands) < 2:
        return None
    quality = {id(d): _prompt_quality(d, image, tmask) for d in cands}
    best_score, best = -1e9, None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            ca, cb = _center(a), _center(b)
            dist = float(np.linalg.norm(ca - cb))
            if dist < 7.0:
                continue
            area_a, area_b = float(a["area"]), float(b["area"])
            balance = 1.0 - abs(area_a - area_b) / max(area_a, area_b, 1.0)
            dist_score = float(np.clip(1.0 - abs(dist - 20.0) / 34.0, 0.0, 1.0))
            row_score = float(np.clip(1.0 - abs(float(ca[1] - cb[1])) / 22.0, 0.0, 1.0))
            far = 0.18 * float(np.clip((dist - 42.0) / 38.0, 0.0, 1.0))
            top = 0.10 * float((a["bbox"][1] <= 23) + (b["bbox"][1] <= 23)) if color == "blue" else 0.0
            score = (
                quality[id(a)]
                + quality[id(b)]
                + 0.12 * dist_score
                + 0.16 * row_score
                + 0.10 * balance
                + bonus.get(id(a), 0.0)
                + bonus.get(id(b), 0.0)
                - far
                - top
            )
            if score > best_score:
                best_score, best = score, [copy.deepcopy(a), copy.deepcopy(b)]
    if best is None:
        return None
    best.sort(key=lambda d: (float(d["centroid"][0]), float(d["centroid"][1])))
    return best_score, best


def _prompt(obj_id, frame_index, det, score, width, height):
    return Prompt(obj_id, int(frame_index), _expand_bbox(det["bbox"], PROMPT_PAD, width, height), [float(v) for v in det["centroid"]], float(score))


def _mask_pair_plans(tracks, frames, tmasks):
    height, width = frames[0].shape[:2]
    plans, obj_id = [], 1
    for color in COLORS:
        best = None
        for t, frame in enumerate(frames):
            pair = _color_prompt_pair(frame, tmasks[t], color)
            if pair is not None and (best is None or pair[0] > best[0]):
                best = (pair[0], t, pair[1])
        for k in range(2):
            if best is not None and best[0] >= 1.45:
                t, det = best[1], best[2][k]
                score = _prompt_quality(det, frames[t], tmasks[t])
            else:
                track = tracks[color][k]
                cands = [(s, t, d) for t, d in enumerate(track) if (s := _prompt_quality(d, frames[t], tmasks[t])) >= 0.35]
                if cands:
                    score, t, det = max(cands, key=lambda x: x[0])
                else:
                    t = int(np.argmax([float(d.get("confidence", 0.0)) for d in track]))
                    score, det = 0.20, track[t]
            plans.append(TrackPlan(obj_id, f"{color}_{k}", color, [_prompt(obj_id, t, det, score, width, height)]))
            obj_id += 1
    return plans


def prompt_plans(tracks, frames, tmasks):
    height, width = frames[0].shape[:2]
    ordered = [(1 + 2 * ci + k, color, f"{color}_{k}", tracks[color][k]) for ci, color in enumerate(COLORS) for k in range(2)]
    anchors = []
    for t in range(len(frames)):
        total, ok, centers = 0.0, True, []
        for _, _, _, track in ordered:
            det = track[t]
            s = _prompt_quality(det, frames[t], tmasks[t])
            if det.get("interpolated") or s < 0.42:
                ok = False
                break
            centers.append(_center(det))
            total += s
        if ok:
            pen = sum(
                max(0.0, 10.0 - float(np.linalg.norm(centers[i] - centers[j]))) * 0.018
                for i in range(len(centers))
                for j in range(i + 1, len(centers))
            )
            anchors.append((total - pen, t))
    if not anchors:
        return _mask_pair_plans(tracks, frames, tmasks)
    anchor = max(anchors)[1]
    plans = []
    for obj_id, color, track_id, track in ordered:
        cands = [(s, t, d) for t, d in enumerate(track) if (s := _prompt_quality(d, frames[t], tmasks[t])) >= 0.35]
        selected = []
        det = track[anchor]
        s = _prompt_quality(det, frames[anchor], tmasks[anchor])
        if s >= 0.35 and not det.get("interpolated"):
            selected.append((s + 0.08, anchor, det))
        if cands and not selected:
            selected.append(max(cands, key=lambda x: x[0]))
        if MAX_PROMPTS_PER_OBJECT > 1 and cands:
            min_gap = max(12, len(frames) // (MAX_PROMPTS_PER_OBJECT + 1))
            while len(selected) < MAX_PROMPTS_PER_OBJECT:
                used = {t for _, t, _ in selected}
                pool = [c for c in cands if c[1] not in used and all(abs(c[1] - u) >= min_gap for u in used)]
                if not pool and min_gap > 8:
                    min_gap = max(8, min_gap // 2)
                    continue
                if not pool:
                    break

                def spread(item):
                    if not used:
                        return item[0]
                    return float(item[0]) + 0.08 * min(min(abs(item[1] - u) for u in used) / max(len(frames), 1), 1.0)

                selected.append(max(pool, key=spread))
        if not selected:
            t = int(np.argmax([float(d.get("confidence", 0.0)) for d in track]))
            selected = [(0.20, t, track[t])]
        selected = sorted(selected, key=lambda x: x[0], reverse=True)[: max(1, MAX_PROMPTS_PER_OBJECT)]
        prompts = [_prompt(obj_id, t, d, s, width, height) for s, t, d in sorted(selected, key=lambda x: x[1])]
        plans.append(TrackPlan(obj_id, track_id, color, prompts))
    return plans


def build_predictor(config, checkpoint, device="cuda"):
    from sam2.build_sam import build_sam2_video_predictor

    device = torch.device(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return build_sam2_video_predictor(config, str(checkpoint), device=device, hydra_overrides_extra=["++model.non_overlap_masks=true"])


def _track(predictor, video_path, frames, tmasks, plans):
    by_obj = {p.obj_id: p for p in plans}
    starts = [q.frame_index for p in plans for q in p.prompts]
    outputs = {"forward": {}, "reverse": {}}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for direction, start, reverse in (("forward", min(starts), False), ("reverse", max(starts), True)):
            state = predictor.init_state(video_path=str(video_path), offload_video_to_cpu=True, offload_state_to_cpu=False)
            for p in plans:
                for q in p.prompts:
                    predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=q.frame_index,
                        obj_id=q.obj_id,
                        points=np.asarray([q.point], dtype=np.float32),
                        labels=np.asarray([1], dtype=np.int32),
                        box=np.asarray(q.bbox, dtype=np.float32),
                        clear_old_points=True,
                        normalize_coords=True,
                    )
            for t, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=int(start), max_frame_num_to_track=len(frames), reverse=reverse
            ):
                logits = logits.detach().cpu().numpy()
                if logits.ndim == 4:
                    logits = logits[:, 0]
                outputs[direction][int(t)] = {
                    by_obj[int(o)].track_id: _mask_det(logits[i], frames[t], tmasks[t], by_obj[int(o)].color, by_obj[int(o)].track_id)
                    for i, o in enumerate(obj_ids)
                }
    return outputs


def _combine(tracks, outputs, n):
    combined = {}
    for color in COLORS:
        for k in range(2):
            track_id = f"{color}_{k}"
            prior = tracks[color][k]
            dets = []
            for t in range(n):
                cands = [d for d in (outputs[x].get(t, {}).get(track_id) for x in ("forward", "reverse")) if d is not None]
                if cands:
                    best = max(cands, key=lambda d: -1e9 if not d["valid"] else float(d["confidence"]) - 0.006 * min(float(np.linalg.norm(_center(d) - _center(prior[t]))), 90.0))
                    dets.append(copy.deepcopy(best) if best["valid"] else None)
                else:
                    dets.append(None)
            prev = None
            for t, det in enumerate(dets):
                if det is None:
                    continue
                c = _center(det)
                if prev is not None and float(np.linalg.norm(c - prev)) > MAX_CENTER_JUMP:
                    dets[t] = None
                    continue
                prev = c
            combined[track_id] = dets
    return combined


def _plausible(cand, color):
    x1, y1, x2, y2 = [int(v) for v in cand["bbox"]]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    area = float(cand["area"])
    extent = float(cand["extent"])
    if area < (55.0 if color in {"yellow", "blue"} else 42.0):
        return False
    if color != "blue" and area > 1200.0 and extent < 0.52:
        return False
    if color == "yellow" and h <= 8 and area < 120.0:
        return False
    if color != "blue":
        return True
    s, v = float(cand["mean_hsv"][1]), float(cand["mean_hsv"][2])
    return not (
        (area > 950.0 and extent < 0.62)
        or v < 95.0
        or (v < 130.0 and s < 165.0)
        or (area >= 420.0 and v < 120.0)
        or (y1 >= 82 and v < 145.0)
        or (w / float(h) < 0.42 and h >= 28 and v < 150.0)
        or (extent < 0.42 and v < 150.0)
    )


def _guard_candidates(cands, color):
    thr = 0.50 if color == "blue" else 0.45
    return [c for c in choose_two(cands) if float(c["confidence"]) >= thr and _plausible(c, color)]


def _assignment(color, cands, sam_dets, heur_dets):
    if not cands:
        return (None, None)

    def score(k, ci):
        sam, heur = sam_dets[k], heur_dets[k]
        if ci is None:
            if sam is None:
                return -0.20
            nearest = min(float(np.linalg.norm(_center(c) - _center(sam))) for c in cands)
            pen = 0.0
            if nearest > SNAP_DIST:
                pen += 0.35 + (0.45 if color == "blue" else 0.0)
            return float(sam["confidence"]) - pen
        c = _center(cands[ci])
        hd = float(np.linalg.norm(c - _center(heur)))
        conf = float(cands[ci]["confidence"])
        if sam is not None:
            sd = float(np.linalg.norm(c - _center(sam)))
            if not (sd <= SNAP_DIST or hd <= RESCUE_DIST):
                return -1e6
            return 1.20 + conf - 0.018 * min(sd, 80.0) - 0.005 * min(hd, 100.0)
        if hd > RESCUE_DIST or conf < 0.52:
            return -1e6
        return 0.92 + conf + (0.12 if not heur.get("interpolated") else 0.0) - 0.010 * min(hd, 120.0)

    n = len(cands)
    options = [(None, None)] + [o for i in range(n) for o in ((i, None), (None, i))] + [(i, j) for i in range(n) for j in range(n) if i != j]
    best_score, best = -1e9, (None, None)
    for o in options:
        s = score(0, o[0]) + score(1, o[1])
        if o[0] is not None and o[1] is not None:
            s += 0.06 * min(float(np.linalg.norm(_center(cands[o[0]]) - _center(cands[o[1]]))) / 28.0, 1.0)
        if s > best_score:
            best_score, best = s, o
    return best


def _supported_by_candidates(color, det, cands):
    if not cands:
        return True
    nearest = min(float(np.linalg.norm(_center(c) - _center(det))) for c in cands)
    if nearest <= SNAP_DIST:
        return True
    return color != "blue" and nearest <= max(42.0, SNAP_DIST * 1.35)


def _snap(combined, candidates_per_frame, tracks):
    out = {k: [copy.deepcopy(d) if d is not None else None for d in v] for k, v in combined.items()}
    for color in COLORS:
        ids = [f"{color}_0", f"{color}_1"]
        for t, per_frame in enumerate(candidates_per_frame):
            cands = _guard_candidates(per_frame[color], color)
            sam_dets = [out[ids[k]][t] for k in range(2)]
            for k, ci in enumerate(_assignment(color, cands, sam_dets, [tracks[color][k][t] for k in range(2)])):
                if ci is not None:
                    det = copy.deepcopy(cands[ci])
                    det.update(track_id=ids[k], interpolated=False, valid=True)
                    det["confidence"] = float(np.clip(float(det["confidence"]) + 0.04, 0.0, 0.995))
                    out[ids[k]][t] = det
                elif sam_dets[k] is not None and not _supported_by_candidates(color, sam_dets[k], cands):
                    out[ids[k]][t] = None
    return out


def _near_candidate(det, cands, dist=18.0):
    return any(float(np.linalg.norm(_center(det) - _center(c))) <= dist or _bbox_iou(det["bbox"], c["bbox"]) >= 0.22 for c in cands)


def _suppress_duplicates(corrected, candidates_per_frame):
    out = {k: [copy.deepcopy(d) if d is not None else None for d in v] for k, v in corrected.items()}
    for color in COLORS:
        ids = [f"{color}_0", f"{color}_1"]
        for t, per_frame in enumerate(candidates_per_frame):
            cands = _guard_candidates(per_frame[color], color)
            dets = [out[ids[k]][t] for k in range(2)]
            if not cands:
                continue
            for i, det in enumerate(dets):
                other = dets[1 - i]
                if det is None or other is None:
                    continue
                same = _bbox_iou(det["bbox"], other["bbox"]) >= 0.35 or float(np.linalg.norm(_center(det) - _center(other))) <= 12.0
                if not same:
                    continue
                if _near_candidate(det, cands) and _near_candidate(other, cands) and len(cands) >= 2:
                    continue
                loser = i if float(det["confidence"]) < float(other["confidence"]) else 1 - i
                out[ids[loser]][t] = None
    return out


def _fill_one(color, track_id, t, left, right, fallback, width, height):
    if left is not None and right is not None and left[0] != right[0]:
        if right[0] - left[0] <= MAX_INTERP_GAP and float(np.linalg.norm(_center(left[1]) - _center(right[1]))) <= MAX_INTERP_DIST:
            a = (t - left[0]) / float(right[0] - left[0])
            c = (1.0 - a) * _center(left[1]) + a * _center(right[1])
            size = (1.0 - a) * _bbox_wh(left[1]["bbox"]) + a * _bbox_wh(right[1]["bbox"])
            area = (1.0 - a) * float(left[1]["area"]) + a * float(right[1]["area"])
        else:
            c, size, area = _center(left[1]), _bbox_wh(left[1]["bbox"]), float(left[1]["area"])
    elif left is not None and t - left[0] <= MAX_ONE_SIDED_FILL:
        c, size, area = _center(left[1]), _bbox_wh(left[1]["bbox"]), float(left[1]["area"])
    elif right is not None and right[0] - t <= MAX_ONE_SIDED_FILL:
        c, size, area = _center(right[1]), _bbox_wh(right[1]["bbox"]), float(right[1]["area"])
    elif left is not None:
        c, size, area = _center(left[1]), _bbox_wh(left[1]["bbox"]), float(left[1]["area"])
    elif right is not None:
        c, size, area = _center(right[1]), _bbox_wh(right[1]["bbox"]), float(right[1]["area"])
    else:
        c, size = _center(fallback), _bbox_wh(fallback["bbox"])
        area = float(fallback.get("area", size[0] * size[1] * 0.65))
    bbox = _clip_bbox([float(c[0] - size[0] / 2.0), float(c[1] - size[1] / 2.0), float(c[0] + size[0] / 2.0), float(c[1] + size[1] / 2.0)], width, height)
    return {
        "color": color,
        "track_id": track_id,
        "centroid": [float(c[0]), float(c[1])],
        "area": float(area),
        "bbox": [int(round(v)) for v in bbox],
        "interpolated": True,
    }


def _fill(corrected, tracks, plans, width, height):
    fallbacks = {}
    for p in plans:
        q = max(p.prompts, key=lambda x: x.score)
        x1, y1, x2, y2 = q.bbox
        fallbacks[p.track_id] = {
            "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
            "centroid": [float(q.point[0]), float(q.point[1])],
            "area": max(40.0, float((x2 - x1) * (y2 - y1) * 0.65)),
        }
    out = {}
    for color in COLORS:
        for k in range(2):
            track_id = f"{color}_{k}"
            track = corrected[track_id]
            known = [i for i, d in enumerate(track) if d is not None]
            row = []
            for t, det in enumerate(track):
                if det is not None:
                    row.append(
                        {
                            "color": color,
                            "track_id": track_id,
                            "centroid": [float(v) for v in det["centroid"]],
                            "area": float(det["area"]),
                            "bbox": [int(v) for v in det["bbox"]],
                            "interpolated": False,
                        }
                    )
                    continue
                li = max((i for i in known if i < t), default=None)
                ri = min((i for i in known if i > t), default=None)
                left = (li, track[li]) if li is not None else None
                right = (ri, track[ri]) if ri is not None else None
                local = tracks[color][k][t]
                if local.get("interpolated"):
                    local = fallbacks[track_id]
                row.append(_fill_one(color, track_id, t, left, right, local, width, height))
            out[track_id] = row
    return out


def detect(predictor, video_path):
    frames = [np.asarray(f[..., :3], dtype=np.uint8) for f in iio.imiter(video_path)]
    height, width = frames[0].shape[:2]
    roi = [int(v) for v in table_roi(frames[0])]
    roi[1] = max(0, roi[1] - TABLE_TOP_EXTENSION)
    tmasks = [table_mask(f, roi) > 0 for f in frames]
    candidates = [frame_candidates(f, roi) for f in frames]
    tracks = heuristic_tracks(candidates)
    plans = prompt_plans(tracks, frames, tmasks)
    combined = _combine(tracks, _track(predictor, video_path, frames, tmasks, plans), len(frames))
    corrected = _suppress_duplicates(_snap(combined, candidates, tracks), candidates)
    filled = _fill(corrected, tracks, plans, width, height)
    out = []
    for t in range(len(frames)):
        dets = [filled[f"{c}_{k}"][t] for c in COLORS for k in range(2)]
        dets.sort(key=lambda d: (COLOR_ORDER[d["color"]], d["track_id"]))
        out.append(dets)
    return {"table_roi": roi, "frames": out}
