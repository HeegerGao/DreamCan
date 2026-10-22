import argparse
import json
import os

import numpy as np

from dreamcan.lt_progress import load_detections
from dreamcan.lt_success import calibrate, passes, plan_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--calibrate", default=None)
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    with open(os.path.join(args.data, "episodes.json")) as f:
        episodes = json.load(f)
    rows = {}
    for name in sorted(os.listdir(args.detections)):
        key = name.removesuffix(".json")
        rows[key] = plan_metrics(args.task, load_detections(os.path.join(args.detections, name)), episodes[key])

    if args.calibrate:
        thresholds = calibrate(list(rows.values()), args.task, args.quantile)
        with open(args.calibrate, "w") as f:
            json.dump(thresholds, f, indent=1)
        print(json.dumps(thresholds, indent=1))
        return

    with open(args.thresholds) as f:
        thresholds = json.load(f)
    results = {}
    for key, metrics in rows.items():
        success, gates = passes(metrics, thresholds, args.task)
        results[key] = {"success": success, "gates": gates, "metrics": metrics}
    summary = {
        "success_rate": float(np.mean([r["success"] for r in results.values()])),
        "gate_rates": {g: float(np.mean([r["gates"][g] for r in results.values()])) for g in next(iter(results.values()))["gates"]},
        "num_plans": len(results),
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({**summary, "plans": results}, f, indent=1)


if __name__ == "__main__":
    main()
