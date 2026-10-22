import argparse
import json
import os
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from dreamcan.lt_progress import TASKS, episode_progress, load_detections


def run(item, task, detections):
    episode, instruction = item
    det = load_detections(os.path.join(detections, f"{episode}.json"))
    return episode, [float(v) for v in episode_progress(task, det, instruction)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--detections", required=True)
    p.add_argument("--episodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    with open(args.episodes) as f:
        episodes = json.load(f)
    fn = partial(run, task=args.task, detections=args.detections)
    with Pool(args.workers) as pool:
        progress = dict(tqdm(pool.imap(fn, sorted(episodes.items())), total=len(episodes)))
    with open(args.out, "w") as f:
        json.dump(progress, f)


if __name__ == "__main__":
    main()
