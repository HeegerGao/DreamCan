import argparse
import json
import os

import h5py
import numpy as np
from tqdm import tqdm

from dreamcan.common import episode_keys, instruction_of
from dreamcan.subgoals import NUM_SUBGOALS, annotate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--value_pairs", required=True)
    p.add_argument("--progress", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_subgoals", type=int, default=None)
    args = p.parse_args()

    with open(args.progress) as f:
        progress = json.load(f)
    subgoals = {}
    with h5py.File(args.data, "r") as f:
        for key in tqdm(episode_keys(f)):
            v_pair = np.load(os.path.join(args.value_pairs, f"{key}.npz"))["v_pair"]
            k = args.num_subgoals or NUM_SUBGOALS[instruction_of(f["episodes"][key]).strip()]
            subgoals[key] = annotate(v_pair, np.asarray(progress[key]), k)
    with open(args.out, "w") as f:
        json.dump(subgoals, f, indent=1)


if __name__ == "__main__":
    main()
