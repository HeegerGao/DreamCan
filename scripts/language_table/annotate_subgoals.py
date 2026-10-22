import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from dreamcan.lt_subgoals import annotate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--value_pairs", required=True)
    p.add_argument("--progress", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_subgoals", type=int, default=16)
    args = p.parse_args()

    with open(args.progress) as f:
        progress = json.load(f)
    subgoals = {}
    for key in tqdm(sorted(progress)):
        v_pair = np.load(os.path.join(args.value_pairs, f"{key}.npz"))["v_pair"]
        subgoals[key] = annotate(v_pair, np.asarray(progress[key]), args.num_subgoals)
    with open(args.out, "w") as f:
        json.dump(subgoals, f, indent=1)


if __name__ == "__main__":
    main()
