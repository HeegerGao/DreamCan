import argparse
import json
import os

import torch.multiprocessing as mp


def worker(rank, gpus, args, videos):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])
    from dreamcan.lt_detection import build_predictor, detect

    predictor = build_predictor(args.sam2_config, args.sam2_checkpoint, "cuda")
    for name in videos[rank :: len(gpus)]:
        out = os.path.join(args.out, f"{name}.json")
        if os.path.exists(out):
            continue
        result = detect(predictor, os.path.join(args.videos, f"{name}.mp4"))
        with open(out + ".tmp", "w") as f:
            json.dump(result, f)
        os.replace(out + ".tmp", out)
        print(f"[gpu {gpus[rank]}] {name}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--sam2_checkpoint", required=True)
    p.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--episodes", default=None)
    p.add_argument("--gpus", default="0")
    args = p.parse_args()

    if args.episodes:
        with open(args.episodes) as f:
            videos = sorted(json.load(f))
    else:
        videos = sorted(n[:-4] for n in os.listdir(args.videos) if n.endswith(".mp4"))
    os.makedirs(args.out, exist_ok=True)
    gpus = [int(g) for g in args.gpus.split(",")]
    mp.spawn(worker, args=(gpus, args, videos), nprocs=len(gpus))


if __name__ == "__main__":
    main()
