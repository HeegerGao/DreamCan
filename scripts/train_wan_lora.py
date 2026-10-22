import argparse
import datetime
import os

import accelerate
import torch

from dreamcan.wan import SubgoalVideoDataset, load_wan, save_lora


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--subgoals", required=True)
    p.add_argument("--wan_root", required=True)
    p.add_argument("--output_dir", default="checkpoints/wan_lora")
    p.add_argument("--num_subgoals", type=int, default=10)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--prompt_dropout", type=float, default=0.0)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_steps", type=int, default=220500)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    accelerate.utils.set_seed(args.seed)
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False),
            accelerate.InitProcessGroupKwargs(timeout=datetime.timedelta(hours=2)),
        ]
    )
    dataset = SubgoalVideoDataset(args.data, args.subgoals, args.num_subgoals, args.size, args.prompt_dropout)
    model = load_wan(args.wan_root, accelerator.device, lora_rank=args.lora_rank)
    opt = torch.optim.AdamW(model.trainable_modules(), lr=args.lr, weight_decay=args.weight_decay)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
        persistent_workers=args.num_workers > 0,
    )
    model, opt, loader = accelerator.prepare(model, opt, loader)
    os.makedirs(args.output_dir, exist_ok=True)

    step = 0
    while step < args.max_steps:
        for data in loader:
            loss = model(data)
            accelerator.backward(loss)
            opt.step()
            opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                loss = accelerator.gather(loss.detach()).mean().item()
                if accelerator.is_main_process:
                    print(f"step {step} loss {loss:.4f}", flush=True)
            if step % args.save_every == 0 or step == args.max_steps:
                save_lora(accelerator, model, os.path.join(args.output_dir, f"lora_{step:07d}.safetensors"))
            if step >= args.max_steps:
                break


if __name__ == "__main__":
    main()
