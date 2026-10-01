"""Launch SimpleDesign training from a YAML config (keys = TrainerArgs fields).

Single process (cuda > mps > cpu):
    uv run python scripts/train.py --config configs/train_config.yaml
Multi-GPU, one node (e.g. inside a SLURM job):
    uv run torchrun --standalone --nproc_per_node=4 scripts/train.py --config ...
"""

import argparse
import os

import torch
import torch.distributed as dist
import yaml

from simpledesign.training.train import Trainer, TrainerArgs


def setup_device() -> torch.device:
    """Under torchrun (WORLD_SIZE > 1): init the process group, NCCL with one GPU per process,
    gloo on CPU. Otherwise pick the best local device."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        if torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl")
            return torch.device("cuda", local_rank)
        dist.init_process_group("gloo")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SimpleDesign training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--device", default=None, help="single process only, e.g. cpu")
    cli = parser.parse_args()

    with open(cli.config) as f:
        args = TrainerArgs(**yaml.safe_load(f))
    torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere+ GPUs
    device = torch.device(cli.device) if cli.device else setup_device()
    try:
        Trainer(args, device).train()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
