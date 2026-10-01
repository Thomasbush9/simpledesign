import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import wandb
from rich import print
from rich.progress import track
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer

from simpledesign.data.dataset import ProteinCollator, ProteinDataset
from simpledesign.models.loss import joint_loss, masked_accuracy
from simpledesign.models.simpledesign import SimpleDesign
from simpledesign.models.utils import (
    aligned_velocity_target,
    augment,
    corrupt,
    sample_timesteps,
)
from simpledesign.viz.visualization import plot_history

SCALARS = ("total", "seq", "struct", "acc", "grad_norm")


@dataclass
class TrainerArgs:
    # model
    esm_ck: str
    sigma: float
    # data
    data_dir: str
    max_len: int | None = 256  # random contiguous crop (residues); None: no crop
    batch_size: int = 4  # proteins per GPU
    n_replicas: int = 1  # views per protein; the model sees batch_size * n_replicas
    num_workers: int = 0
    cache: bool = True  # keep parsed proteins in memory
    # optimization
    num_steps: int = 10_000
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lambda_seq: float = 1.0
    lambda_struct: float = 1.0
    seed: int = 0
    fixed_corruption: bool = False  # overfit check: one batch, one corruption, every step
    translation_std: float = 0.0  # Angstrom, random shift applied after the random rotation
    # logging / outputs (rank 0 only)
    out_dir: str = "runs/debug"
    log_every: int = 100
    plot_history: bool = True
    plot_smooth: int = 20
    use_wandb: bool = False
    wandb_project: str = "simpledesign"
    wandb_name: str | None = None
    # checkpoints
    ckpt_dir: str | None = None  # None: never save
    ckpt_every: int = 1_000
    ckpt_keep_all: bool = False  # False: overwrite ckpt_dir/last.pt, True: step_XXXXXXX.pt
    resume_from: str | None = None


class Trainer:
    """Step-based trainer, single process or DDP. Under torchrun the launcher owns the process
    group: init it before building the Trainer, destroy it after `train()`."""

    def __init__(self, args: TrainerArgs, device: str | torch.device):
        self.args = args
        self.device = torch.device(device)
        self.ddp = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.ddp else 0
        self.world_size = dist.get_world_size() if self.ddp else 1
        self.is_main = self.rank == 0

        torch.manual_seed(args.seed)  # same init on every rank
        self.model = SimpleDesign.from_esm2(args.esm_ck, sigma=args.sigma).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0
        self.history = []
        if args.resume_from:
            self.load(args.resume_from)
        # different noise per rank, and no replay of the first steps' noise after a resume
        torch.manual_seed(args.seed + self.world_size * self.step + self.rank)
        # checkpoints hold the bare model: wrap after loading
        self.ddp_model = (
            DDP(self.model, device_ids=[self.device.index] if self.device.type == "cuda" else None)
            if self.ddp
            else self.model
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.esm_ck)
        ds = ProteinDataset(args.data_dir, cache=args.cache)
        self.sampler = DistributedSampler(ds, seed=args.seed, drop_last=True) if self.ddp else None
        self.loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            drop_last=True,
            collate_fn=ProteinCollator(self.tokenizer, max_len=args.max_len),
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,  # keeps the per-worker dataset cache
            pin_memory=self.device.type == "cuda",
        )
        assert len(self.loader) > 0, (
            f"{len(ds)} proteins < batch_size {args.batch_size} x {self.world_size} ranks"
        )

    def batches(self):
        """Endless stream of device batches; every epoch reshuffles."""
        epoch = self.step // len(self.loader)
        while True:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            for batch in self.loader:
                yield {
                    k: v.to(self.device, non_blocking=True)
                    for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
            epoch += 1

    def make_views(self, batch: dict) -> tuple[dict, dict]:
        """Repeat each protein n_replicas times; each replica gets its own random rigid motion,
        (t, t'), sequence mask and structure noise."""
        b = {k: v.repeat_interleave(self.args.n_replicas, dim=0) for k, v in batch.items()}
        b["coords"] = augment(b["coords"], b["struct_mask"], self.args.translation_std)
        t, t_prime = sample_timesteps(b["seq"].shape[0], self.device)
        seq_t, coords_t, noise_mask, _, eps, x_clean = corrupt(b, t, t_prime)
        views = {
            "t": t,
            "t_prime": t_prime,
            "seq_t": seq_t,
            "coords_t": coords_t,
            "noise_mask": noise_mask,
            "eps": eps,
            "x_clean": x_clean,
        }
        return b, views

    def training_step(self, b: dict, v: dict) -> dict:
        """One optimizer step. Returns detached device tensors (no host sync)."""
        logits, velocity = self.ddp_model(
            v["seq_t"],
            v["coords_t"],
            b["seq_mask"],
            b["struct_mask"],
            b["idx"],
            v["t"],
            v["t_prime"],
        )
        v_target = aligned_velocity_target(
            v["x_clean"], v["eps"], v["coords_t"], velocity, v["t_prime"], b["struct_mask"]
        )
        total, parts = joint_loss(
            logits,
            b["seq"],
            v["noise_mask"],
            velocity,
            v_target,
            b["struct_mask"],
            v["t"],
            lambda_seq=self.args.lambda_seq,
            lambda_struct=self.args.lambda_struct,
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        return {
            "total": total.detach(),
            "seq": parts["seq"],
            "struct": parts["struct"],
            "acc": masked_accuracy(logits.detach(), b["seq"], v["noise_mask"]),
            "grad_norm": grad_norm,
            "t": v["t"],
            "t_prime": v["t_prime"],
        }

    def log(self, buffer: list[tuple[int, dict]]) -> None:
        """Move buffered step stats to the host in one go, append to history, print the window
        mean, send per-step scalars to wandb."""
        keys = buffer[0][1].keys()
        host = {k: torch.stack([s[k] for _, s in buffer]).float().cpu().tolist() for k in keys}
        rows = [{"step": s, **{k: host[k][i] for k in keys}} for i, (s, _) in enumerate(buffer)]
        self.history.extend(rows)
        m = {k: float(np.nanmean([r[k] for r in rows])) for k in SCALARS}
        print(
            f"step {self.step:6d} | total {m['total']:.4f} | seq {m['seq']:.4f} acc {m['acc']:.2f} "
            f"| struct {m['struct']:.4f} | |g| {m['grad_norm']:.2f}"
        )
        if self.args.use_wandb:
            for r in rows:
                wandb.log({k: r[k] for k in SCALARS}, step=r["step"])

    def save(self) -> None:
        ckpt_dir = Path(self.args.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        name = f"step_{self.step:07d}.pt" if self.args.ckpt_keep_all else "last.pt"
        tmp = ckpt_dir / f"{name}.tmp"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "history": self.history,
                "args": asdict(self.args),
            },
            tmp,
        )
        os.replace(tmp, ckpt_dir / name)  # atomic: a job killed mid-save keeps the old file
        print(f"saved {ckpt_dir / name}")

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.history = ckpt["history"]
        if self.is_main:
            print(f"resumed from {path} at step {self.step}")

    def train(self) -> SimpleDesign:
        a = self.args
        if self.is_main:
            Path(a.out_dir).mkdir(parents=True, exist_ok=True)
            if a.use_wandb:
                wandb.init(
                    project=a.wandb_project, name=a.wandb_name, config=asdict(a), dir=a.out_dir
                )

        self.model.train()
        batches = self.batches()
        fixed = None
        buffer = []
        for step in track(
            range(self.step, a.num_steps), description="Training ", disable=not self.is_main
        ):
            if fixed is None:
                b, v = self.make_views(next(batches))
                if a.fixed_corruption:
                    fixed = (b, v)
            else:
                b, v = fixed
            stats = self.training_step(b, v)
            self.step = step + 1

            if not self.is_main:
                continue
            buffer.append((self.step, stats))
            last = self.step == a.num_steps
            save = a.ckpt_dir is not None and (self.step % a.ckpt_every == 0 or last)
            if self.step % a.log_every == 0 or save or last:
                self.log(buffer)
                buffer = []
            if save:
                self.save()

        if self.is_main:
            if a.plot_history and self.history:
                fig = plot_history(self.history, smooth=a.plot_smooth)
                fig.savefig(Path(a.out_dir) / "history.png", dpi=150)
                if a.use_wandb:
                    wandb.log({"history": wandb.Image(fig)}, step=self.step)
                plt.close(fig)
            if a.use_wandb:
                wandb.finish()
        return self.model
