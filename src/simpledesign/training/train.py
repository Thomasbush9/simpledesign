import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

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

from simpledesign.data.afdb import sample_afdb
from simpledesign.data.dataset import ProteinCollator, ProteinDataset
from simpledesign.data.parsing import write_ca_structure
from simpledesign.models.loss import joint_loss, masked_accuracy
from simpledesign.models.simpledesign import SimpleDesign
from simpledesign.models.utils import (
    SIGMA_DATA,
    aligned_velocity_target,
    augment,
    ca_rmsd,
    corrupt,
    sample_timesteps,
    structure_rigid_alignment,
)
from simpledesign.viz.visualization import (
    plot_history,
    plot_joint_history,
    plot_sequence,
    plot_velocity,
)

SCALARS = ("total", "seq", "struct", "fape", "acc", "grad_norm")
JOINT_METRICS = ("seq_agreement", "struct_mse", "struct_rmsd", "struct_rmsd_mirror")


@dataclass
class TrainerArgs:
    # model
    esm_ck: str
    sigma: float
    # data
    data_dir: str
    afdb_tar: str | None = None  # either tar or log enables AFDB preparation
    afdb_log: str | None = None  # existing: replay; new: save; None: data_dir/selection.json
    afdb_n: int = 100
    afdb_min_plddt: float = 90.0
    afdb_seed: int = 0  # independent of per-rank training randomness
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
    beta_fape: float = 0.0  # normalized CA-frame FAPE of the predicted clean structure
    seed: int = 0
    fixed_corruption: bool = False  # overfit check: one batch, one corruption, every step
    translation_std: float = 0.0  # Angstrom, random shift applied after the random rotation
    # validation args:
    joint_eval_every: int | None = 5_000  # None: off; otherwise periodic and final step
    joint_eval_length: int = 89
    joint_eval_n_samples: int = 2
    joint_eval_n_steps: int = 200
    joint_eval_seed: int = 1234
    # logging / outputs (rank 0 only)
    out_dir: str = "runs/debug"
    log_every: int = 100
    plot_history: bool = True
    plot_smooth: int = 20
    plot_every: int | None = None  # None: off; velocity/sequence figures of a fixed protein
    viz_levels: tuple[float, ...] = (0.25, 0.5, 0.75)  # t = t' of each figure row
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

        ds = self._load_dataset()

        torch.manual_seed(args.seed)  # same init on every rank
        self.model = SimpleDesign.from_esm2(args.esm_ck, sigma=args.sigma).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        self.step = 0
        self.history = []
        self._viz = None  # (batch, views) for plot(), built on first use
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

    def _load_dataset(self) -> ProteinDataset:
        a = self.args
        if a.afdb_tar is None and a.afdb_log is None:
            return ProteinDataset(a.data_dir, cache=a.cache)
        ds = None
        status = [None]
        if self.is_main:
            try:
                ds = sample_afdb(
                    a.afdb_tar, a.data_dir, a.afdb_n, a.afdb_min_plddt,
                    a.afdb_seed, a.afdb_log, cache=a.cache,
                )
            except Exception as error:
                if not self.ddp:
                    raise
                status[0] = f"{type(error).__name__}: {error}"
        if self.ddp:
            # All ranks observe preparation failures instead of training on partial output.
            dist.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(f"AFDB preparation failed on rank 0: {status[0]}")
        return ds if ds is not None else ProteinDataset(a.data_dir, cache=a.cache)

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
            beta_fape=self.args.beta_fape,
            x_pred=(
                v["coords_t"] + (1 - v["t_prime"])[:, None, None] * velocity
                if self.args.beta_fape > 0 else None
            ),
            x_true=v["x_clean"] if self.args.beta_fape > 0 else None,
        )
        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.optimizer.step()
        return {
            "total": total.detach(),
            "seq": parts["seq"],
            "struct": parts["struct"],
            "fape": parts["fape"],
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
            f"| struct {m['struct']:.4f} | fape {m['fape']:.4f} | |g| {m['grad_norm']:.2f}"
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

    def viz_inputs(self) -> tuple[dict, dict]:
        """First dataset protein, one row per viz level (t = t'), no augmentation. Built on CPU
        under a fixed seed: the same crop and noise in every figure, also across resumes."""
        levels = torch.tensor(self.args.viz_levels, dtype=torch.float32)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.args.seed)
            batch = self.loader.collate_fn([self.loader.dataset[0]])
            b = {
                k: v.repeat_interleave(len(levels), dim=0)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            seq_t, coords_t, noise_mask, _, eps, x_clean = corrupt(b, levels, levels)
        v = {
            "t": levels,
            "t_prime": levels,
            "seq_t": seq_t,
            "coords_t": coords_t,
            "noise_mask": noise_mask,
            "eps": eps,
            "x_clean": x_clean,
        }
        b = {k: x.to(self.device) for k, x in b.items()}
        return b, {k: x.to(self.device) for k, x in v.items()}

    @torch.no_grad()
    def plot(self) -> None:
        """Velocity + sequence figures -> out_dir/viz/<name>_<step>.png and wandb viz/<name>."""
        if self._viz is None:
            self._viz = self.viz_inputs()
        b, v = self._viz
        self.model.eval()  # bare model: a DDP forward on rank 0 alone could hang
        logits, velocity = self.model(
            v["seq_t"],
            v["coords_t"],
            b["seq_mask"],
            b["struct_mask"],
            b["idx"],
            v["t"],
            v["t_prime"],
        )
        self.model.train()
        v_target = aligned_velocity_target(
            v["x_clean"], v["eps"], v["coords_t"], velocity, v["t_prime"], b["struct_mask"]
        )
        figs = {
            "velocity": plot_velocity(
                v["coords_t"], velocity, v_target, v["t_prime"], b["struct_mask"]
            ),
            "sequence": plot_sequence(logits, b["seq"], v["noise_mask"], v["t"], b["struct_mask"]),
        }
        out = Path(self.args.out_dir) / "viz"
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in figs.items():
            fig.suptitle(f"step {self.step}")
            fig.savefig(out / f"{name}_{self.step:07d}.png", dpi=120)
        if self.args.use_wandb:
            wandb.log({f"viz/{k}": wandb.Image(fig) for k, fig in figs.items()}, step=self.step)
        for fig in figs.values():
            plt.close(fig)

    @torch.no_grad()
    def evaluate_joint(self) -> dict[str, torch.Tensor]:
        """Seeded internal round trips and per-sample metrics on the model device.

        Structures/RMSD are in Angstrom; MSE averages squared error over residues and xyz.
        Sampling preserves training RNG state and model mode, including on MPS.
        """
        a = self.args
        rng_devices = [] if self.device.type == "cpu" else [self.device]
        was_training = self.model.training
        with torch.random.fork_rng(devices=rng_devices, device_type=self.device.type):
            torch.random.default_generator.manual_seed(a.joint_eval_seed)
            if rng_devices:
                getattr(torch, self.device.type).manual_seed(a.joint_eval_seed)
            try:
                seq, struct = self.model.joint_sample(
                    [a.joint_eval_length] * a.joint_eval_n_samples,
                    a.joint_eval_n_steps,
                    struct_schedule_type="log",
                )
                # All samples have the same length: cls, residues, eos, with no padding.
                idx = torch.arange(seq.shape[1], device=seq.device).expand_as(seq)
                seq_mask = torch.ones_like(seq, dtype=torch.bool)
                struct_mask = (idx > 0) & (idx <= a.joint_eval_length)
                inverted_folding_seq = self.model.sample_seq(
                    seq_mask=seq_mask,
                    struct_mask=struct_mask,
                    idx=idx,
                    struct=struct / SIGMA_DATA,
                    t_struct=torch.ones(seq.shape[0], device=seq.device),
                    n_steps=a.joint_eval_n_steps,
                )
                folded_structure = self.model.sample_struct(
                    seq=seq,
                    seq_mask=seq_mask,
                    struct_mask=struct_mask,
                    idx=idx,
                    n_steps=a.joint_eval_n_steps,
                    struct_schedule_type="log",
                )
            finally:
                self.model.train(was_training)

        n_residues = struct_mask.sum(dim=1)
        matches = seq == inverted_folding_seq
        agreement = (matches & struct_mask).sum(dim=1) / n_residues
        # Reuse the loss's Kabsch alignment; proper rotations only, no CPU transfer.
        aligned = structure_rigid_alignment(folded_structure.float(), struct.float(), struct_mask)
        squared_error = (aligned - struct.float()).square().sum(dim=-1)
        mse = (squared_error * struct_mask).sum(dim=1) / (3 * n_residues)
        # Reflect a copy of the refolded structure, then align; never alter the saved sample.
        reflected = folded_structure * folded_structure.new_tensor([1.0, 1.0, -1.0])
        rmsd_mirror = ca_rmsd(reflected, struct, struct_mask)

        return {
            "seq": seq,
            "struct": struct,
            "inverted_folding_seq": inverted_folding_seq,
            "folded_structure": folded_structure,
            "seq_mask": seq_mask,
            "struct_mask": struct_mask,
            "idx": idx,
            "seq_agreement": agreement,
            "struct_mse": mse,
            "struct_rmsd": (3 * mse).sqrt(),
            "struct_rmsd_mirror": rmsd_mirror,
        }

    def log_joint_eval(self, result: dict[str, torch.Tensor]) -> None:
        """Save immutable step-numbered samples, per-sample metrics, and learning curves."""
        root = Path(self.args.out_dir) / "joint_eval"
        root.mkdir(parents=True, exist_ok=True)
        out = root / f"step_{self.step:07d}"
        if out.exists():
            raise FileExistsError(f"evaluation already exists: {out}; use a new out_dir")
        # Metrics stay on-device until this reporting boundary; transfer each tensor once.
        cpu = {k: v.detach().cpu() for k, v in result.items()}
        rows = [
            {
                "step": self.step,
                "sample": f"sample_{i:04d}",
                "length": int(cpu["struct_mask"][i].sum()),
                **{key: float(cpu[key][i]) for key in JOINT_METRICS},
            }
            for i in range(len(cpu["seq"]))
        ]
        # Rebuild from saved evaluations so curves also include evaluations before a resume.
        history = []
        for path in sorted(root.glob("step_*/metrics.tsv")):
            with path.open(newline="") as f:
                history.extend(
                    r for r in csv.DictReader(f, delimiter="\t") if int(r["step"]) < self.step
                )
        fig = plot_joint_history(history + rows)
        try:
            # Publish a complete sample set atomically; never replace an earlier evaluation.
            with TemporaryDirectory(prefix=".joint_eval_", dir=root) as temp:
                stage = Path(temp) / out.name
                stage.mkdir()
                metadata = {
                    "step": self.step,
                    "args": asdict(self.args),
                    "structure_units": "Angstrom",
                    "structure_representation": "CA-only",
                    "struct_schedule_type": "log",
                    "evaluation": "SimpleDesign internal conditional round trips",
                    "metrics": {
                        "seq_agreement": "matching residue token fraction",
                        "struct_mse": "Kabsch-aligned mean squared xyz error (Angstrom^2)",
                        "struct_rmsd": "Kabsch-aligned CA RMSD (Angstrom)",
                        "struct_rmsd_mirror": "CA RMSD after reflection then Kabsch (Angstrom)",
                    },
                }
                (stage / "args.json").write_text(json.dumps(metadata, indent=2) + "\n")
                torch.save(cpu, stage / "samples.pt")
                with (stage / "metrics.tsv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
                    writer.writeheader()
                    writer.writerows(rows)
                for i, row in enumerate(rows):
                    sample = stage / row["sample"]
                    sample.mkdir()
                    mask = cpu["struct_mask"][i]
                    seq = "".join(
                        self.tokenizer.convert_ids_to_tokens(cpu["seq"][i, mask].tolist())
                    )
                    inverse = "".join(
                        self.tokenizer.convert_ids_to_tokens(
                            cpu["inverted_folding_seq"][i, mask].tolist()
                        )
                    )
                    (sample / "sequence.fasta").write_text(f">{row['sample']}\n{seq}\n")
                    (sample / "inverse_folded.fasta").write_text(
                        f">{row['sample']}_inverse_folded\n{inverse}\n"
                    )
                    # Preserve original coordinates, not aligned copies, for independent checking.
                    write_ca_structure(
                        sample / "structure.pdb", seq, cpu["struct"][i, mask].numpy()
                    )
                    write_ca_structure(
                        sample / "folded_structure.pdb",
                        seq,
                        cpu["folded_structure"][i, mask].numpy(),
                    )
                fig.savefig(stage / "metrics.png", dpi=130)
                stage.rename(out)
            means = {key: float(np.mean([r[key] for r in rows])) for key in JOINT_METRICS}
            print(
                f"joint eval {self.step:6d} | agreement {means['seq_agreement']:.3f} "
                f"| MSE {means['struct_mse']:.3f} A^2 | RMSD {means['struct_rmsd']:.3f} A"
                f" | reflected RMSD {means['struct_rmsd_mirror']:.3f} A"
            )
            print(f"wrote {out}")
            if self.args.use_wandb:
                wandb.log(
                    {
                        **{f"joint_eval/{k}": v for k, v in means.items()},
                        "joint_eval/metrics": wandb.Image(fig),
                    },
                    step=self.step,
                )
        finally:
            plt.close(fig)

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
            joint_eval = a.joint_eval_every is not None and (
                self.step % a.joint_eval_every == 0 or last
            )
            save = a.ckpt_dir is not None and (self.step % a.ckpt_every == 0 or last)
            plot = a.plot_every is not None and (self.step % a.plot_every == 0 or last)
            # Flush before reporting: wandb drops rows logged below the current step.
            if self.step % a.log_every == 0 or save or plot or joint_eval or last:
                self.log(buffer)
                buffer = []
            if save:
                self.save()
            if plot:
                self.plot()
            if joint_eval:
                self.log_joint_eval(self.evaluate_joint())

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
