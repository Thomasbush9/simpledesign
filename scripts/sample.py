"""Fold dataset proteins with a trained checkpoint and score the generated structures.

    uv run python scripts/sample.py --ckpt runs/mps-20k/ckpt/last.pt
    uv run python scripts/sample.py --ckpt ... --ids Q9UI30 --n_samples 16 --tau 1.0
    uv run python scripts/sample.py --ckpt ... --ode             # ODE: no noise after x_0
    uv run python scripts/sample.py --ckpt ... --t_start 0.5     # refine from the noised truth
    uv run python scripts/sample.py --ckpt ... --schedule log    # paper's log-spaced t' grid

Writes to out_dir (default <run>/samples/<settings>):
    <id>/true.pdb, <id>/sample_<k>.pdb   CA-only, samples Kabsch-aligned onto the truth
    <id>/traj.pdb                        with --traj_every: sample 0 over time (one model/frame)
    <id>/samples.png                     CA traces + CA-CA distance histogram
    metrics.tsv                          one row per sample
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from rich import print
from rich.table import Table
from transformers import AutoTokenizer

from simpledesign.data.dataset import ProteinCollator, ProteinDataset
from simpledesign.data.parsing import write_ca_structure
from simpledesign.models.simpledesign import SimpleDesign
from simpledesign.models.utils import ca_rmsd, structure_rigid_alignment
from simpledesign.viz.visualization import plot_samples


def load_model(ckpt_path: str | Path, device: torch.device):
    """Trainer checkpoint -> (model in eval mode, tokenizer, TrainerArgs dict, step)."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = SimpleDesign.from_esm2(args["esm_ck"], sigma=args["sigma"])
    model.load_state_dict(ckpt["model"])
    tokenizer = AutoTokenizer.from_pretrained(args["esm_ck"])
    return model.to(device).eval(), tokenizer, args, ckpt["step"]


def pairwise_rmsd(x: torch.Tensor, struct_mask: torch.Tensor) -> float:
    """Mean Kabsch RMSD over all sample pairs (diversity); nan for a single sample."""
    i, j = torch.triu_indices(len(x), len(x), 1)
    return ca_rmsd(x[i], x[j], struct_mask[i]).mean().item() if len(i) else float("nan")


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="SimpleDesign structure sampling")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default=None, help="default: the checkpoint's data_dir")
    parser.add_argument("--ids", nargs="*", default=None, help="default: first --n_proteins")
    parser.add_argument("--n_proteins", type=int, default=4)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--schedule", choices=("linear", "log"), default="linear")
    parser.add_argument("--ode", action="store_true", help="Euler ODE instead of the SDE")
    parser.add_argument("--tau", type=float, default=0.5, help="SDE noise scale")
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--no_rotate", action="store_true", help="no random rotation per step")
    parser.add_argument("--t_start", type=float, default=0.0, help=">0: start from noised truth")
    parser.add_argument("--traj_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default=None)
    cli = parser.parse_args()

    torch.manual_seed(cli.seed)
    device = torch.device(cli.device) if cli.device else default_device()
    model, tokenizer, args, step = load_model(cli.ckpt, device)
    ds = ProteinDataset(cli.data_dir or args["data_dir"])
    ids = cli.ids or ds.ids[: cli.n_proteins]
    collate = ProteinCollator(tokenizer, max_len=None)

    method = "ode" if cli.ode else f"sde_tau{cli.tau}"
    settings = f"{method}_{cli.schedule}_steps{cli.n_steps}"
    settings += f"_t0{cli.t_start}" if cli.t_start > 0 else ""
    out = Path(cli.out_dir or Path(cli.ckpt).parent.parent / "samples" / settings)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(cli), "ckpt_step": step}, indent=2))

    table = Table(title=f"{cli.ckpt} (step {step}) | {settings} | {cli.n_samples} samples")
    cols = ("id", "N", "RMSD mean", "RMSD min", "mirrored", "pairwise RMSD", "CA-CA (A)")
    for col in cols:
        table.add_column(col)
    rows = []
    for pid in ids:
        item = ds[ds.ids.index(pid)]
        n = len(item["sequence"])
        batch = {
            k: v.repeat_interleave(cli.n_samples, dim=0).to(device)
            for k, v in collate([item]).items()
            if isinstance(v, torch.Tensor)
        }
        sm, true = batch["struct_mask"], batch["coords"]
        res = model.sample_struct(
            batch["seq"],
            batch["seq_mask"],
            sm,
            batch["idx"],
            n_steps=cli.n_steps,
            struct_schedule_type=cli.schedule,
            sde=not cli.ode,
            tau=cli.tau,
            eta=cli.eta,
            rotate=not cli.no_rotate,
            x_init=true if cli.t_start > 0 else None,
            t_start=cli.t_start,
            traj_every=cli.traj_every,
        )
        x, traj = res if cli.traj_every else (res, None)

        rmsd = ca_rmsd(x, true, sm)
        # Kabsch cannot undo a reflection: a mirror-image fold scores a large RMSD to the truth
        rmsd_mirror = ca_rmsd(x, true * true.new_tensor([1.0, 1.0, -1.0]), sm)
        mirrored = rmsd_mirror < rmsd
        aligned = structure_rigid_alignment(x.float(), true.float(), sm)
        caca = (aligned[:, 2 : n + 1] - aligned[:, 1:n]).norm(dim=-1)  # (K, N-1)
        diversity = pairwise_rmsd(x, sm)

        d = out / pid
        d.mkdir(exist_ok=True)
        res_slots = slice(1, n + 1)  # residues sit after <cls>
        write_ca_structure(d / "true.pdb", item["sequence"], true[0, res_slots].cpu().numpy())
        for k in range(cli.n_samples):
            path = d / f"sample_{k}.pdb"
            write_ca_structure(path, item["sequence"], aligned[k, res_slots].cpu().numpy())
        if traj is not None:
            frames = torch.stack([s[0] for s in traj])  # sample 0 over time
            frames = structure_rigid_alignment(
                frames.float(), true[:1].float().expand_as(frames), sm[:1].expand(len(frames), -1)
            )
            write_ca_structure(d / "traj.pdb", item["sequence"], frames[:, res_slots].cpu().numpy())
        fig = plot_samples(
            aligned[:, res_slots], true[0, res_slots], rmsd, title=f"{pid} | {settings}"
        )
        fig.savefig(d / "samples.png", dpi=130)
        plt.close(fig)

        for k in range(cli.n_samples):
            rows.append(
                {
                    "id": pid,
                    "sample": k,
                    "length": n,
                    "rmsd": round(rmsd[k].item(), 3),
                    "rmsd_mirror": round(rmsd_mirror[k].item(), 3),
                    "caca_mean": round(caca[k].mean().item(), 3),
                    "caca_std": round(caca[k].std().item(), 3),
                }
            )
        table.add_row(
            pid,
            str(n),
            f"{rmsd.mean():.2f}",
            f"{rmsd.min():.2f}",
            f"{int(mirrored.sum())}/{cli.n_samples}",
            f"{diversity:.2f}",
            f"{caca.mean():.2f} +- {caca.std():.2f}",
        )

    with open(out / "metrics.tsv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(table)
    print(f"wrote {out}")
