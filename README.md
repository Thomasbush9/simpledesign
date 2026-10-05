# simpledesign

## Setup

```bash
uv sync
```

## Pretrained weights

The trunk's sequence branch is initialized from ESM2. Download a checkpoint into `checkpoints/`:

```bash
uv run hf download facebook/esm2_t6_8M_UR50D \
  config.json model.safetensors vocab.txt tokenizer_config.json special_tokens_map.json \
  --local-dir checkpoints/esm2_t6_8M_UR50D
```

Swap the repo id for larger models (`esm2_t12_35M_UR50D`, `esm2_t33_650M_UR50D`).

## Data

A dataset directory pairs structures and sequences by file stem:

```
<root>/
  structures/<id>.cif | <id>.pdb   single protein chain
  fasta/<id>.fasta                  one record (header ignored)
  summary.tsv                       optional, tab-separated, `id` + `length` columns
```

```python
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from simpledesign.data.dataset import ProteinCollator, ProteinDataset

ds = ProteinDataset("<root>")   # fails on unpaired files
ds.validate()                    # optional: check every structure/FASTA pair up front
tok = AutoTokenizer.from_pretrained("checkpoints/esm2_t6_8M_UR50D")
loader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=ProteinCollator(tok))
```

Each item's structure sequence must equal its FASTA sequence (and `length` in `summary.tsv`).
Batches carry `seq`, `coords`, `seq_mask`, `struct_mask`, `idx` for `SimpleDesign.forward`.
`ProteinDataset(..., cache=True)` parses each protein once; `ProteinCollator(tok, max_len=256)`
randomly crops longer proteins.

## Training

Every key in `configs/train_config.yaml` is a `TrainerArgs` field
(`src/simpledesign/training/train.py`).

```bash
uv run python scripts/train.py --config configs/train_config.yaml                  # cuda > mps > cpu
uv run torchrun --standalone --nproc_per_node=4 scripts/train.py --config <yaml>  # DDP, one node
```

- Each protein is repeated `n_replicas` times; every replica gets its own random rotation
  (+ `translation_std` shift), `t`, `t'`, sequence mask and noise. Rank `r` seeds with `seed + r`.
- `out_dir` receives `history.png` at the end (and `wandb/` if `use_wandb: true`; on nodes without
  internet set `WANDB_MODE=offline`, then `wandb sync`).
- `plot_every: N`: every N steps (and at the end) the first dataset protein is corrupted at each
  `viz_levels` value (t = t', same noise every time) and plotted: `velocity_<step>.png`
  (input coordinates, target vs predicted velocity arrows and the CA traces they lead to) and
  `sequence_<step>.png` (predicted amino-acid probabilities, true residues marked) in
  `out_dir/viz/`, and as `viz/velocity`, `viz/sequence` on wandb.
- `ckpt_dir` set: checkpoint every `ckpt_every` steps and at the end, overwriting `last.pt`
  (`ckpt_keep_all: true` keeps `step_XXXXXXX.pt`). Writes are atomic.
- `resume_from: <ckpt>` restores model, optimizer, step and history, then trains up to `num_steps`.

## Sampling

`src/simpledesign/models/sampling.py` holds the shared pieces: `masked_sequence` (start tokens),
`observed_fraction` (sequence time), `struct_schedule` (`linear` or the paper's `log` t' grid),
and one update per modality, `seq_step` (Gumbel + temperature proposal, top-K unmasking with
optional re-masking) and `struct_step` (Euler ODE or Langevin-corrected SDE, re-centering, random
rotation). The `SimpleDesign` samplers are loops of one forward pass + step(s):

- `sample_struct(seq, seq_mask, struct_mask, idx, ...)`: structure for a fixed sequence, t' from
  noise (or from a noised `x_init` at `t_start`) to 1. Returns Angstrom coordinates (+ trajectory
  with `traj_every`). Select the structure grid with `struct_schedule_type="linear"` or `"log"`.
- `sample_seq(seq_mask, struct_mask, idx, struct, t_struct, ...)`: sequence for a fixed structure
  (`struct` in nm), linear unmasking schedule.
- `joint_sample(lengths, n_steps, *, struct_schedule_type="linear", ...)`: unconditional
  co-design from fully masked sequence and structure noise. One forward pass updates both
  modalities; sequence time is the actual observed fraction, while the structure grid is
  selected independently. Returns `(seq, coords)` in Angstrom, or `(seq, coords, traj)` with
  `traj_every` (initial, periodic, and final coordinate frames, also in Angstrom).

```python
seq, coords, traj = model.joint_sample(
    [89, 128], n_steps=200, struct_schedule_type="log", traj_every=20
)
```

Joint sampling currently accepts lengths only, not initial or fixed sequence/structure inputs.
The folding CLI retains `--schedule`; it passes this to `sample_struct` as `struct_schedule_type`.

`joint_start(lengths, device=...)` in `models/sampling.py` builds unconditional inputs without
a dataset batch. For `[89, 128]`, it pads to `L = 130`: fully masked residue tokens with
cls/eos/pad, centered Gaussian structure noise, masks, indices, and `t = t_prime = 0`.
Lengths must be positive integers; use `[N] * K` for K samples of length N. Coordinates are
already in model units (nm), unlike clean `ProteinCollator` coordinates in Angstrom.

```python
from simpledesign.models.sampling import joint_start

state = joint_start([89, 128], device=model.device)
with torch.no_grad():
    logits, velocity = model(**state)
```

This initializes the joint state only; it does not run a joint sampling loop.
`model.device` reflects the current parameter device, including after `model.to(...)`.

```bash
uv run python scripts/sample.py --ckpt runs/<run>/ckpt/last.pt                # SDE, tau 0.5
uv run python scripts/sample.py --ckpt ... --ode --ids Q9UI30 --n_samples 16   # no injected noise
uv run python scripts/sample.py --ckpt ... --t_start 0.7 --traj_every 10       # refine + movie
uv run python scripts/sample.py --ckpt ... --schedule log                      # log-spaced t'
```

Folds dataset proteins and writes `<run>/samples/<settings>/`: CA-only PDBs (samples aligned onto
`true.pdb`, `traj.pdb` with one model per frame), `samples.png`, and `metrics.tsv` (RMSD, RMSD to
the mirrored truth, CA-CA distances).

## Layout

```
src/simpledesign/
  models/     # model architecture
  data/       # datasets, preprocessing, loaders
  training/   # training / evaluation loops
configs/      # experiment configs
scripts/      # entry points (train, eval, ...)
notebooks/    # exploration
tests/        # pytest suite
data/         # raw/processed data        (gitignored)
checkpoints/  # saved weights             (gitignored)
runs/         # logs / experiment outputs (gitignored)
```

## Commands

```bash
uv run jupyter lab       # notebooks (vim mode on; toggle in Settings menu)
uv run pytest            # tests
uv run ruff check .      # lint
uv run ruff format .     # format
```
