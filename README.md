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
The collator materializes batch indices so CUDA DataLoader pinned-memory transfers do not
encounter overlapping storage from expanded tensor views.
`ProteinDataset(..., cache=True)` parses each protein once; `ProteinCollator(tok, max_len=256)`
randomly crops longer proteins.

### AFDB subsets

```bash
uv run python scripts/sample_afdb.py --tar /path/to/swissprot.tar --n 100 --min-plddt 90 --seed 0 --out data/subset
uv run python scripts/sample_afdb.py --log data/subset/selection.json --out data/replayed
```

Samples uniformly without replacement from `.cif`, `.pdb`, or their `.gz` members whose
mean Cα pLDDT is at least the threshold (0–100; default 90). The initial pass scans the
entire archive; fewer than N eligible structures is an error. Repeated IDs, including
CIF/PDB copies, use the first representation encountered and count only once.
Writes paired structures/FASTA, `summary.tsv`, and a JSON selection log, then loads
and validates a `ProteinDataset`; `simpledesign.data.afdb.sample_afdb(...)` returns that
dataset. Existing output is reused only with a selection log and matching protein IDs,
structure contents, and valid structure/FASTA pairs; conflicting output is never overwritten.

`--log` defaults to `<out>/selection.json`: a new path saves the selection; an existing
log replays exactly those members, ignoring N, threshold, and seed. The log records the
archive path, size/mtime, member offsets, hashes, and scores. Replay checks the archive
and member hashes and does not resample or overwrite the log. Plain tar replay seeks
directly to selected members; outer-compressed tar still requires decompression.

Training uses the same sampler through `configs/train_config.yaml`:

```yaml
data_dir: data/afdb_100
afdb_tar: /path/to/swissprot.tar
afdb_log: null
afdb_n: 100
afdb_min_plddt: 90.0
afdb_seed: 0
```

Set either `afdb_tar` or `afdb_log` to enable preparation before model initialization.
`data_dir` is the export directory; `cache` also applies to sampled data. With an existing
`afdb_log`, the tar path can be null (read from the log), and sampling parameters are ignored.
The default log is `data_dir/selection.json`, so rerunning/resuming the same configuration
reuses the same subset. To sample a different subset, choose a new data directory and log.
With both AFDB paths null, training loads `data_dir` directly as before.

Under DDP, rank 0 prepares the subset and communicates completion/errors before other ranks
load it; the data directory and log must be on shared storage. For a large initial archive
scan, prepare the subset with the standalone CLI before launching DDP to avoid process-group
timeouts while other ranks wait.

The checked-in config samples N=100 at mean pLDDT ≥ 80 from the SwissProt PDB v6 tar.

## Training

Every key in `configs/train_config.yaml` is a `TrainerArgs` field
(`src/simpledesign/training/train.py`).

```bash
uv run python scripts/train.py --config configs/train_config.yaml                  # cuda > mps > cpu
uv run torchrun --standalone --nproc_per_node=4 scripts/train.py --config <yaml>  # DDP, one node
```

Single H100 on the Kempner partition (submit from the project root):

```bash
mkdir -p runs/slurm
sbatch scripts/train_single_gpu.sbatch configs/train_config.yaml
```

The launcher uses `.venv`, requests one GPU, 8 CPUs, 64 GiB RAM, and 24 hours, and forces
online W&B logging. Configure credentials with `.venv/bin/wandb login` if not already
authenticated, or provide `WANDB_API_KEY` through the environment (never in YAML).
It preserves the config's output/checkpoint paths and W&B project/name.
Slurm output is `runs/slurm/train-<jobid>.log`; GPU/data preparation precedes W&B run
initialization, so the dashboard run appears only after the subset and model are ready.

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

### Optional CA-frame FAPE

`beta_fape` (default `0.0`, disabled) adds a reflection-sensitive auxiliary term:
`total = lambda_seq * sequence_loss + lambda_struct * velocity_loss + beta_fape * FAPE`.
It compares the predicted clean coordinates `x_hat = x_t + (1 - t') * v_pred` with the
clean target. This is **CA-frame FAPE**, not full-backbone N/CA/C FAPE: each consecutive
CA triple defines a right-handed frame, and every valid CA is compared in every valid frame.
Distances are clamped at 1 nm and normalized by 1 nm, averaged per protein then per batch.
The term is invariant to independent proper rigid motions but penalizes nonplanar reflections.
Masked/degenerate target frames are excluded; proteins without valid frames contribute zero.
Degenerate predicted frames remain penalized. Geometry is computed in float32, outside autocast.

`fape` is logged separately to history, console, and W&B; positive FAPE histories receive an
additional plot panel. `configs/train_fape_config.yaml` configures a fresh 50,000-step run on
the same logged 100-protein subset, with `beta_fape: 1.0`, 4 proteins × 16 independently
augmented replicas = 64 views/step, and separate outputs under `runs/afdb100-fape1-r16-50k`.
The baseline used 4 replicas, so this comparison changes both FAPE and replica count; it does
not isolate FAPE's causal effect. Neither CA-only FAPE nor same-model diagnostics establish
full stereochemical validity.

```bash
sbatch scripts/train_single_gpu.sbatch configs/train_fape_config.yaml
```

### Internal joint-sampling diagnostics

`joint_eval_every: N` runs an internal joint-sampling probe every N steps and at the final step
(`null` disables it). It pauses training while sampling; no independent folding model is loaded.
`joint_eval_length`, `joint_eval_n_samples`, and `joint_eval_n_steps` set the fixed probe panel.
`joint_eval_seed` fixes probe randomness; CPU/device RNG states and model mode are restored.

`Trainer.evaluate_joint()` generates sequence/structure pairs, inverse-folds each generated
structure, and folds each generated sequence from fresh noise. All three samplers use
`joint_eval_n_steps`; joint generation and folding use the log structure schedule. It returns
`seq`, `struct`, `inverted_folding_seq`, `folded_structure`, `seq_mask`, `struct_mask`, `idx`, and
four per-sample metric tensors, computed on the model device:

- `seq_agreement`: residue-token agreement between the joint and inverse-folded sequences.
- `struct_mse`: mean squared xyz error after the existing Kabsch alignment, in Angstrom².
- `struct_rmsd`: Cα RMSD after the same alignment, in Angstrom; `RMSD² = 3 × MSE`.
- `struct_rmsd_mirror`: Cα RMSD after reflecting the refolded structure across one axis and then
  applying Kabsch alignment to the joint structure, in Angstrom. Saved coordinates are unchanged.

Only real residues contribute. Alignment allows proper rotations/translations, not reflections.
Both returned structures are in Angstrom; only the inverse-folding input is converted to model
units. These are same-model consistency diagnostics, not independent validation or native accuracy.
A low reflected RMSD with high ordinary RMSD indicates relative handedness disagreement, not
which structure is correct. Both RMSDs appear together in the comparison plot; earlier saved
evaluations without the reflected metric have no reflected data points.

`Trainer.log_joint_eval()` transfers results to CPU for export/plotting. Training calls it after
flushing buffered logs. Every evaluation is retained under `out_dir/joint_eval/step_XXXXXXX/`:

```text
args.json                       # step, training/probe settings, units, metric definitions
metrics.tsv                     # one row per sample
samples.pt                      # full-precision CPU tensors, including masks and metrics
metrics.png                     # metric history: samples, means, and min/max bands
sample_0000/
  sequence.fasta                # joint-generated sequence
  structure.pdb                 # joint-generated Cα structure, original coordinates
  inverse_folded.fasta          # sequence generated from that structure
  folded_structure.pdb          # structure generated from sequence.fasta, original coordinates
```

Sample directories repeat for each generated pair. PDB coordinates have standard text precision;
`samples.pt` preserves full precision. An evaluation directory is published atomically and never
overwritten; use a fresh `out_dir` for a new experiment. Curves include earlier saved evaluations
in the same run, including after resume. With W&B enabled, per-sample means are logged as
`joint_eval/seq_agreement`, `joint_eval/struct_mse`, `joint_eval/struct_rmsd`, and
`joint_eval/struct_rmsd_mirror`, plus the figure as `joint_eval/metrics`.
Saved pairs support later independent checking without rerunning sampling;
they do not include an additional model checkpoint or a reconstructed full-atom backbone.

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
