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
