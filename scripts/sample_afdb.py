"""Sample high-confidence AFDB structures, log the selection, and load a ProteinDataset."""

import argparse
from pathlib import Path

from simpledesign.data.afdb import sample_afdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int)
    parser.add_argument("--min-plddt", type=float, default=90.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, help="Existing log: replay; new path: save selection")
    args = parser.parse_args()
    db = sample_afdb(args.tar, args.out, args.n, args.min_plddt, args.seed, args.log)
    print(f"Loaded {len(db)} proteins into {args.out}")
