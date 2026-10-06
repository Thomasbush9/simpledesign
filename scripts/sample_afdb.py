"""Sample high-confidence AFDB mmCIFs, log the selection, and load a ProteinDataset."""

import argparse
import gzip
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

import numpy as np
from biotite.structure.io.pdbx import CIFFile

from simpledesign.data.dataset import ProteinDataset
from simpledesign.data.parsing import read_structure


def decode(data, name):
    return gzip.decompress(data) if name.endswith(".gz") else data


def sample_afdb(tar_path, out, n=None, min_plddt=90.0, seed=0, log=None):
    """Return a validated dataset; an existing log replays its exact selection."""
    out = Path(out)
    log = Path(log) if log else out / "selection.json"
    replay = log.exists()
    if out.exists():
        raise ValueError(f"Use a new output directory: {out}")
    if replay:
        manifest = json.loads(log.read_text())
        tar_path = tar_path or manifest["tar"]
    elif tar_path is None or n is None or n < 1 or not 0 <= min_plddt <= 100:
        raise ValueError("Sampling requires --tar, --n >= 1 and --min-plddt in [0, 100]")
    tar_path = Path(tar_path).resolve()
    stat = tar_path.stat()
    identity = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    selected = []
    with tarfile.open(tar_path, "r:*") as archive:
        if replay:
            if identity != manifest["archive"]:
                raise ValueError("Archive has changed since the log was written")
            for row in sorted(manifest["proteins"], key=lambda row: row["offset"]):
                archive.fileobj.seek(row["offset"])
                raw = archive.fileobj.read(row["size"])
                if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                    raise ValueError(f"Logged member changed: {row['member']}")
                selected.append((row, decode(raw, row["member"])))
        else:
            rng = random.Random(seed)
            eligible = 0
            for member in archive:
                # AFDB also includes PDB copies and metadata: sample mmCIFs only.
                if not member.isfile() or not member.name.endswith((".cif", ".cif.gz")):
                    continue
                with archive.extractfile(member) as handle:
                    raw = handle.read()
                data = decode(raw, member.name)
                atoms = CIFFile.read(io.StringIO(data.decode())).block["atom_site"]
                ca = atoms["label_atom_id"].as_array(str) == "CA"
                scores = atoms["B_iso_or_equiv"].as_array(float)[ca]
                if not len(scores) or not np.isfinite(scores).all():
                    continue
                mean = float(scores.mean())
                if mean < min_plddt:
                    continue
                eligible += 1
                slot = len(selected) if len(selected) < n else rng.randrange(eligible)
                if slot >= n:
                    continue
                name = Path(member.name).name.removesuffix(".gz")
                row = {
                    "id": Path(name).stem, "member": member.name,
                    "offset": member.offset_data, "size": member.size,
                    "sha256": hashlib.sha256(raw).hexdigest(), "mean_plddt": mean,
                }
                if slot == len(selected):
                    selected.append((row, data))
                else:
                    selected[slot] = (row, data)
            if eligible < n:
                raise ValueError(f"Requested {n} proteins, but only {eligible} meet the threshold")
            manifest = {
                "tar": str(tar_path), "archive": identity,
                "min_plddt": min_plddt, "seed": seed, "eligible": eligible,
            }
    selected.sort(key=lambda item: item[0]["id"])
    ids = [row["id"] for row, _ in selected]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Selection is empty or contains duplicate protein IDs")
    (out / "structures").mkdir(parents=True)
    (out / "fasta").mkdir()
    summary = ["id\tlength\tmean_plddt\n"]
    for row, data in selected:
        pid = row["id"]
        if Path(pid).name != pid or pid in (".", ".."):
            raise ValueError(f"Invalid protein ID: {pid!r}")
        path = out / "structures" / f"{pid}.cif"
        path.write_bytes(data)
        sequence = read_structure(path).sequence
        (out / "fasta" / f"{pid}.fasta").write_text(f">{pid}\n{sequence}\n")
        summary.append(f"{pid}\t{len(sequence)}\t{row['mean_plddt']:.6f}\n")
    (out / "summary.tsv").write_text("".join(summary))
    db = ProteinDataset(out, cache=True)
    db.validate()
    if not replay:
        manifest["proteins"] = [row for row, _ in selected]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
    return db


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
