"""Paired structure/sequence dataset and the collator that builds SimpleDesign inputs.

Expected layout of `root`:

    root/
      structures/<id>.cif | <id>.pdb
      fasta/<id>.fasta
      summary.tsv            optional; tab-separated with an `id` column (and `length`)

Structures and sequences are paired by file stem (<id>).
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from simpledesign.data.parsing import read_fasta, read_structure

STRUCTURE_SUFFIXES = (".cif", ".pdb")


class ProteinDataset(Dataset):
    """One item per protein: its FASTA sequence and the CA coordinates of its structure.

    Pairing is checked at construction (every structure has a FASTA and vice versa, no id with
    two structure files). Every item is validated when loaded: the sequence read from the
    structure must be identical to the FASTA sequence, and to `length` in summary.tsv if present.
    `validate()` checks all items up front and reports every failure at once.
    cache=True keeps every parsed item in memory (per DataLoader worker: use persistent workers).
    """

    def __init__(self, root: str | Path, center: bool = True, cache: bool = False):
        self.root = Path(root)
        self.center = center
        self._cache: dict[int, dict] | None = {} if cache else None
        structures: dict[str, Path] = {}
        for path in sorted((self.root / "structures").iterdir()):
            if path.suffix not in STRUCTURE_SUFFIXES:
                continue
            if path.stem in structures:
                raise ValueError(
                    f"two structure files for {path.stem!r}: {structures[path.stem]}, {path}"
                )
            structures[path.stem] = path
        fastas = {p.stem: p for p in sorted((self.root / "fasta").glob("*.fasta"))}

        only_structure = sorted(structures.keys() - fastas.keys())
        only_fasta = sorted(fastas.keys() - structures.keys())
        if only_structure or only_fasta:
            raise ValueError(
                f"unpaired files in {self.root}: structures without FASTA {only_structure}, "
                f"FASTA without structure {only_fasta}"
            )
        self.ids = sorted(structures)
        self.structure_paths = structures
        self.fasta_paths = fastas

        summary = self.root / "summary.tsv"
        self.summary: dict[str, dict[str, str]] = {}
        if summary.exists():
            with open(summary, newline="") as f:
                self.summary = {row["id"]: row for row in csv.DictReader(f, delimiter="\t")}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict:
        """-> {"id", "sequence" (str, L), "coords" (L, 3) float32 in Angstrom (centered on the CA
        centroid if center=True), "b_factor" (L,) float32 (pLDDT for predicted structures)}"""
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        pid = self.ids[i]
        structure = read_structure(self.structure_paths[pid])
        sequence = read_fasta(self.fasta_paths[pid])
        if structure.sequence != sequence:
            raise ValueError(f"{pid}: {_describe_mismatch(sequence, structure.sequence)}")
        if pid in self.summary and "length" in self.summary[pid]:
            expected = int(self.summary[pid]["length"])
            if expected != len(sequence):
                raise ValueError(
                    f"{pid}: summary.tsv length {expected} != sequence length {len(sequence)}"
                )

        coords = torch.from_numpy(structure.ca_coords)
        if self.center:
            coords = coords - coords.mean(dim=0, keepdim=True)
        item = {
            "id": pid,
            "sequence": sequence,
            "coords": coords,
            "b_factor": torch.from_numpy(structure.b_factor),
        }
        if self._cache is not None:
            self._cache[i] = item
        return item

    def validate(self) -> None:
        """Load every item; raise one ValueError listing all failures."""
        errors = []
        for i in range(len(self)):
            try:
                self[i]
            except ValueError as e:
                errors.append(str(e))
        if errors:
            raise ValueError(
                f"{len(errors)}/{len(self)} proteins failed validation:\n" + "\n".join(errors)
            )


def _describe_mismatch(fasta_seq: str, structure_seq: str) -> str:
    if len(fasta_seq) != len(structure_seq):
        return f"FASTA length {len(fasta_seq)} != structure length {len(structure_seq)}"
    k = next(i for i, (a, b) in enumerate(zip(fasta_seq, structure_seq, strict=True)) if a != b)
    return (
        f"sequence mismatch at residue {k + 1}: "
        f"FASTA {fasta_seq[k]!r}, structure {structure_seq[k]!r}"
    )


class ProteinCollator:
    """Batch ProteinDataset items into SimpleDesign inputs (clean, uncorrupted).

    tokenizer: the ESM2 tokenizer, e.g.
    AutoTokenizer.from_pretrained("checkpoints/esm2_t6_8M_UR50D").
    max_len: proteins longer than this are cropped to a random contiguous window of max_len
    residues, re-centered on its CA centroid. None: no cropping.
    Output (L = longest (cropped) protein + 2 for <cls>/<eos>):
        ids          list[str]
        seq          (B, L) long   ESM2 token ids, <pad> = 1
        seq_mask     (B, L) bool   real tokens incl. <cls>/<eos>
        struct_mask  (B, L) bool   real residues only
        coords       (B, L, 3)     CA coordinates at residue slots, 0 at <cls>/<eos>/pad
        b_factor     (B, L)        per-residue b-factor at residue slots, 0 elsewhere
        idx          (B, L) long   0..L-1, shared by both modalities
    """

    def __init__(self, tokenizer, max_len: int | None = None):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def crop(self, item: dict) -> dict:
        n = len(item["sequence"])
        if self.max_len is None or n <= self.max_len:
            return item
        s = int(torch.randint(0, n - self.max_len + 1, ()))
        e = s + self.max_len
        coords = item["coords"][s:e]
        return {
            "id": item["id"],
            "sequence": item["sequence"][s:e],
            "coords": coords - coords.mean(dim=0, keepdim=True),
            "b_factor": item["b_factor"][s:e],
        }

    def __call__(self, items: list[dict]) -> dict:
        items = [self.crop(item) for item in items]
        enc = self.tokenizer(
            [item["sequence"] for item in items], padding=True, return_tensors="pt"
        )
        seq, seq_mask = enc["input_ids"], enc["attention_mask"].bool()
        B, L = seq.shape
        coords = torch.zeros(B, L, 3)
        b_factor = torch.zeros(B, L)
        struct_mask = torch.zeros(B, L, dtype=torch.bool)
        for b, item in enumerate(items):
            n = len(item["sequence"])
            # one token per residue + <cls>/<eos>: residues sit at positions 1..n
            assert seq_mask[b].sum() == n + 2, (
                f"{item['id']}: tokenizer did not give {n} + 2 tokens"
            )
            coords[b, 1 : n + 1] = item["coords"]
            b_factor[b, 1 : n + 1] = item["b_factor"]
            struct_mask[b, 1 : n + 1] = True
        return {
            "ids": [item["id"] for item in items],
            "seq": seq,
            "seq_mask": seq_mask,
            "struct_mask": struct_mask,
            "coords": coords,
            "b_factor": b_factor,
            "idx": torch.arange(L).expand(B, L),
        }
