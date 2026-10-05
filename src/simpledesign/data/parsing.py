"""Reading structures (.cif / .pdb via biotite) and sequences (FASTA via biotite)."""

from dataclasses import dataclass
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.io import fasta


@dataclass(frozen=True)
class Structure:
    """One protein chain, one entry per residue (in file order)."""

    sequence: str  # one-letter codes; non-standard residues -> "X"
    ca_coords: np.ndarray  # (L, 3) float32, Angstrom
    b_factor: np.ndarray  # (L,) float32; per-residue pLDDT (0-100) for ESMFold/AlphaFold files
    res_id: np.ndarray  # (L,) int, residue numbering from the file


def _three_to_one(res_name: str) -> str:
    try:
        return ProteinSequence.convert_letter_3to1(res_name)
    except KeyError:
        return "X"


def read_structure(path: str | Path, chain: str | None = None) -> Structure:
    """Parse the first model of a .cif/.pdb file into CA coordinates + sequence.

    chain=None requires the file to contain exactly one protein chain.
    Raises ValueError for multiple chains (without `chain`), missing chains, or residues
    without a CA atom (the sequence would silently lose residues otherwise).
    """
    atoms = strucio.load_structure(path, model=1, extra_fields=["b_factor"])
    atoms = atoms[struc.filter_amino_acids(atoms)]
    chains = sorted(set(atoms.chain_id.tolist()))
    if chain is None:
        if len(chains) != 1:
            raise ValueError(f"{path}: expected one protein chain, found {chains}; pass chain=")
        chain = chains[0]
    elif chain not in chains:
        raise ValueError(f"{path}: chain {chain!r} not found, available {chains}")
    atoms = atoms[atoms.chain_id == chain]

    ca = atoms[atoms.atom_name == "CA"]
    n_residues = struc.get_residue_count(atoms)
    if len(ca) != n_residues:
        raise ValueError(f"{path}: {n_residues} residues but {len(ca)} CA atoms")
    return Structure(
        sequence="".join(_three_to_one(r) for r in ca.res_name),
        ca_coords=ca.coord.astype(np.float32),
        b_factor=ca.b_factor.astype(np.float32),
        res_id=ca.res_id.astype(np.int64),
    )


def read_fasta(path: str | Path) -> str:
    """Sequence of a single-record FASTA file (header ignored)."""
    records = list(fasta.FastaFile.read(path).values())
    if len(records) != 1:
        raise ValueError(f"{path}: expected one FASTA record, found {len(records)}")
    return records[0].upper()


def write_ca_structure(path: str | Path, sequence: str, coords: np.ndarray) -> None:
    """Write a CA-only chain A to .pdb/.cif. coords: (L, 3) Angstrom, or (M, L, 3) for M models
    (e.g. a sampling trajectory, playable as states in PyMOL)."""
    coords = np.asarray(coords, dtype=np.float32)
    stack = coords[None] if coords.ndim == 2 else coords
    n = len(sequence)
    assert stack.shape[1:] == (n, 3), f"coords {coords.shape} do not match {n} residues"
    atoms = struc.AtomArrayStack(len(stack), n)
    atoms.coord = stack
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = np.arange(1, n + 1)
    atoms.res_name[:] = [ProteinSequence.convert_letter_1to3(aa) for aa in sequence]
    atoms.atom_name[:] = "CA"
    atoms.element[:] = "C"
    strucio.save_structure(path, atoms if len(stack) > 1 else atoms[0])
