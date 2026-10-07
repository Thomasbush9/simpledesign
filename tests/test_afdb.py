import gzip
import tarfile

import biotite.structure as struc
import biotite.structure.io as strucio
import pytest

from simpledesign.data.afdb import sample_afdb


@pytest.fixture(params=[".cif", ".pdb", ".pdb.gz"])
def archive(tmp_path, request):
    path = tmp_path / "proteins.tar"
    with tarfile.open(path, "w") as tar:
        for pid, score in [("low", 89), ("boundary", 90), ("high", 95)]:
            atoms = struc.AtomArray(1)
            atoms.chain_id[:] = "A"
            atoms.res_id[:] = 1
            atoms.res_name[:] = "ALA"
            atoms.atom_name[:] = "CA"
            atoms.element[:] = "C"
            atoms.coord[:] = [[1, 2, 3]]
            atoms.set_annotation("b_factor", [float(score)])
            structure = tmp_path / f"{pid}{request.param.removesuffix('.gz')}"
            strucio.save_structure(structure, atoms)
            if request.param.endswith(".gz"):
                compressed = tmp_path / f"{pid}{request.param}"
                compressed.write_bytes(gzip.compress(structure.read_bytes()))
                structure = compressed
            tar.add(structure, arcname=structure.name)
    return path


def test_threshold_replay_and_existing_export(archive, tmp_path):
    out = tmp_path / "sampled"
    sampled = sample_afdb(archive, out, n=2, min_plddt=90)
    assert sampled.ids == ["boundary", "high"]
    log = out / "selection.json"
    before = log.read_bytes()
    replay = sample_afdb(None, tmp_path / "replayed", n=1, min_plddt=100, log=log)
    reused = sample_afdb(archive, out, n=3, min_plddt=0, cache=False)
    assert replay.ids == reused.ids == sampled.ids
    for db in (sampled, replay, reused):
        assert [db[i]["sequence"] for i in range(2)] == ["A", "A"]
        assert [db[i]["b_factor"].item() for i in range(2)] == [90, 95]
    assert log.read_bytes() == before


def test_reuse_rejects_changed_export(archive, tmp_path):
    out = tmp_path / "sampled"
    db = sample_afdb(archive, out, n=2)
    structure = db.structure_paths[db.ids[0]]
    structure.write_bytes(structure.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="structure differs"):
        sample_afdb(archive, out, n=2)


def test_existing_directory_without_log_is_not_overwritten(archive, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    with pytest.raises(ValueError, match="requires a selection log"):
        sample_afdb(archive, out, n=2)
