import importlib.util
import json

import pytest

import crystalatte as cle
from crystalatte.compute import ComputeSpec, InputFileBackend

qcmanybody = pytest.importorskip("qcmanybody")
qcel_models = pytest.importorskip("qcelemental.models")


def _spec(program="psi4", bsse="cp"):
    return ComputeSpec(program=program, method="hf", basis="sto-3g", bsse=bsse, use_defaults=False)


def _core(multimer, spec):
    return qcmanybody.ManyBodyCore(multimer.to_molecule(), **spec.manybody_kwargs(len(multimer.monomers)))


def _labels(multimer, spec):
    return {label for _, label, _ in _core(multimer, spec).iterate_molecules()}


@pytest.fixture(scope="module")
def nh3_dimers(ammonia):
    return cle.generate(ammonia, N=2, R=4.0, verbose=0)


@pytest.fixture(scope="module")
def nh3_trimers(ammonia):
    return cle.generate(ammonia, N=3, R=4.0, verbose=0)


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


def _document(path, multimer):
    return json.loads((path / _manifest(path)["multimers"][multimer.id]["file"]).read_text())


def test_writes_one_json_per_multimer_and_a_manifest(tmp_path, nh3_dimers, nh3_trimers):
    InputFileBackend(tmp_path).submit(nh3_dimers + nh3_trimers, _spec())
    assert sorted(p.stem for p in (tmp_path / "dimers").glob("*.json")) == sorted(m.id for m in nh3_dimers)
    assert sorted(p.stem for p in (tmp_path / "trimers").glob("*.json")) == sorted(m.id for m in nh3_trimers)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["dimers", "manifest.json", "trimers"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_str_output_dir_accepted(tmp_path, nh3_dimers):
    InputFileBackend(str(tmp_path)).submit(nh3_dimers[:1], _spec())
    assert (tmp_path / "dimers" / f"{nh3_dimers[0].id}.json").exists()


def test_document_has_one_task_per_subcalculation(tmp_path, nh3_trimers):
    spec = _spec()
    InputFileBackend(tmp_path).submit(nh3_trimers, spec)
    for m in nh3_trimers:
        doc = _document(tmp_path, m)
        assert {t["label"] for t in doc["tasks"]} == _labels(m, spec)
        assert len(doc["tasks"]) == len(_labels(m, spec))
        for t in doc["tasks"]:
            assert json.loads(t["label"])[1:] == [t["real"], t["basis"]]


def test_document_describes_the_multimer(tmp_path, nh3_dimers):
    spec = _spec(bsse="nocp")
    InputFileBackend(tmp_path).submit(nh3_dimers[:1], spec)
    m = nh3_dimers[0]
    doc = _document(tmp_path, m)
    assert doc["id"] == m.id
    assert (doc["N"], doc["multiplicity"], doc["weight"]) == (2, m.multiplicity, m.weight)
    assert doc["formulas"] == ["H3N", "H3N"]
    assert (doc["program"], doc["bsse"]) == ("psi4", "nocp")
    assert doc["result_key"] == "nocp_corrected_2_body_contribution_to_energy"


def test_tasks_are_valid_atomic_inputs(tmp_path, nh3_dimers):
    spec = ComputeSpec(program="psi4", method="mp2", basis="cc-pvdz", keywords={"scf_type": "pk"})
    InputFileBackend(tmp_path).submit(nh3_dimers[:1], spec)
    for t in _document(tmp_path, nh3_dimers[0])["tasks"]:
        inp = qcel_models.AtomicInput(**t["atomic_input"])
        assert t["atomic_input"]["schema_name"] == "qcschema_input"
        assert inp.driver == "energy"
        assert (inp.model.method, inp.model.basis) == ("mp2", "cc-pvdz")
        assert inp.keywords == spec.final_keywords


def test_counterpoise_tasks_carry_ghost_atoms(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path).submit(nh3_dimers[:1], _spec(bsse="cp"))
    tasks = _document(tmp_path, nh3_dimers[0])["tasks"]
    ghosted = [t for t in tasks if len(t["basis"]) > len(t["real"])]
    assert ghosted
    for t in ghosted:
        mol = t["atomic_input"]["molecule"]
        assert not all(mol["real"])
        assert len(mol["fragments"]) == len(t["basis"])


@pytest.mark.parametrize("program", ["psi4", "nwchem", "qchem", "rdkit"])
def test_any_program_is_written_without_it_installed(tmp_path, nh3_dimers, program):
    InputFileBackend(tmp_path).submit(nh3_dimers[:1], _spec(program))
    assert _document(tmp_path, nh3_dimers[0])["program"] == program


def test_inputs_match_local_backend(tmp_path, nh3_dimers):
    from crystalatte.compute.tasks import manybody_core, subcalculations

    spec = ComputeSpec(program="psi4", method="hf", basis="sto-3g")
    m = nh3_dimers[0]
    InputFileBackend(tmp_path).submit([m], spec)
    def canonical(atomic_input):
        return json.loads(qcel_models.AtomicInput(**atomic_input).json())

    written = {t["label"]: canonical(t["atomic_input"]) for t in _document(tmp_path, m)["tasks"]}
    local = {t.label: canonical(t.atomic_input) for t in subcalculations(manybody_core(m, spec), spec)}
    assert written == local


def _core_from_document(doc):
    return qcmanybody.ManyBodyCore(qcel_models.Molecule(**doc["molecule"]), **doc["manybody"])


def test_document_alone_rebuilds_the_manybody_analysis(tmp_path, nh3_trimers):
    spec = _spec()
    m = nh3_trimers[0]
    InputFileBackend(tmp_path).submit([m], spec)
    doc = _document(tmp_path, m)
    assert {label for _, label, _ in _core_from_document(doc).iterate_molecules()} == _labels(m, spec)
    fake = {t["label"]: {"energy": -1.0 - 0.1 * len(t["real"]) - 0.01 * len(t["basis"])} for t in doc["tasks"]}
    by_doc = _core_from_document(doc).analyze(fake)["results"][doc["result_key"]]
    assert by_doc == pytest.approx(_core(m, spec).analyze(fake)["results"][doc["result_key"]], abs=1e-12)


def test_running_the_written_inputs_reproduces_local_backend(tmp_path, nh3_dimers):
    qcengine = pytest.importorskip("qcengine")
    if importlib.util.find_spec("psi4") is None:
        pytest.skip("psi4 is not installed")
    spec = ComputeSpec(program="psi4", method="hf", basis="sto-3g")
    m = nh3_dimers[0]
    InputFileBackend(tmp_path / "inputs").submit([m], spec)
    doc = _document(tmp_path / "inputs", m)
    energies = {}
    for t in doc["tasks"]:
        out = qcengine.compute(t["atomic_input"], doc["program"], raise_error=True, task_config={"ncores": 1})
        energies[t["label"]] = {"energy": out.return_result}
    by_hand = _core_from_document(doc).analyze(energies)["results"][doc["result_key"]]

    local = cle.LocalBackend(tmp_path / "local", ncores=1)
    local.submit([m], spec)
    assert by_hand == pytest.approx(local.collect().rows[0].energy, abs=1e-10)


def test_manifest_records_run(tmp_path, nh3_dimers):
    spec = _spec()
    InputFileBackend(tmp_path).submit(nh3_dimers, spec)
    manifest = _manifest(tmp_path)
    assert manifest["program"] == "psi4"
    assert manifest["spec_fingerprint"] == spec.fingerprint()
    assert ComputeSpec.model_validate(manifest["spec"]) == spec
    assert manifest["crystal_fingerprint"] == nh3_dimers[0].generation.crystal_fingerprint
    assert manifest["crystalatte_version"]
    assert manifest["active"] == {"2": [m.id for m in nh3_dimers]}
    assert manifest["generation"]["2"] == nh3_dimers[0].generation.model_dump(mode="json")
    for m in nh3_dimers:
        entry = manifest["multimers"][m.id]
        assert (entry["N"], entry["multiplicity"], entry["weight"]) == (2, m.multiplicity, m.weight)
        assert entry["geometric_mean"] == m.geometric_mean
        assert entry["min_separation"] == m.min_separation
        assert entry["formulas"] == ["H3N", "H3N"]
        assert entry["file"] == f"dimers/{m.id}.json"
        assert "tasks" not in entry


def test_resubmitting_another_order_extends_the_manifest(tmp_path, nh3_dimers, nh3_trimers):
    InputFileBackend(tmp_path).submit(nh3_dimers, _spec())
    InputFileBackend(tmp_path).submit(nh3_trimers, _spec())
    manifest = _manifest(tmp_path)
    assert set(manifest["active"]) == {"2", "3"}
    assert set(manifest["multimers"]) == {m.id for m in nh3_dimers + nh3_trimers}


def test_resubmitting_other_spec_into_same_dir_raises(tmp_path, nh3_dimers):
    InputFileBackend(tmp_path).submit(nh3_dimers, _spec("psi4"))
    with pytest.raises(ValueError, match="spec"):
        InputFileBackend(tmp_path).submit(nh3_dimers, _spec("nwchem"))


def test_resubmitting_other_crystal_into_same_dir_raises(tmp_path, nh3_dimers, benzene):
    InputFileBackend(tmp_path).submit(nh3_dimers, _spec())
    with pytest.raises(ValueError, match="crystal fingerprint"):
        InputFileBackend(tmp_path).submit(cle.generate(benzene, N=2, R=4.0, verbose=0), _spec())


def test_backend_exported_from_package():
    assert cle.InputFileBackend is InputFileBackend
