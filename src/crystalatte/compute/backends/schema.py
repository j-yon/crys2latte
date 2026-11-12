import json
import os
from pathlib import Path

from ...core.multimer import Multimer, order_name
from ..results import Results, Status
from ..spec import ComputeSpec
from ..tasks import manybody_core, result_key, subcalculations
from .base import Backend, crystalatte_version, manifest_entry

_INSTALL_HINT = "SchemaBackend needs qcmanybody; install it with: pip install 'crystalatte[local]'"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


class SchemaBackend(Backend):
    """Write QCSchema inputs for the user to run, one JSON file per multimer plus a ``manifest.json``

    Each multimer becomes ``<order>s/<multimer id>.json`` holding its QCManyBody sub-calculations,
    each a QCSchema ``AtomicInput`` with its label and real and basis (ghost) fragments. Run each
    ``atomic_input`` with QCEngine (``qcengine.compute(atomic_input, program)``), then pass
    ``{label: {"energy": e}}`` to ``ManyBodyCore(Molecule(**doc["molecule"]), **doc["manybody"])
    .analyze`` and read the file's ``result_key``. The manifest records the spec, the generation settings, the active multimers
    and, per multimer, its order, multiplicity, lattice weight and file. Nothing is read back.

    Memory, cores and scratch are not part of an ``AtomicInput``; set them when running, e.g. with
    QCEngine's ``task_config``.

    :param output_dir: directory for the input files and the manifest
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def submit(self, multimers: list[Multimer], spec: ComputeSpec) -> None:
        by_order = self._validate(multimers)
        manifest = self._start_manifest(multimers, spec)
        documents = {m.id: self._document(m, spec) for group in by_order.values() for m in group}

        for N, group in by_order.items():
            order_dir = self.output_dir / f"{order_name(N)}s"
            order_dir.mkdir(parents=True, exist_ok=True)
            manifest["generation"][str(N)] = group[0].generation.model_dump(mode="json")
            manifest["active"][str(N)] = [m.id for m in group]
            for m in group:
                path = order_dir / f"{m.id}.json"
                _write_json(path, documents[m.id])
                manifest["multimers"][m.id] = {**manifest_entry(m), "file": path.relative_to(self.output_dir).as_posix()}
        manifest["crystalatte_version"] = crystalatte_version()
        _write_json(self.output_dir / "manifest.json", manifest)

    @staticmethod
    def _document(multimer: Multimer, spec: ComputeSpec) -> dict:
        """Everything needed to run one multimer and assemble its energy"""
        try:
            from qcelemental.models import AtomicInput

            core = manybody_core(multimer, spec)
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        N = len(multimer.monomers)
        return {
            "id": multimer.id,
            **manifest_entry(multimer),
            "program": spec.program,
            "bsse": spec.bsse,
            "molecule": json.loads(multimer.to_molecule().json()),
            "manybody": spec.manybody_kwargs(N),
            "result_key": result_key(spec, N),
            "tasks": [
                {
                    "label": task.label,
                    "real": task.real,
                    "basis": task.basis,
                    "atomic_input": json.loads(AtomicInput(**task.atomic_input).json()),
                }
                for task in subcalculations(core, spec)
            ],
        }

    def _start_manifest(self, multimers: list[Multimer], spec: ComputeSpec) -> dict:
        crystal_fp = multimers[0].generation.crystal_fingerprint
        path = self.output_dir / "manifest.json"
        if not path.exists():
            return {
                "program": spec.program,
                "spec": spec.model_dump(mode="json"),
                "spec_fingerprint": spec.fingerprint(),
                "crystal_fingerprint": crystal_fp,
                "generation": {},
                "active": {},
                "multimers": {},
            }
        manifest = json.loads(path.read_text())
        if manifest["spec_fingerprint"] != spec.fingerprint():
            raise ValueError(
                f"spec differs from the one used in {self.output_dir} (spec fingerprint "
                f"{manifest['spec_fingerprint'][:12]} vs {spec.fingerprint()[:12]}); use a new output_dir"
            )
        if manifest["crystal_fingerprint"] != crystal_fp:
            raise ValueError(
                f"crystal fingerprint differs from the one used in {self.output_dir} "
                f"({manifest['crystal_fingerprint'][:12]} vs {crystal_fp[:12]}); use a new output_dir"
            )
        return manifest

    def status(self) -> Status:
        raise NotImplementedError(_MSG)

    def collect(self, *, partial: bool = False) -> Results:
        raise NotImplementedError(_MSG)


_MSG = "SchemaBackend only writes input files; run them yourself or use LocalBackend/FractalBackend"
