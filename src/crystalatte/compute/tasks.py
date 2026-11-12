"""QCManyBody sub-calculations of a multimer, as QCSchema single-point inputs."""

import json
from typing import NamedTuple

from ..core.multimer import Multimer
from .spec import ComputeSpec


class Subcalculation(NamedTuple):
    """One single-point of a many-body expansion

    ``real`` and ``basis`` are 1-based fragment indices; fragments in ``basis`` but not in ``real``
    are ghosts. ``label`` is the QCManyBody key for :meth:`qcmanybody.ManyBodyCore.analyze`.
    """

    label: str
    real: list[int]
    basis: list[int]
    atomic_input: dict


def manybody_core(multimer: Multimer, spec: ComputeSpec):
    """QCManyBody bookkeeping for ``multimer`` under ``spec``"""
    from qcmanybody import ManyBodyCore

    return ManyBodyCore(multimer.to_molecule(), **spec.manybody_kwargs(len(multimer.monomers)))


def subcalculations(core, spec: ComputeSpec) -> list[Subcalculation]:
    """Every single-point ``core`` needs, each as an ``AtomicInput`` dict for ``spec.program``"""
    tasks = []
    for _, label, molecule in core.iterate_molecules():
        _, real, basis = json.loads(label)
        atomic_input = {**spec.atomic_template(), "molecule": json.loads(molecule.json())}
        tasks.append(Subcalculation(label, real, basis, atomic_input))
    return tasks


def result_key(spec: ComputeSpec, N: int) -> str:
    """Key of the multimer's N-body contribution in ``ManyBodyCore.analyze(...)["results"]``"""
    return f"{spec.bsse}_corrected_{N}_body_contribution_to_energy"
