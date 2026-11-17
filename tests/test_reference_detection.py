"""Reference detection on hand-built crystals, and generation with no partners in range.

Each crystal is small enough that its correct references can be read off by hand.
"""

import numpy as np
import pytest

from crystalatte.core.crystal import Crystal
from crystalatte.core.multimer import ASU
from crystalatte.core.space_group import SpaceGroup
from crystalatte.core.sym_ops import SymOp
from crystalatte.work.generate import generate

from conftest import load

ORTHO = (90.0, 90.0, 90.0)
SCREW_B = SymOp.from_components(
    np.diag([-1, 1, -1]).astype(np.int16), np.array([0, 6, 0], dtype=np.int16)
)
INVERSION = SymOp.from_components(np.diag([-1, -1, -1]).astype(np.int16), np.zeros(3, dtype=np.int16))


def _crystal(vectors, symbols, positions, ops=()) -> Crystal:
    group = SpaceGroup.from_sym_ops([SymOp.identity(), *ops], number=1)
    return Crystal(vectors, ORTHO, group, ASU(symbols, np.array(positions, dtype=float)))


def test_chain_along_a_lattice_vector_is_rejected():
    # 1.5 A C-C bonds repeat forever along a, so no finite molecule exists
    crystal = _crystal((1.5, 10.0, 10.0), ["C"], [[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="not a finite molecule"):
        crystal.get_references()


def test_chain_through_a_screw_axis_is_rejected():
    # neighbours are 1.48 A apart but only related by the 2_1 screw, never by translation alone
    crystal = _crystal((10.0, 2.5, 10.0), ["C"], [[0.04, 0.0, 0.0]], [SCREW_B])
    with pytest.raises(ValueError, match="not a finite molecule"):
        crystal.get_references()


def test_molecule_split_across_the_cell_boundary_is_finite():
    # C at x = 0.1 and O at x = 0.98 bond through the boundary, 1.2 A apart
    crystal = _crystal((10.0, 10.0, 10.0), ["C", "O"], [[0.1, 0.5, 0.5], [0.98, 0.5, 0.5]])
    assert [sorted(r.symbols) for r in crystal.get_references()] == [["C", "O"]]


def test_edge_fragment_is_dropped_when_rounding_would_split_its_site():
    # 0.0015 rounds differently from its image at -0.9985, where the C is cut off from its O
    crystal = _crystal(
        (10.0, 10.0, 10.0), ["C", "O"], [[0.0015, 0.5, 0.5], [0.8815, 0.5, 0.5]]
    )
    assert [sorted(r.symbols) for r in crystal.get_references()] == [["C", "O"]]


def test_centrosymmetric_molecule_longer_than_the_cell_is_built_whole():
    # a 6-C chain at 45 deg about the inversion centre (1/2, 1/2, 1/2) spans y = -0.16 to 1.16, longer than b;
    # its b-images are 2.8 A apart, so it is finite. The third ASU atom is the other half's end, given as its
    # wrapped inversion image (as pymatgen's CifWriter writes ASUs), so no copy of the chain is whole unless
    # symmetry images are wrapped into the cell before the -1..1 translations
    crystal = _crystal(
        (10.0, 4.0, 10.0),
        ["C", "C", "C"],
        [[0.55303, 0.63258, 0.5], [0.65910, 0.89775, 0.5], [0.23483, 0.83709, 0.5]],
        [INVERSION],
    )
    assert [r.symbols for r in crystal.get_references()] == [["C"] * 6]


def test_one_orbit_gives_one_reference_when_rounding_would_split_its_centroid():
    # every image of this lone atom is the same molecule, whatever its coordinate rounds to
    crystal = _crystal((10.0, 10.0, 10.0), ["Ar"], [[0.0015, 0.5, 0.5]])
    assert [r.symbols for r in crystal.get_references()] == [["Ar"]]


@pytest.mark.parametrize("N", [2, 3, 4])
def test_cutoff_with_no_partners_gives_no_multimers(N):
    # benzene's closest intermolecular contact is far longer than 0.5 A
    assert generate(load("benzene"), N, 0.5, verbose=0) == []
