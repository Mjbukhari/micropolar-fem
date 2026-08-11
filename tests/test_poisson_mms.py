"""
Verification of the Stage 1 Poisson solver by the method of manufactured
solutions.

These tests assert convergence rates. They do not plot them and they do not
print them for a human to inspect. A test that a person has to look at is not a
test. The point is that this suite runs in continuous integration and fails
loudly if a change to the discretisation breaks the rates.

Run with:  pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.poisson import solve
from src.manufactured import poisson_exact
from src.diagnostics import error_L2, error_H1_semi, convergence_rates

# Mesh resolutions. Coarse enough to run quickly, fine enough that the
# asymptotic regime is reached.
RESOLUTIONS = [8, 16, 32, 64]
MESH_SIZES = [1.0 / n for n in RESOLUTIONS]

# Tolerance on the observed rate. The theoretical rates are integers; 0.1 is
# tight enough to catch a genuine defect and loose enough to absorb the small
# preasymptotic deviation on the coarsest pair.
RATE_TOL = 0.1


def _errors(degree):
    """Solve on the whole sequence of meshes and return both error sequences."""
    e_l2, e_h1 = [], []
    for n in RESOLUTIONS:
        uh, msh = solve(n, degree=degree)
        u_ex = poisson_exact(msh)
        e_l2.append(error_L2(uh, u_ex, msh, degree))
        e_h1.append(error_H1_semi(uh, u_ex, msh, degree))
    return e_l2, e_h1


@pytest.mark.parametrize("degree", [1, 2])
def test_L2_convergence_rate(degree):
    """L2 error converges at order degree + 1."""
    e_l2, _ = _errors(degree)
    rates = convergence_rates(MESH_SIZES, e_l2)
    expected = degree + 1
    assert rates[-1] > expected - RATE_TOL, (
        f"L2 rate for degree {degree} is {rates[-1]:.3f}, expected {expected}. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


@pytest.mark.parametrize("degree", [1, 2])
def test_H1_convergence_rate(degree):
    """H1 seminorm error converges at order degree."""
    _, e_h1 = _errors(degree)
    rates = convergence_rates(MESH_SIZES, e_h1)
    expected = degree
    assert rates[-1] > expected - RATE_TOL, (
        f"H1 rate for degree {degree} is {rates[-1]:.3f}, expected {expected}. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


def test_errors_decrease_monotonically():
    """A basic sanity check, independent of the rate."""
    e_l2, e_h1 = _errors(degree=2)
    assert all(a > b for a, b in zip(e_l2, e_l2[1:])), f"L2 errors not decreasing: {e_l2}"
    assert all(a > b for a, b in zip(e_h1, e_h1[1:])), f"H1 errors not decreasing: {e_h1}"


def test_boundary_condition_is_imposed():
    """The computed solution vanishes on the boundary to machine precision."""
    from dolfinx import mesh as dmesh, fem

    uh, msh = solve(16, degree=2)
    V = uh.function_space
    facets = dmesh.locate_entities_boundary(
        msh, msh.topology.dim - 1, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    assert np.max(np.abs(uh.x.array[dofs])) < 1e-12