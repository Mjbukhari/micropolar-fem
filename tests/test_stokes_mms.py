"""
Verification of the Stage 2 Stokes solver by the method of manufactured
solutions.

Three rates are asserted, not two: the pressure now has its own error and its
own convergence order, and a mistake in the pressure treatment can leave the
velocity rates untouched. Checking only the velocity would miss a broken
inf-sup pair entirely, which is precisely the failure mode described in
Lesson 4.

Run with:  pytest tests/test_stokes_mms.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stokes import solve
from src.manufactured import stokes_velocity_exact, stokes_pressure_exact
from src.diagnostics import error_L2, error_H1_semi, convergence_rates

RESOLUTIONS = [8, 16, 32, 64]
MESH_SIZES = [1.0 / n for n in RESOLUTIONS]

RATE_TOL = 0.1


def _errors(degree=2):
    """Solve on the whole sequence of meshes and return the three error lists."""
    e_u_l2, e_u_h1, e_p_l2 = [], [], []

    for n in RESOLUTIONS:
        wh, W, msh = solve(n, degree=degree)
        uh, ph = ufl.split(wh)

        u_ex = stokes_velocity_exact(msh)
        p_ex = stokes_pressure_exact(msh)

        e_u_l2.append(error_L2(uh, u_ex, msh, degree))
        e_u_h1.append(error_H1_semi(uh, u_ex, msh, degree))
        e_p_l2.append(error_L2(ph, p_ex, msh, degree))

    return e_u_l2, e_u_h1, e_p_l2


def test_velocity_L2_rate():
    """Velocity L2 error converges at order degree + 1, that is 3 for P2."""
    e_u_l2, _, _ = _errors()
    rates = convergence_rates(MESH_SIZES, e_u_l2)
    assert rates[-1] > 3 - RATE_TOL, (
        f"velocity L2 rate is {rates[-1]:.3f}, expected 3. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


def test_velocity_H1_rate():
    """Velocity H1 seminorm error converges at order degree, that is 2 for P2."""
    _, e_u_h1, _ = _errors()
    rates = convergence_rates(MESH_SIZES, e_u_h1)
    assert rates[-1] > 2 - RATE_TOL, (
        f"velocity H1 rate is {rates[-1]:.3f}, expected 2. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


def test_pressure_L2_rate():
    """Pressure L2 error converges at order degree, that is 2 for P1 pressure.

    A degraded rate here with healthy velocity rates is the signature of an
    unstable velocity-pressure pair.
    """
    _, _, e_p_l2 = _errors()
    rates = convergence_rates(MESH_SIZES, e_p_l2)
    assert rates[-1] > 2 - RATE_TOL, (
        f"pressure L2 rate is {rates[-1]:.3f}, expected 2. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


def test_discrete_divergence_is_small():
    """The computed velocity is discretely divergence free.

    Exact pointwise vanishing is not expected: Taylor-Hood enforces the
    constraint only weakly, against the discrete pressure space, so div(u_h) is
    orthogonal to Q_h rather than identically zero. This is Proposition on the
    kernels differing, in Lesson 4. What is expected is that the residual is
    small and decreasing under refinement.
    """
    from dolfinx import fem

    prev = None
    for n in [8, 16, 32]:
        wh, W, msh = solve(n)
        uh, _ = ufl.split(wh)

        local = fem.assemble_scalar(
            fem.form(ufl.div(uh) * ufl.div(uh) * ufl.dx)
        )
        norm = np.sqrt(msh.comm.allreduce(local, op=1))

        if prev is not None:
            assert norm < prev, f"divergence residual not decreasing: {prev} -> {norm}"
        prev = norm


def test_pressure_has_zero_mean():
    """The returned pressure lies in L^2_0, as the analysis requires."""
    from dolfinx import fem

    wh, W, msh = solve(16)
    _, ph = ufl.split(wh)

    local = fem.assemble_scalar(fem.form(ph * ufl.dx))
    mean = msh.comm.allreduce(local, op=1)

    assert abs(mean) < 1e-12, f"pressure mean is {mean}, expected 0"


def test_velocity_boundary_condition():
    """The computed velocity vanishes on the boundary.

    Measured as the L2 norm of the velocity over the boundary, which is the
    trace that H^1_0 membership requires to vanish. This avoids reaching into
    the dof numbering of the mixed space.
    """
    from dolfinx import fem

    wh, W, msh = solve(16)
    uh, _ = ufl.split(wh)

    ds = ufl.ds(metadata={"quadrature_degree": 6})
    local = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds))
    norm = np.sqrt(msh.comm.allreduce(local, op=1))

    assert norm < 1e-12, f"boundary velocity L2 norm is {norm}"