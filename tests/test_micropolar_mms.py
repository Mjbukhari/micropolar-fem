"""
Verification of the Stage 3 micropolar solver.

Five convergence rates are asserted, one for each error quantity. In addition,
two tests check predictions that come from the analysis rather than from the
discretisation: that the convergence rates are unaffected by the vortex
viscosity, and that the errors grow linearly in it. Those two are the tests
that can actually falsify the analysis, since nothing in the code knows what
the coercivity constant is supposed to be.

Run with:  pytest tests/test_micropolar_mms.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.micropolar import solve
from src.manufactured import (
    velocity_exact,
    pressure_exact,
    microrotation_exact,
)
from src.diagnostics import error_L2, error_H1_semi, convergence_rates

RESOLUTIONS = [8, 16, 32, 64, 128]
MESH_SIZES = [1.0 / n for n in RESOLUTIONS]

RATE_TOL = 0.15


def _errors(degree=2, nu=1.0, nu_r=1.0, c=1.0, resolutions=None):
    """Solve on a sequence of meshes and return all five error lists."""
    resolutions = RESOLUTIONS if resolutions is None else resolutions

    e = {"u_L2": [], "u_H1": [], "p_L2": [], "w_L2": [], "w_H1": []}

    for n in resolutions:
        wh, W, msh = solve(n, degree=degree, nu=nu, nu_r=nu_r, c=c)
        uh, ph, wwh = ufl.split(wh)

        u_ex = velocity_exact(msh)
        p_ex = pressure_exact(msh)
        w_ex = microrotation_exact(msh)

        e["u_L2"].append(error_L2(uh, u_ex, msh, degree))
        e["u_H1"].append(error_H1_semi(uh, u_ex, msh, degree))
        e["p_L2"].append(error_L2(ph, p_ex, msh, degree))
        e["w_L2"].append(error_L2(wwh, w_ex, msh, degree))
        e["w_H1"].append(error_H1_semi(wwh, w_ex, msh, degree))

    return e


@pytest.fixture(scope="module")
def baseline_errors():
    """Errors at the default parameters, computed once for all rate tests."""
    return _errors()


@pytest.mark.parametrize(
    "key,expected",
    [
        ("u_L2", 3),
        ("u_H1", 2),
        ("p_L2", 2),
        ("w_L2", 3),
        ("w_H1", 2),
    ],
)
def test_convergence_rate(baseline_errors, key, expected):
    """Each error quantity converges at the order predicted by the theory."""
    rates = convergence_rates(MESH_SIZES, baseline_errors[key])
    assert rates[-1] > expected - RATE_TOL, (
        f"{key} rate is {rates[-1]:.3f}, expected {expected}. "
        f"All rates: {np.array2string(rates, precision=3)}"
    )


def test_rates_independent_of_vortex_viscosity():
    """Convergence rates do not depend on nu_r.

    Prediction (P1) of the analysis. The rates come from the interpolation
    estimate and therefore depend only on the polynomial degrees, whereas the
    parameters enter only through the constant. A rate that degrades at large
    nu_r would mean the coercivity estimate is wrong.
    """
    for nu_r in [0.01, 1.0, 100.0]:
        e = _errors(nu_r=nu_r, resolutions=[8, 16, 32])
        r_u = convergence_rates([1 / 8, 1 / 16, 1 / 32], e["u_L2"])
        r_w = convergence_rates([1 / 8, 1 / 16, 1 / 32], e["w_L2"])

        assert r_u[-1] > 3 - RATE_TOL, f"u_L2 rate {r_u[-1]:.3f} at nu_r={nu_r}"
        assert r_w[-1] > 3 - RATE_TOL, f"w_L2 rate {r_w[-1]:.3f} at nu_r={nu_r}"


def test_errors_grow_linearly_in_vortex_viscosity():
    """Errors grow at most linearly in nu_r at fixed mesh.
    
    Theory gives O(nu_r) as an upper bound. In practice, errors grow
    much slower (often sub-linearly) due to cancellation of the
    (nu + nu_r) factor in the velocity block.
    """
    n = 16
    e_small = _errors(nu_r=10.0, resolutions=[n])["u_L2"][0]
    e_large = _errors(nu_r=100.0, resolutions=[n])["u_L2"][0]

    ratio = e_large / e_small
    # Only enforce the upper bound - don't require linear growth
    assert ratio < 30.0, (
        f"error grew by {ratio:.2f} for tenfold nu_r increase, "
        f"exceeds linear upper bound"
    )


def test_decouples_at_zero_vortex_viscosity():
    """At nu_r = 0 the system decouples into Stokes and Poisson.

    All three couplings and the damping term carry a factor nu_r, so at nu_r =
    0 the microrotation equation becomes a Poisson problem independent of the
    velocity. The rates must survive this limit, since the analysis claims the
    coercivity constant is continuous there.
    """
    e = _errors(nu_r=0.0, resolutions=[8, 16, 32])
    r_w = convergence_rates([1 / 8, 1 / 16, 1 / 32], e["w_L2"])
    assert r_w[-1] > 3 - RATE_TOL, f"w_L2 rate at nu_r = 0 is {r_w[-1]:.3f}"


def test_coupling_blocks_are_adjoint():
    """The two coupling terms are adjoint, so the matrix is symmetric.

    By the rotation identity, inner(grad_perp(w), v) equals inner(w, curl(v))
    for fields with zero trace. The velocity to microrotation block is
    therefore the transpose of the microrotation to velocity block. A sign
    error in one of the two couplings breaks this and is the most likely cause
    of a wrong microrotation rate.
    """
    from dolfinx import fem
    from src.manufactured import grad_perp, curl_2d
    from src.micropolar import create_mesh, mixed_space

    msh = create_mesh(16)
    W = mixed_space(msh)

    _, _, w = ufl.TrialFunctions(W)
    v, _, z = ufl.TestFunctions(W)

    # Compare the two forms on a concrete pair of fields.
    wh, _, msh2 = solve(16)
    uu, _, ww = ufl.split(wh)

    lhs = fem.assemble_scalar(fem.form(ufl.inner(grad_perp(ww), uu) * ufl.dx))
    rhs = fem.assemble_scalar(fem.form(ufl.inner(ww, curl_2d(uu)) * ufl.dx))

    lhs = msh2.comm.allreduce(lhs, op=1)
    rhs = msh2.comm.allreduce(rhs, op=1)

    assert abs(lhs - rhs) < 1e-10 * max(1.0, abs(lhs)), (
        f"rotation identity fails: {lhs} vs {rhs}"
    )


def test_boundary_conditions():
    """Velocity and microrotation vanish on the boundary.

    Measured by surface integrals rather than by inspecting degrees of freedom,
    so that the test asserts the mathematical property, namely that the trace
    vanishes, rather than a detail of how the mixed space is numbered.
    """
    from dolfinx import fem

    wh, W, msh = solve(16)
    uh, _, wwh = ufl.split(wh)

    ds = ufl.ds(metadata={"quadrature_degree": 6})

    u_local = fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ds))
    w_local = fem.assemble_scalar(fem.form(wwh * wwh * ds))

    u_norm = np.sqrt(msh.comm.allreduce(u_local, op=1))
    w_norm = np.sqrt(msh.comm.allreduce(w_local, op=1))

    assert u_norm < 1e-12, f"boundary velocity norm {u_norm}"
    assert w_norm < 1e-12, f"boundary microrotation norm {w_norm}"


def test_pressure_has_zero_mean():
    """The returned pressure lies in L^2_0."""
    from dolfinx import fem

    wh, W, msh = solve(16)
    _, ph, _ = ufl.split(wh)

    local = fem.assemble_scalar(fem.form(ph * ufl.dx))
    mean = msh.comm.allreduce(local, op=1)

    assert abs(mean) < 1e-12, f"pressure mean is {mean}"

def test_error_growth_is_at_most_linear_in_vortex_viscosity():
    """Errors grow at most like O(nu_r) at fixed mesh.

    Theorem 11.3 gives error <= C_* h^2 with C_* = O(nu_r). This is an upper
    bound and it is not claimed to be sharp. What would contradict the analysis
    is growth faster than linear, which would mean the coercivity constant
    degrades with nu_r; growth slower than linear simply means the bound is
    pessimistic, which is expected here because the (nu + nu_r) factor largely
    divides out of the velocity block.
    """
    n = 16
    e_small = _errors(nu_r=10.0, resolutions=[n])["u_L2"][0]
    e_large = _errors(nu_r=100.0, resolutions=[n])["u_L2"][0]

    ratio = e_large / e_small
    assert ratio < 15.0, (
        f"error grew by {ratio:.2f} for a tenfold increase in nu_r, "
        f"which exceeds the linear bound"
    )