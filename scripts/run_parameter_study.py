"""
Parameter study: dependence of the error on the vortex viscosity.

This is the script that tests the analysis rather than the implementation. The
convergence table checks that the code solves the equations it claims to solve.
This script checks two predictions that follow from the well-posedness
estimates and from nothing else:

  (P1)  The convergence rates are independent of nu_r, because the rates come
        from the interpolation estimate and depend only on the polynomial
        degrees.

  (P2)  The errors grow linearly in nu_r at fixed mesh, because the coercivity
        constant alpha is independent of nu_r while the boundedness constant M
        grows linearly, so the ratio M/alpha appearing in the quasi-optimality
        estimate is O(nu_r).

Prediction (P2) is the falsifiable one. Nothing in the solver knows what alpha
is; if the coercivity estimate were wrong, in particular if alpha degraded with
nu_r, the observed growth would be faster than linear.

Usage:  python scripts/run_parameter_study.py
"""

import sys
from pathlib import Path

import numpy as np
import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.micropolar import solve
from src.manufactured import (
    velocity_exact,
    pressure_exact,
    microrotation_exact,
)
from src.diagnostics import error_L2, error_H1_semi, convergence_rates

NU_R_VALUES = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RESOLUTIONS = [8, 16, 32, 64, 128]
MESH_SIZES = [1.0 / n for n in RESOLUTIONS]
FIXED_N = 128


def _errors_at(nu_r, resolutions):
    """Return the five error lists for a given nu_r."""
    out = {"u_L2": [], "u_H1": [], "p_L2": [], "w_L2": [], "w_H1": []}

    for n in resolutions:
        wh, W, msh = solve(n, nu_r=nu_r)
        uh, ph, wwh = ufl.split(wh)

        out["u_L2"].append(error_L2(uh, velocity_exact(msh), msh, 2))
        out["u_H1"].append(error_H1_semi(uh, velocity_exact(msh), msh, 2))
        out["p_L2"].append(error_L2(ph, pressure_exact(msh), msh, 2))
        out["w_L2"].append(error_L2(wwh, microrotation_exact(msh), msh, 2))
        out["w_H1"].append(error_H1_semi(wwh, microrotation_exact(msh), msh, 2))

    return out


def rates_table():
    """Prediction (P1): rates should be constant down the nu_r column."""
    print("\n" + "=" * 74)
    print("Observed convergence rates as a function of nu_r")
    print("Prediction (P1): every column constant, at 3, 2, 2, 3, 2")
    print("=" * 74)
    print(f"{'nu_r':>10} {'u L2':>10} {'u H1':>10} {'p L2':>10} "
          f"{'w L2':>10} {'w H1':>10}")
    print("-" * 74)

    for nu_r in NU_R_VALUES:
        e = _errors_at(nu_r, RESOLUTIONS)
        r = {k: convergence_rates(MESH_SIZES, v) for k, v in e.items()}
        print(f"{nu_r:10.3g}  u_L2: {np.array2string(r['u_L2'], precision=2)}"
              f"  w_L2: {np.array2string(r['w_L2'], precision=2)}")


def growth_table():
    """Prediction (P2): errors should grow like O(nu_r) at fixed mesh."""
    print("\n" + "=" * 74)
    print(f"Errors at fixed mesh h = 1/{FIXED_N}, as a function of nu_r")
    print("Prediction (P2): errors O(nu_r), so the scaled column is constant")
    print("=" * 74)
    print(f"{'nu_r':>10} {'u L2 error':>16} {'error / nu_r':>16} "
          f"{'w L2 error':>16}")
    print("-" * 74)

    for nu_r in NU_R_VALUES:
        e = _errors_at(nu_r, [FIXED_N])
        eu = e["u_L2"][0]
        ew = e["w_L2"][0]
        scaled = eu / nu_r if nu_r > 0 else float("nan")
        scaled_str = "n/a" if nu_r == 0 else f"{scaled:16.6e}"
        print(f"{nu_r:10.3g} {eu:16.6e} {scaled_str:>16} {ew:16.6e}")

    print("\nThe scaled column should approach a constant for large nu_r.")
    print("Growth faster than linear would contradict the claim that the")
    print("coercivity constant is independent of nu_r.")


def condition_estimate():
    """Report the predicted ratio M/alpha for reference."""
    nu, c, C_omega = 1.0, 1.0, 1.0 / (np.sqrt(2)*np.pi )  # Poincare constant for the unit square

    print("\n" + "=" * 74)
    print("Predicted ratio M / alpha")
    print("=" * 74)
    print(f"{'nu_r':>10} {'M':>14} {'alpha':>14} {'M / alpha':>14}")
    print("-" * 74)

    alpha = min(nu, c) / (C_omega**2 + 1.0)
    for nu_r in NU_R_VALUES:
        M = nu + c + 9.0 * nu_r
        print(f"{nu_r:10.3g} {M:14.4f} {alpha:14.4f} {M / alpha:14.4f}")


if __name__ == "__main__":
    rates_table()
    growth_table()
    condition_estimate()
    print()