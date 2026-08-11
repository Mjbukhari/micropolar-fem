"""
Print the convergence table for the Stage 2 Stokes solver.

Usage:  python scripts/run_stokes_convergence.py [velocity_degree]
"""

import sys
from pathlib import Path

import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.stokes import solve
from src.manufactured import stokes_velocity_exact, stokes_pressure_exact
from src.diagnostics import error_L2, error_H1_semi, rate_table

RESOLUTIONS = [8, 16, 32, 64]


def main(degree=2):
    mesh_sizes, e_u_l2, e_u_h1, e_p_l2 = [], [], [], []

    for n in RESOLUTIONS:
        wh, W, msh = solve(n, degree=degree)
        uh, ph = ufl.split(wh)

        u_ex = stokes_velocity_exact(msh)
        p_ex = stokes_pressure_exact(msh)

        mesh_sizes.append(1.0 / n)
        e_u_l2.append(error_L2(uh, u_ex, msh, degree))
        e_u_h1.append(error_H1_semi(uh, u_ex, msh, degree))
        e_p_l2.append(error_L2(ph, p_ex, msh, degree))

    print(f"\nStokes, P{degree}/P{degree - 1} Taylor-Hood on the unit square")
    print(f"Expected rates: velocity L2 = {degree + 1}, "
          f"velocity H1 = {degree}, pressure L2 = {degree}\n")
    print(rate_table(mesh_sizes, {
        "u, L2": e_u_l2,
        "u, H1": e_u_h1,
        "p, L2": e_p_l2,
    }))
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
