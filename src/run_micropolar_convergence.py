"""
Print the convergence table for the Stage 3 micropolar solver.

Usage:  python scripts/run_micropolar_convergence.py [degree] [nu_r]
"""

import sys
from pathlib import Path

import ufl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.micropolar import solve
from src.manufactured import (
    velocity_exact,
    pressure_exact,
    microrotation_exact,
)
from src.diagnostics import error_L2, error_H1_semi, rate_table

RESOLUTIONS = [8, 16, 32, 64, 128]


def main(degree=2, nu_r=1.0):
    mesh_sizes = []
    e = {"u, L2": [], "u, H1": [], "p, L2": [], "w, L2": [], "w, H1": []}

    for n in RESOLUTIONS:
        wh, W, msh = solve(n, degree=degree, nu_r=nu_r)
        uh, ph, wwh = ufl.split(wh)

        u_ex = velocity_exact(msh)
        p_ex = pressure_exact(msh)
        w_ex = microrotation_exact(msh)

        mesh_sizes.append(1.0 / n)
        e["u, L2"].append(error_L2(uh, u_ex, msh, degree))
        e["u, H1"].append(error_H1_semi(uh, u_ex, msh, degree))
        e["p, L2"].append(error_L2(ph, p_ex, msh, degree))
        e["w, L2"].append(error_L2(wwh, w_ex, msh, degree))
        e["w, H1"].append(error_H1_semi(wwh, w_ex, msh, degree))

    print(f"\nMicropolar, P{degree}/P{degree - 1}/P{degree}, nu_r = {nu_r}")
    print(f"Expected rates: u L2 = {degree + 1}, u H1 = {degree}, "
          f"p L2 = {degree}, w L2 = {degree + 1}, w H1 = {degree}\n")
    print(rate_table(mesh_sizes, e))
    print()


if __name__ == "__main__":
    degree = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    nu_r = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    main(degree, nu_r)