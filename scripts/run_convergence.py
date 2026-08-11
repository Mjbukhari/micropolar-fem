"""
Print the convergence table for the Stage 1 Poisson solver.

The output of this script goes in the README. A professor opening the
repository should see observed rates next to theoretical ones on the front
page, without cloning anything or running anything.

Usage:  python scripts/run_convergence.py [degree]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.poisson import solve
from src.manufactured import poisson_exact
from src.diagnostics import error_L2, error_H1_semi, rate_table

RESOLUTIONS = [8, 16, 32, 64, 128]


def main(degree=2):
    mesh_sizes, e_l2, e_h1 = [], [], []

    for n in RESOLUTIONS:
        uh, msh = solve(n, degree=degree)
        u_ex = poisson_exact(msh)
        mesh_sizes.append(1.0 / n)
        e_l2.append(error_L2(uh, u_ex, msh, degree))
        e_h1.append(error_H1_semi(uh, u_ex, msh, degree))

    print(f"\nPoisson, P{degree} elements on the unit square")
    print(f"Expected rates: L2 = {degree + 1}, H1 = {degree}\n")
    print(rate_table(mesh_sizes, {"L2 error": e_l2, "H1 error": e_h1}))
    print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)