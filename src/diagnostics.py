"""
Error norms and convergence rates.

The errors are computed by integrating the UFL expression for the exact
solution directly, with the quadrature degree raised well above the degree of
the discrete space. The exact solution is never interpolated into a finite
element space, so no interpolation error enters the measurement.

Interpolating the exact solution into the same space used for the solution and
then differencing nodal values is the standard way to get wrong convergence
rates: what is measured is then the interpolation error, and the observed rate
plateaus for reasons unrelated to the discretisation.
"""

import numpy as np
import ufl
from dolfinx import fem
from mpi4py import MPI


def _integrate(mesh, integrand, quadrature_degree):
    """Assemble a scalar integral over the whole mesh, summed across ranks."""
    dx = ufl.dx(metadata={"quadrature_degree": quadrature_degree})
    local = fem.assemble_scalar(fem.form(integrand * dx))
    return mesh.comm.allreduce(local, op=MPI.SUM)


def error_L2(uh, u_exact, mesh, degree):
    """L2 norm of uh - u_exact.

    Parameters
    ----------
    uh : dolfinx.fem.Function
        Computed solution.
    u_exact : ufl expression
        Exact solution.
    mesh : dolfinx.mesh.Mesh
    degree : int
        Polynomial degree of the discrete space. Quadrature is set to
        2*degree + 3, which is more than enough to resolve the integrand.

    Returns
    -------
    float
    """
    e = uh - u_exact
    return np.sqrt(_integrate(mesh, ufl.inner(e, e), 2 * degree + 3))


def error_H1_semi(uh, u_exact, mesh, degree):
    """H1 seminorm of uh - u_exact, that is the L2 norm of the gradient error.

    Parameters
    ----------
    See error_L2.

    Returns
    -------
    float
    """
    e = ufl.grad(uh - u_exact)
    return np.sqrt(_integrate(mesh, ufl.inner(e, e), 2 * degree + 3))


def convergence_rates(mesh_sizes, errors):
    """Observed convergence rates between successive refinements.

    Computes log(e_{m-1} / e_m) / log(h_{m-1} / h_m) for each consecutive pair.

    Parameters
    ----------
    mesh_sizes : sequence of float
        Mesh parameters h, in decreasing order.
    errors : sequence of float
        Corresponding error values.

    Returns
    -------
    numpy.ndarray
        Array of length len(errors) - 1.
    """
    h = np.asarray(mesh_sizes, dtype=float)
    e = np.asarray(errors, dtype=float)
    return np.log(e[:-1] / e[1:]) / np.log(h[:-1] / h[1:])


def rate_table(mesh_sizes, error_dict):
    """Format a convergence table as a string.

    Parameters
    ----------
    mesh_sizes : sequence of float
    error_dict : dict
        Mapping from norm name to sequence of errors.

    Returns
    -------
    str
    """
    names = list(error_dict.keys())
    header = f"{'h':>10} " + " ".join(f"{n:>14} {'rate':>6}" for n in names)
    lines = [header, "-" * len(header)]

    rates = {n: convergence_rates(mesh_sizes, error_dict[n]) for n in names}

    for i, h in enumerate(mesh_sizes):
        row = f"{h:10.5f} "
        for n in names:
            r = "" if i == 0 else f"{rates[n][i-1]:6.2f}"
            row += f" {error_dict[n][i]:14.6e} {r:>6}"
        lines.append(row)

    return "\n".join(lines)