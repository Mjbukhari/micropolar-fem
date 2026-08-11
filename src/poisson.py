"""
Stage 1: Poisson problem on the unit square.

    -div(grad(u)) = f   in  Omega = (0,1)^2
                u = 0   on  boundary

Weak form (Lesson 1, Problem B): find u in H^1_0(Omega) such that

    inner(grad(u), grad(v)) dx = inner(f, v) dx    for all v in H^1_0(Omega)

This is the simplest instance of the machinery. There is no mixed space, no
pressure, and no coupling. Its purpose is to establish that the toolchain, the
manufactured solution, and the error measurement are all correct, so that when
those pieces are reused at Stage 2 and Stage 3 any wrong rate can be attributed
to the new terms rather than to the infrastructure.

Expected rates for degree k:
    L2 norm of the error          O(h^(k+1))
    H1 seminorm of the error      O(h^k)
"""

import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from src.manufactured import poisson_source

# Direct solver. At these problem sizes an LU factorisation is faster than any
# iterative method and removes solver tolerance as a possible source of error
# in the convergence study.
DIRECT_SOLVER = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


def create_mesh(n, comm=MPI.COMM_WORLD):
    """Uniform triangulation of the unit square with n cells per side.

    Parameters
    ----------
    n : int
        Number of cells along each edge. Mesh parameter is h = 1/n.
    comm : mpi4py communicator

    Returns
    -------
    dolfinx.mesh.Mesh
    """
    return dmesh.create_unit_square(comm, n, n)


def function_space(msh, degree):
    """Continuous Lagrange space of the given degree.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    degree : int

    Returns
    -------
    dolfinx.fem.FunctionSpace
    """
    return fem.functionspace(msh, ("Lagrange", degree))


def homogeneous_dirichlet_bc(V, msh):
    """Zero Dirichlet condition on the whole boundary.

    Three steps. First locate the boundary facets, which in two dimensions are
    the edges, hence entities of dimension topology.dim - 1. The marker
    function receives an array of shape (gdim, npoints) and must return a
    boolean array of length npoints; returning True everywhere selects the
    entire boundary, since locate_entities_boundary only ever offers boundary
    entities as candidates. Second, find the degrees of freedom attached to
    those facets. Third, construct the condition on those dofs.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
    msh : dolfinx.mesh.Mesh

    Returns
    -------
    dolfinx.fem.DirichletBC
    """
    fdim = msh.topology.dim - 1

    facets = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological(V, fdim, facets)

    return fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)


def bilinear_form(V):
    """The form a(u,v) = inner(grad(u), grad(v)) dx.

    This is the left hand side of the weak formulation derived in Lesson 1.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace

    Returns
    -------
    ufl form
    """
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    return ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx


def linear_form(V, f):
    """The form F(v) = inner(f, v) dx.

    Parameters
    ----------
    V : dolfinx.fem.FunctionSpace
    f : ufl expression

    Returns
    -------
    ufl form
    """
    v = ufl.TestFunction(V)
    return ufl.inner(f, v) * ufl.dx


def solve(n, degree=2, comm=MPI.COMM_WORLD):
    """Assemble and solve the Poisson problem on an n by n mesh.

    Parameters
    ----------
    n : int
        Cells per side.
    degree : int
        Polynomial degree of the Lagrange space.
    comm : mpi4py communicator

    Returns
    -------
    (dolfinx.fem.Function, dolfinx.mesh.Mesh)
        The computed solution and the mesh it was computed on. The mesh is
        returned because the caller needs it to build the exact solution as a
        UFL expression over the same spatial coordinate.
    """
    msh = create_mesh(n, comm)
    V = function_space(msh, degree)

    bc = homogeneous_dirichlet_bc(V, msh)
    f = poisson_source(msh)

    a = bilinear_form(V)
    L = linear_form(V, f)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options=DIRECT_SOLVER,
        petsc_options_prefix="poisson_",
    )
    uh = problem.solve()

    return uh, msh