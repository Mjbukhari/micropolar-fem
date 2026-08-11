"""
Stage 2: Stokes problem on the unit square, Taylor-Hood elements.

    -nu*div(grad(u)) + grad(p) = f   in  Omega = (0,1)^2
                       div(u) = 0   in  Omega
                            u = 0   on  boundary

Mixed weak form (Lesson 3): find (u,p) in V x Q with V = [H^1_0]^2 and
Q = L^2_0 such that

    nu * inner(grad(u), grad(v)) dx + b(v,p) = inner(f,v) dx   for all v in V
                                     b(u,q) = 0                for all q in Q

where b(v,q) = -inner(q, div(v)) dx.

This is the first saddle point problem in the sequence. Three things appear
here that had no analogue at Stage 1: the mixed function space, the fact that
the resulting matrix is indefinite rather than positive definite, and the
constant pressure nullspace which must be removed before the system can be
factorised.

Expected rates for P2/P1:
    velocity, L2 norm         O(h^3)
    velocity, H1 seminorm     O(h^2)
    pressure, L2 norm         O(h^2)
"""

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from petsc4py import PETSc

from src.manufactured import stokes_source

# Direct solver. The Stokes matrix is symmetric but indefinite, so the
# conjugate gradient method is not applicable; see Lesson 4, Proposition on
# indefiniteness. MUMPS performs an LU factorisation and handles indefinite
# systems without difficulty at these sizes.
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
        Cells along each edge. Mesh parameter h = 1/n.
    comm : mpi4py communicator

    Returns
    -------
    dolfinx.mesh.Mesh
    """
    return dmesh.create_unit_square(comm, n, n)


def taylor_hood_space(msh, degree=2):
    """Mixed Taylor-Hood space of degree (k, k-1) for velocity and pressure.

    The velocity is a vector valued continuous Lagrange field of degree k and
    the pressure a scalar continuous Lagrange field of degree k-1. Making the
    velocity space one degree richer than the pressure space is what supplies
    the discrete inf-sup condition; the equal order choice P1/P1 fails, and
    fails in the specific way described in Lesson 4.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    degree : int
        Velocity degree k. Pressure degree is k-1.

    Returns
    -------
    dolfinx.fem.FunctionSpace
        The mixed space W. Velocity is W.sub(0), pressure is W.sub(1).
    """
    cell = msh.basix_cell()

    velocity_element = basix.ufl.element(
        "Lagrange", cell, degree, shape=(msh.geometry.dim,)
    )
    pressure_element = basix.ufl.element("Lagrange", cell, degree - 1)

    mixed = basix.ufl.mixed_element([velocity_element, pressure_element])

    return fem.functionspace(msh, mixed)


def velocity_bc(W, msh):
    """Zero Dirichlet condition on the velocity over the whole boundary.

    The condition applies to the velocity subspace only, not to the pressure.
    Locating degrees of freedom in a subspace of a mixed space requires the
    pair (W.sub(0), V), where V is the collapsed velocity space, because the
    condition must be expressed both in the numbering of the mixed space and
    in the numbering of the standalone velocity space.

    Parameters
    ----------
    W : dolfinx.fem.FunctionSpace
        Mixed space.
    msh : dolfinx.mesh.Mesh

    Returns
    -------
    dolfinx.fem.DirichletBC
    """
    fdim = msh.topology.dim - 1

    V, _ = W.sub(0).collapse()

    facets = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

    zero = fem.Function(V)
    zero.x.array[:] = 0.0

    return fem.dirichletbc(zero, dofs, W.sub(0))


def pressure_pin(W, msh):
    """Pin one pressure degree of freedom to remove the constant nullspace.

    The pressure appears in the equations only through its gradient, so p and
    p + C give the same velocity for every constant C. In the discrete system
    this means the vector representing a constant pressure lies in the kernel
    of B transpose, the matrix is singular, and the LU factorisation fails.
    This is Remark 2.6 of Lesson 3 in its concrete form: passing to L^2_0 in
    the analysis and removing the singular mode in the code are the same act.

    The remedy here is to fix the pressure at a single point, which selects one
    representative from the family {p + C}. The representative selected is not
    the zero mean one, so the computed pressure is shifted afterwards; see
    normalise_pressure.

    Parameters
    ----------
    W : dolfinx.fem.FunctionSpace
    msh : dolfinx.mesh.Mesh

    Returns
    -------
    dolfinx.fem.DirichletBC
    """
    Q, _ = W.sub(1).collapse()

    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

    dofs = fem.locate_dofs_geometrical((W.sub(1), Q), corner)

    zero = fem.Function(Q)
    zero.x.array[:] = 0.0

    return fem.dirichletbc(zero, dofs, W.sub(1))


def normalise_pressure(wh, W, msh):
    """Shift the computed pressure so that it has zero mean.

    Pinning a single degree of freedom fixes the additive constant, but not at
    the value that places the pressure in L^2_0. Since the exact pressure was
    constructed with zero mean, the computed pressure must be shifted by its
    own mean before the two are compared, or the error will be dominated by a
    constant offset and the convergence rate will be meaningless.

    Parameters
    ----------
    wh : dolfinx.fem.Function
        Mixed solution, modified in place.
    W : dolfinx.fem.FunctionSpace
    msh : dolfinx.mesh.Mesh
    """
    _, p_map = W.sub(1).collapse()

    _, ph_ufl = ufl.split(wh)

    mean_local = fem.assemble_scalar(fem.form(ph_ufl * ufl.dx))
    area_local = fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ufl.dx))

    mean = msh.comm.allreduce(mean_local, op=MPI.SUM)
    area = msh.comm.allreduce(area_local, op=MPI.SUM)

    wh.x.array[p_map] -= mean / area
    wh.x.scatter_forward()


def bilinear_form(W, nu):
    """The mixed bilinear form.

        a((u,p),(v,q)) = nu*inner(grad(u),grad(v)) dx
                         - inner(p, div(v)) dx
                         - inner(q, div(u)) dx

    The two off diagonal terms are b(v,p) and b(u,q) of Lesson 3. They carry
    the same sign here, which makes the resulting matrix symmetric. Note that
    there is no term pairing p with q: the pressure never appears against
    itself, which is exactly why the form is not coercive and why the matrix is
    indefinite.

    Parameters
    ----------
    W : dolfinx.fem.FunctionSpace
    nu : float

    Returns
    -------
    ufl form
    """
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)

    return (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(q, ufl.div(u)) * ufl.dx
    )


def linear_form(W, f):
    """The form F(v,q) = inner(f,v) dx.

    The continuity equation has no source, since the exact velocity is
    divergence free by construction, so the pressure test function q does not
    appear.

    Parameters
    ----------
    W : dolfinx.fem.FunctionSpace
    f : ufl expression, vector valued

    Returns
    -------
    ufl form
    """
    v, _ = ufl.TestFunctions(W)
    return ufl.inner(f, v) * ufl.dx


def solve(n, degree=2, nu=1.0, comm=MPI.COMM_WORLD):
    """Assemble and solve the Stokes problem on an n by n mesh.

    Parameters
    ----------
    n : int
        Cells per side.
    degree : int
        Velocity degree. Pressure degree is degree - 1.
    nu : float
        Viscosity.
    comm : mpi4py communicator

    Returns
    -------
    (dolfinx.fem.Function, dolfinx.fem.FunctionSpace, dolfinx.mesh.Mesh)
        The mixed solution, the mixed space, and the mesh. All three are
        returned because the caller needs the space to split the solution and
        the mesh to build the exact fields over the same spatial coordinate.
    """
    msh = create_mesh(n, comm)
    W = taylor_hood_space(msh, degree)

    bcs = [velocity_bc(W, msh), pressure_pin(W, msh)]
    f = stokes_source(msh, nu)

    a = bilinear_form(W, nu)
    L = linear_form(W, f)

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options=DIRECT_SOLVER,
        petsc_options_prefix="stokes_",
    )
    wh = problem.solve()

    normalise_pressure(wh, W, msh)

    return wh, W, msh