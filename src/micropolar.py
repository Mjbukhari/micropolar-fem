"""
Stage 3: the stationary linear micropolar system on the unit square.

    -(nu + nu_r) div(grad(u)) + grad(p) = 2*nu_r*grad_perp(w) + f
                                 div(u) = 0
       -c div(grad(w)) + 4*nu_r*w       = 2*nu_r*curl(u) + g

with u = 0 and w = 0 on the boundary, and c = c_a + c_d.

Mixed weak form: find (u, p, w) in V x Q x W, with V = [H^1_0]^2, Q = L^2_0 and
W = H^1_0, such that for all (v, q, z),

    a((u,w),(v,z)) + b(v,p) + b(u,q) = F(v,z)

where

    a((u,w),(v,z)) = (nu + nu_r) inner(grad u, grad v)
                   + c inner(grad w, grad z)
                   + 4 nu_r inner(w, z)
                   - 2 nu_r inner(grad_perp w, v)
                   - 2 nu_r inner(curl u, z)

    b(v,q) = -inner(q, div v).

The two coupling terms are the only structural addition over Stage 2. They have
no sign, so they are what obstructs coercivity, and absorbing them is what fixed
the Young parameter at epsilon = 1/2 in the analysis.

Expected rates for P2/P1/P2:
    velocity, L2                O(h^3)
    velocity, H1 seminorm       O(h^2)
    pressure, L2                O(h^2)
    microrotation, L2           O(h^3)
    microrotation, H1 seminorm  O(h^2)
"""

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

from src.manufactured import (
    grad_perp,
    curl_2d,
    micropolar_momentum_source,
    micropolar_rotation_source,
)

# Direct solver. The system is symmetric but indefinite, because the pressure
# never pairs with itself and the corresponding diagonal block is zero, so the
# conjugate gradient method does not apply. MUMPS performs an LU factorisation
# and handles indefinite systems without difficulty at these sizes.
DIRECT_SOLVER = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}

# Default physical parameters. The vortex viscosity nu_r is the one varied in
# the parameter study; the others are held fixed.
DEFAULTS = {"nu": 1.0, "nu_r": 1.0, "c": 1.0}


def create_mesh(n, comm=MPI.COMM_WORLD):
    """Uniform triangulation of the unit square with n cells per side."""
    return dmesh.create_unit_square(comm, n, n)


def mixed_space(msh, degree=2):
    """Three field mixed space for velocity, pressure and microrotation.

    Velocity is a vector valued continuous Lagrange field of degree k, pressure
    a scalar of degree k-1, and microrotation a scalar of degree k. The
    velocity and pressure degrees are related as in Taylor-Hood, since it is
    that pair which must satisfy the discrete inf-sup condition; the
    microrotation is unconstrained by inf-sup considerations, since it does not
    appear in the constraint form b, and is given degree k so that it converges
    at the same rate as the velocity.

    Parameters
    ----------
    msh : dolfinx.mesh.Mesh
    degree : int
        Velocity and microrotation degree k. Pressure degree is k-1.

    Returns
    -------
    dolfinx.fem.FunctionSpace
        Mixed space W. Velocity is W.sub(0), pressure W.sub(1), microrotation
        W.sub(2).
    """
    cell = msh.basix_cell()

    velocity = basix.ufl.element(
        "Lagrange", cell, degree, shape=(msh.geometry.dim,)
    )
    pressure = basix.ufl.element("Lagrange", cell, degree - 1)
    rotation = basix.ufl.element("Lagrange", cell, degree)

    return fem.functionspace(
        msh, basix.ufl.mixed_element([velocity, pressure, rotation])
    )


def _zero_bc_on_subspace(W, msh, index):
    """Homogeneous Dirichlet condition on one subspace over the whole boundary.

    Locating degrees of freedom in a subspace of a mixed space requires the
    pair (W.sub(i), S), where S is the collapsed subspace, because the
    condition must be expressed in the numbering of both spaces at once.
    """
    fdim = msh.topology.dim - 1

    S, _ = W.sub(index).collapse()

    facets = dmesh.locate_entities_boundary(
        msh, fdim, lambda x: np.full(x.shape[1], True)
    )
    dofs = fem.locate_dofs_topological((W.sub(index), S), fdim, facets)

    zero = fem.Function(S)
    zero.x.array[:] = 0.0

    return fem.dirichletbc(zero, dofs, W.sub(index))


def velocity_bc(W, msh):
    """No slip condition on the velocity."""
    return _zero_bc_on_subspace(W, msh, 0)


def rotation_bc(W, msh):
    """Homogeneous Dirichlet condition on the microrotation."""
    return _zero_bc_on_subspace(W, msh, 2)


def pressure_pin(W, msh):
    """Pin one pressure degree of freedom to remove the constant nullspace.

    The pressure enters the equations only through its gradient, so p and p + C
    give the same velocity for every constant C. Discretely, the vector
    representing a constant pressure lies in the kernel of B transpose, the
    matrix is singular, and the factorisation fails. Restricting the pressure
    space to L^2_0 in the analysis and removing this mode in the code are the
    same act.

    Pinning selects one representative of the family {p + C}, but not the zero
    mean one, so the result is shifted afterwards; see normalise_pressure.
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

    The exact pressure was constructed with zero mean, so the computed pressure
    must be placed in the same equivalence class representative before the two
    are compared. Without this the error is dominated by a constant offset and
    the convergence rate is meaningless.
    """
    _, p_map = W.sub(1).collapse()

    _, ph, _ = ufl.split(wh)

    mean_local = fem.assemble_scalar(fem.form(ph * ufl.dx))
    area_local = fem.assemble_scalar(
        fem.form(fem.Constant(msh, 1.0) * ufl.dx)
    )

    mean = msh.comm.allreduce(mean_local, op=MPI.SUM)
    area = msh.comm.allreduce(area_local, op=MPI.SUM)

    wh.x.array[p_map] -= mean / area
    wh.x.scatter_forward()


def bilinear_form(W, msh, nu, nu_r, c):
    u, p, w = ufl.TrialFunctions(W)
    v, q, z = ufl.TestFunctions(W)

    # Wrapped as Constants rather than Python floats so that the forms retain
    # their integration domain when a coefficient is zero. A float zero causes
    # UFL to collapse the term to Zero(), which has no domain and cannot be
    # integrated.
    nu_ = fem.Constant(msh, float(nu))
    nu_r_ = fem.Constant(msh, float(nu_r))
    c_ = fem.Constant(msh, float(c))

    viscous = (nu_ + nu_r_) * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    angular = c_ * ufl.inner(ufl.grad(w), ufl.grad(z)) * ufl.dx
    damping = 4.0 * nu_r_ * ufl.inner(w, z) * ufl.dx

    coupling_uw = -2.0 * nu_r_ * ufl.inner(grad_perp(w), v) * ufl.dx
    coupling_wu = -2.0 * nu_r_ * ufl.inner(curl_2d(u), z) * ufl.dx

    constraint = (
        -ufl.inner(p, ufl.div(v)) * ufl.dx
        - ufl.inner(q, ufl.div(u)) * ufl.dx
    )

    return viscous + angular + damping + coupling_uw + coupling_wu + constraint


def linear_form(W, f, g):
    """The form F(v,q,z) = inner(f,v) dx + inner(g,z) dx.

    The continuity equation has no source, since the exact velocity is
    divergence free by construction, so the pressure test function q does not
    appear.
    """
    v, _, z = ufl.TestFunctions(W)
    return ufl.inner(f, v) * ufl.dx + ufl.inner(g, z) * ufl.dx


def solve(n, degree=2, nu=None, nu_r=None, c=None, comm=MPI.COMM_WORLD):
    """Assemble and solve the micropolar system on an n by n mesh.

    Parameters
    ----------
    n : int
        Cells per side.
    degree : int
        Velocity and microrotation degree. Pressure degree is degree - 1.
    nu, nu_r, c : float, optional
        Kinematic viscosity, vortex viscosity, angular viscosity. Defaults
        from DEFAULTS.
    comm : mpi4py communicator

    Returns
    -------
    (dolfinx.fem.Function, dolfinx.fem.FunctionSpace, dolfinx.mesh.Mesh)
        Mixed solution, mixed space, and mesh.
    """
    nu = DEFAULTS["nu"] if nu is None else nu
    nu_r = DEFAULTS["nu_r"] if nu_r is None else nu_r
    c = DEFAULTS["c"] if c is None else c

    msh = create_mesh(n, comm)
    W = mixed_space(msh, degree)

    bcs = [velocity_bc(W, msh), rotation_bc(W, msh), pressure_pin(W, msh)]

    f = micropolar_momentum_source(msh, nu, nu_r)
    g = micropolar_rotation_source(msh, nu_r, c)

    a = bilinear_form(W, msh, nu, nu_r, c)
    L = linear_form(W, f, g)

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options=DIRECT_SOLVER,
        petsc_options_prefix="micropolar_",
    )
    wh = problem.solve()

    normalise_pressure(wh, W, msh)

    return wh, W, msh