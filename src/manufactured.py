"""
Manufactured solutions for verification.

Every exact solution is a UFL expression of the spatial coordinate, and every
forcing term is obtained by applying the differential operator to it
symbolically. There is no transcription step, so there is no opportunity for a
sign error, and the forcing is consistent with the operator actually being
discretised by construction.
"""

import ufl


# -----------------------------------------------------------------------------
# Two dimensional differential operators
# -----------------------------------------------------------------------------

def grad_perp(w):
    """Perpendicular gradient of a scalar, grad_perp(w) = (d_2 w, -d_1 w).

    The gradient rotated by ninety degrees. Maps scalars to vectors, and is the
    operator by which the microrotation acts on the velocity equation.
    """
    g = ufl.grad(w)
    return ufl.as_vector([g[1], -g[0]])


def curl_2d(u):
    """Scalar curl of a plane vector field, curl(u) = d_1 u_2 - d_2 u_1.

    In three dimensions the curl of a vector is a vector; in two dimensions
    only the out of plane component survives, so it is treated as a scalar.
    This is the operator by which the velocity acts on the microrotation
    equation, and it is the formal adjoint of grad_perp on H^1_0.
    """
    return u[1].dx(0) - u[0].dx(1)


# -----------------------------------------------------------------------------
# Poisson
# -----------------------------------------------------------------------------

def poisson_exact(mesh):
    """Exact solution for the Poisson problem, vanishing on the boundary."""
    x = ufl.SpatialCoordinate(mesh)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


def poisson_source(mesh):
    """Forcing f with -div(grad(u)) = f for u = poisson_exact(mesh)."""
    return -ufl.div(ufl.grad(poisson_exact(mesh)))


# -----------------------------------------------------------------------------
# Shared exact fields
# -----------------------------------------------------------------------------

def _stream_function(mesh):
    """Stream function generating the exact velocity.

    Both the function and its gradient vanish on the boundary of the unit
    square, which is what makes the velocity satisfy the no slip condition.
    """
    x = ufl.SpatialCoordinate(mesh)
    return ufl.sin(ufl.pi * x[0])**2 * ufl.sin(ufl.pi * x[1])**2


def velocity_exact(mesh):
    """Exact velocity, divergence free by construction.

    Defined as the perpendicular gradient of a stream function, so that

        div grad_perp psi = d_1 d_2 psi - d_2 d_1 psi = 0

    identically. The incompressibility constraint is therefore satisfied
    exactly rather than approximately, and the manufactured problem is a
    genuine instance of the system rather than a perturbation of one.
    """
    return grad_perp(_stream_function(mesh))


def pressure_exact(mesh):
    """Exact pressure, with zero mean over the unit square.

    Since the integral of cos(pi s) over (0,1) vanishes, the double integral
    factorises to zero, so this lies in L^2_0 and is the unique representative
    of its equivalence class.
    """
    x = ufl.SpatialCoordinate(mesh)
    return ufl.cos(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])


def microrotation_exact(mesh):
    """Exact microrotation, vanishing on the boundary."""
    x = ufl.SpatialCoordinate(mesh)
    return ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])


# -----------------------------------------------------------------------------
# Stokes
# -----------------------------------------------------------------------------

# Retained under their original names so that the Stage 2 tests keep working.
stokes_velocity_exact = velocity_exact
stokes_pressure_exact = pressure_exact


def stokes_source(mesh, nu=1.0):
    """Forcing f with -nu*div(grad(u)) + grad(p) = f for the exact fields."""
    u = velocity_exact(mesh)
    p = pressure_exact(mesh)
    return -nu * ufl.div(ufl.grad(u)) + ufl.grad(p)


# -----------------------------------------------------------------------------
# Micropolar
# -----------------------------------------------------------------------------

def micropolar_momentum_source(mesh, nu, nu_r):
    """Forcing f for the momentum equation,

        -(nu + nu_r) div(grad(u)) + grad(p) - 2*nu_r*grad_perp(w) = f.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    nu : float
        Kinematic viscosity.
    nu_r : float
        Vortex viscosity.

    Returns
    -------
    ufl expression, vector valued
    """
    u = velocity_exact(mesh)
    p = pressure_exact(mesh)
    w = microrotation_exact(mesh)

    return (
        -(nu + nu_r) * ufl.div(ufl.grad(u))
        + ufl.grad(p)
        - 2.0 * nu_r * grad_perp(w)
    )


def micropolar_rotation_source(mesh, nu_r, c):
    """Forcing g for the microrotation equation,

        -c div(grad(w)) + 4*nu_r*w - 2*nu_r*curl(u) = g.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    nu_r : float
        Vortex viscosity.
    c : float
        Angular viscosity, the combination c_a + c_d.

    Returns
    -------
    ufl expression, scalar
    """
    u = velocity_exact(mesh)
    w = microrotation_exact(mesh)

    return (
        -c * ufl.div(ufl.grad(w))
        + 4.0 * nu_r * w
        - 2.0 * nu_r * curl_2d(u)
    )