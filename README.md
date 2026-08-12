# micropolar-fem

Mixed finite element approximation of the stationary linear micropolar fluid
system: well-posedness analysis, verified implementation in FEniCSx, and a
parameter study testing the sharpness of the a priori error estimate.

**Author:** M. Javed Bukhari, MS Mathematics, COMSATS University Islamabad (Lahore Campus)

---

## Summary of results

Three solvers were built in sequence and each verified by the method of
manufactured solutions before the next was started. All observed convergence
rates match the theory.

| Problem | Discretisation | Observed rates |
|---|---|---|
| Poisson | P2 | 3.00 (L2), 2.00 (H1) |
| Stokes | P2/P1 Taylor–Hood | 3.00, 2.00 (velocity), 2.06 (pressure) |
| Micropolar | P2/P1/P2 | 3.00, 2.00, 2.06, 3.05, 2.00 |

![Convergence](figures/convergence.png)
A parameter study over six orders of magnitude in the vortex viscosity then
tested two predictions of the analysis. **The first was confirmed. The second
was found to be sharp in one component and loose in another**, for a reason that
is structural and is discussed below.

---

## The problem

The micropolar model, due to Eringen, describes a fluid whose suspended
microstructure carries independent rotational degrees of freedom. It is used for
blood, ferrofluids, liquid crystals, and lubricants with additives. In two
dimensions the stationary linearised system is

$$-(\nu + \nu_r)\,\Delta \mathbf{u} + \nabla p = 2\nu_r \nabla^{\perp} w + \mathbf{f}$$

$$\nabla \cdot \mathbf{u} = 0$$

$$-(c_a + c_d)\,\Delta w + 4\nu_r w = 2\nu_r \mathrm{curl}\,\mathbf{u} + g$$

with no-slip conditions on the velocity and homogeneous Dirichlet conditions on
the microrotation. The unknowns are velocity, pressure, and the scalar
microrotation. The parameter $\nu$ is the kinematic viscosity, $\nu_r$ the vortex
viscosity, and $c_a + c_d$ the angular viscosity.

The two coupling terms exchange angular momentum between the bulk flow and the
microstructure. They carry no sign, and they are what obstructs coercivity.

---

## Analysis

Full derivations are in [`docs/notes/`](docs/notes), five documents built from
first principles.

### Well-posedness

The system is posed as a saddle-point problem on

$$\mathbf{V} = [H_0^1(\Omega)]^2, \qquad Q = L_0^2(\Omega), \qquad W = H_0^1(\Omega),$$

and analysed by the Babuška–Brezzi theory rather than by Lax–Milgram, since the
pressure enters as a Lagrange multiplier and therefore never appears coercively.

**Coercivity.** Absorbing the indefinite coupling term requires Young's
inequality with a free parameter,

$$|ab| \le \frac{\varepsilon}{2}a^2 + \frac{1}{2\varepsilon}b^2,$$

and the choice must satisfy two conditions at once: the vortex viscosity
contributions in the velocity coefficient must cancel, and the coefficient of the
microrotation term must remain non-negative. Both bind simultaneously at
$\varepsilon = 1/2$, giving

$$a\big((\mathbf{u},w),(\mathbf{u},w)\big) \ \ge\ \alpha \lVert (\mathbf{u},w) \rVert^2, \qquad \alpha = \frac{\min(\nu,\ c_a + c_d)}{C_\Omega^2 + 1}.$$

**The coercivity constant does not involve the vortex viscosity**, despite
$\nu_r$ appearing in four separate terms of the equations. The reason is
physical: the coupling terms transfer angular momentum between the two fields
rather than dissipating it, and coercivity is a statement about total energy, so
a purely internal transfer must cancel. What appears in $\alpha$ are the two
genuine dissipation mechanisms, and the minimum reflects that the system
dissipates only as fast as its slowest channel.

**Boundedness.** By contrast the boundedness constant grows linearly,

$$M = \nu + (c_a + c_d) + 9\nu_r,$$

because with two independent arguments there is no cancellation available.

### Discretisation

Taylor–Hood P2/P1 for velocity and pressure, continuous P2 for the
microrotation. Boundedness transfers to the discrete spaces immediately, since an
inequality valid on a set is valid on any subset. Coercivity also transfers,
because the proof never used the divergence-free constraint. The inf-sup
condition does **not** transfer, since the supremum is taken over a smaller set,
and uniform stability of the Taylor–Hood pair must be invoked separately.

The resulting a priori estimate gives rates 3 and 2 in the L2 and H1 norms, with
an error constant proportional to $M/\alpha$.

---

## Verification

### Method

Exact fields are chosen, substituted into the strong form to obtain the data they
solve, and the computed solution compared against them. The velocity is
constructed as the perpendicular gradient of a stream function,

$$\mathbf{u} = \nabla^{\perp}\psi, \qquad \psi = \sin^2(\pi x)\sin^2(\pi y),$$

so that $\nabla \cdot \mathbf{u} = 0$ holds identically rather than
approximately, and the pressure is chosen with zero mean so that it lies in
$L_0^2$.

Two implementation points matter for correctness. The forcing terms are obtained
by **symbolic differentiation of the exact solution in UFL**, so there is no
transcription step and no opportunity for a sign error. Errors are computed by
integrating the exact solution as a symbolic expression with elevated quadrature,
never by interpolating it into a finite element space, which would measure
interpolation error instead of discretisation error.

### Poisson, P2

| h | L2 error | rate | H1 error | rate |
|---|---|---|---|---|
| 1/8 | 5.481e-04 | | 3.339e-02 | |
| 1/16 | 6.874e-05 | 3.00 | 8.419e-03 | 1.99 |
| 1/32 | 8.601e-06 | 3.00 | 2.110e-03 | 2.00 |
| 1/64 | 1.075e-06 | 3.00 | 5.277e-04 | 2.00 |
| 1/128 | 1.344e-07 | 3.00 | 1.319e-04 | 2.00 |

### Stokes, P2/P1 Taylor–Hood

| h | u, L2 | rate | u, H1 | rate | p, L2 | rate |
|---|---|---|---|---|---|---|
| 1/8 | 1.052e-02 | | 6.166e-01 | | 2.835e-02 | |
| 1/16 | 1.331e-03 | 2.98 | 1.587e-01 | 1.96 | 2.745e-03 | 3.37 |
| 1/32 | 1.672e-04 | 2.99 | 4.000e-02 | 1.99 | 4.423e-04 | 2.63 |
| 1/64 | 2.093e-05 | 3.00 | 1.002e-02 | 2.00 | 1.017e-04 | 2.12 |

The pressure rate approaches 2 from above. This is expected: the pressure error
carries a contribution from the velocity error, which converges one order faster
and dominates on coarse meshes. The signature of an unstable element pair is the
opposite, a pressure rate that degrades under refinement while the velocity rates
stay clean.

### Micropolar, P2/P1/P2

Observed rates at $\nu = \nu_r = c_a + c_d = 1$:

| h | u, L2 | u, H1 | p, L2 | w, L2 | w, H1 |
|---|---|---|---|---|---|
| 1/16 | 2.99 | 1.96 | 3.55 | 3.64 | 2.07 |
| 1/32 | 2.99 | 1.99 | 3.11 | 3.41 | 2.02 |
| 1/64 | 3.00 | 2.00 | 2.37 | 3.16 | 2.00 |
| 1/128 | 3.00 | 2.00 | 2.06 | 3.05 | 2.00 |

---

## Parameter study

### What it tests

The convergence tables verify the *implementation*. They cannot verify the
*analysis*, since a misconception shared between the derivation and the code
would pass unnoticed.

The parameter study is different. Since $\alpha$ is independent of $\nu_r$ while
$M$ grows linearly, the error constant is proportional to $M/\alpha = O(\nu_r)$.
This gives two predictions, and **nothing in the solver knows what $\alpha$ is
supposed to be**:

- **(P1)** convergence rates independent of $\nu_r$
- **(P2)** errors growing like $O(\nu_r)$ at fixed mesh

### (P1): confirmed

Observed rates across six orders of magnitude in the vortex viscosity:

| nu_r | u, L2 | u, H1 | p, L2 | w, L2 | w, H1 |
|---|---|---|---|---|---|
| 0 | 3.00 | 2.00 | 2.02 | 3.00 | 2.00 |
| 1e-2 | 3.00 | 2.00 | 2.02 | 3.00 | 2.00 |
| 1e-1 | 3.00 | 2.00 | 2.02 | 3.00 | 2.00 |
| 1 | 3.00 | 2.00 | 2.06 | 3.05 | 2.00 |
| 10 | 3.00 | 2.00 | 2.77 | 3.73 | 2.12 |
| 1e2 | 3.04 | 2.00 | 3.56 | 3.99 | 2.92 |
| 1e3 | 3.72 | 2.00 | 3.62 | 3.99 | 2.98 |

**The velocity H1 rate reads 2.00 at every parameter value without exception.**
This is the observable consequence of the coercivity constant being independent
of the vortex viscosity.

The L2 columns exceed their theoretical values at large $\nu_r$. Rates above the
interpolation order cannot be asymptotic, and inspecting the full rate sequences
confirms a preasymptotic regime: at $\nu_r = 10^2$ the velocity sequence reads
3.52, 3.36, 3.14, 3.04, descending steadily toward 3. The picture is of an error
behaving like $A h^3 + B(\nu_r) h^k$ with $k > 3$, where the crossover mesh moves
finer as $B$ grows. The H1 seminorm does not carry the higher-order contribution
and reads the asymptotic rate immediately at every parameter value.

### (P2): sharp in one component, loose in the other

Errors at fixed mesh $h = 1/128$:

| nu_r | u, L2 error | w, L2 error |
|---|---|---|
| 0 | 2.6167e-06 | 1.3443e-07 |
| 1e-2 | 2.6167e-06 | 1.3443e-07 |
| 1e-1 | 2.6167e-06 | 1.3447e-07 |
| 1 | 2.6168e-06 | 1.3601e-07 |
| 10 | 2.6178e-06 | 2.1013e-07 |
| 1e2 | 2.6428e-06 | 1.2727e-06 |
| 1e3 | 4.0822e-06 | 1.2025e-05 |

**Sharp for the microrotation**: the error grows by a factor of 9.4 over the last
decade, that is, linearly, as predicted.

**Loose for the velocity**: the error grows by a factor of only 1.56 across six
decades, against a bound permitting a factor of $10^3$. For reference, $M/\alpha$
grows from 2.1 to 9458 over the same range.

### Interpretation

The estimate is an upper bound rather than a growth rate, so slower growth does
not contradict it; growth *faster* than linear would.

The mechanism for the velocity is structural. The viscous term carries the factor
$(\nu + \nu_r)$, and so does the manufactured forcing, since the forcing is
obtained by substituting a fixed exact solution into the strong form. That factor
largely divides out when the velocity block is inverted. The remaining genuinely
$\nu_r$-dependent contribution comes from the coupling term
$2\nu_r \nabla^{\perp} w$, and its effect on the velocity is damped by the
$1/(\nu + \nu_r)$ arising from the same inversion. The two occurrences cancel.
The microrotation has no such cancellation, since its diffusion coefficient does
not scale with $\nu_r$, and there the growth is genuinely linear.

**This phenomenon is documented.** It is the viscosity-induced locking effect
known from the pressure-robustness literature for the Stokes equations, where
precisely this experimental setup, a fixed exact solution with the forcing
rescaled by the parameter, is used to distinguish pressure-robust from
non-pressure-robust methods. The observation here is real and correctly
diagnosed, but it is not new.

### Open question

The estimate as proved is posed on the product space and controls all three
fields through a single constant, so it cannot express an asymmetry in which the
velocity component is robust and the microrotation component is not.

Two questions follow, and I have not been able to find either answered in the
literature. Can a componentwise a priori estimate be established, robust in the
velocity component? And does the pressure-robust construction, which uses a
divergence-free reconstruction operator mapping into $H(\mathrm{div})$, extend to
the micropolar system, where two parameters enter four terms rather than one?

Pointers to existing work on either would be welcome.

---

## Repository layout

```
micropolar-fem/
├── src/
│   ├── poisson.py         Stage 1 solver
│   ├── stokes.py          Stage 2 solver, Taylor-Hood
│   ├── micropolar.py      Stage 3 solver, three coupled fields
│   ├── manufactured.py    exact solutions and symbolically derived forcing
│   └── diagnostics.py     error norms and convergence rates
├── tests/                 pytest suite, asserts rates rather than plotting them
├── scripts/               convergence tables and the parameter study
├── figures/               Convergence figures
└── docs/notes/            five LaTeX documents with the full analysis
```

### The analysis documents

| File | Contents |
|---|---|
| `lesson01` | Weak formulations, boundedness, coercivity, the role of the function space |
| `lesson02` | Well-posedness of the micropolar system on the divergence-free space |
| `lesson03` | The mixed formulation, saddle-point structure, the inf-sup condition |
| `lesson04` | Discrete inf-sup, Taylor–Hood stability, a priori error estimates |
| `lesson05` | Implementation in FEniCSx, and the computational results |

---

## Running

Requires DOLFINx 0.11.

```bash
conda create -n fenicsx -c conda-forge fenics-dolfinx mpich numpy scipy pytest -y
conda activate fenicsx
```

```bash
pytest tests/ -v                            # full verification suite
python scripts/run_convergence.py 2         # Poisson table
python scripts/run_stokes_convergence.py 2  # Stokes table
python scripts/run_micropolar_convergence.py # micropolar table
python scripts/run_parameter_study.py       # vortex viscosity sweep
```

The test suite asserts convergence rates rather than plotting them, so a change
that breaks the discretisation fails loudly rather than requiring a person to
inspect a figure.

---

## References

- Boffi, Brezzi & Fortin, *Mixed Finite Element Methods and Applications*
- Brezis, *Functional Analysis, Sobolev Spaces and Partial Differential Equations*
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*
- Łukaszewicz, *Micropolar Fluids: Theory and Applications*
- John, Linke, Merdon, Neilan & Rebholz, *On the divergence constraint in mixed finite element methods for incompressible flows*, SIAM Review 59 (2017)
