# micropolar-fem

Mixed finite element approximation of coupled Stokes-type systems, with the
stationary micropolar fluid model as the target problem.

This repository contains both the mathematical analysis and the numerical
implementation. The analysis is developed from first principles in `docs/notes/`;
the solver is built and verified in `src/` and `tests/`.

**Author:** M. Javed Bukhari
**Status:** in progress — theory notes underway, Stokes baseline next

---

## Scope

The micropolar (Eringen) model describes a fluid whose suspended microstructure
carries independent rotational degrees of freedom. It is used for blood,
ferrofluids, liquid crystals, and lubricants with additives. In two dimensions,
the stationary linearised system reads

    -(nu + nu_r) Laplacian(u) + grad(p) = 2 nu_r curl_perp(w) + f
    div(u) = 0
    -(c_a + c_d) Laplacian(w) + 4 nu_r w = 2 nu_r curl(u) + g

with homogeneous Dirichlet conditions on velocity `u` and microrotation `w`.

The project has three components:

1. **Analysis.** Weak formulation, function space selection, well-posedness via
   Lax–Milgram and the Babuška–Brezzi theory.
2. **Discretisation.** Taylor–Hood `P2/P1` for velocity–pressure, continuous `P2`
   for microrotation; discrete inf-sup stability and a priori error estimates.
3. **Verification and parameter study.** Convergence rates confirmed by the
   method of manufactured solutions; behaviour of the error constants as the
   vortex viscosity `nu_r` varies, including the Stokes limit `nu_r -> 0`.

---

## Repository layout

    micropolar-fem/
    ├── README.md
    ├── .gitignore
    ├── docs/
    │   └── notes/            theory notes, one LaTeX file per topic
    │       ├── lesson01.tex  weak formulations, boundedness, coercivity
    │       └── macros.tex    shared LaTeX macros
    ├── src/                  solver modules
    ├── tests/                automated verification, including convergence rates
    └── scripts/              parameter studies and figure generation

---

## Theory notes

| File | Contents | Status |
|------|----------|--------|
| `lesson01.tex` | Weak formulation of scalar elliptic problems in 1D and 2D; boundedness via Cauchy–Schwarz; coercivity via Poincaré; the reaction–diffusion case | in progress |

---

## Verification

Convergence rates are checked automatically against the method of manufactured
solutions. Expected rates for the mixed `P2/P1/P2` discretisation:

| Quantity | Norm | Expected rate |
|----------|------|---------------|
| velocity | L2   | h^3 |
| velocity | H1   | h^2 |
| pressure | L2   | h^2 |
| microrotation | L2 | h^3 |
| microrotation | H1 | h^2 |

Results will be reported here as they are obtained.

---

## References

- Boffi, Brezzi & Fortin, *Mixed Finite Element Methods and Applications*
- Brezis, *Functional Analysis, Sobolev Spaces and Partial Differential Equations*
- Elman, Silvester & Wathen, *Finite Elements and Fast Iterative Solvers*
- Łukaszewicz, *Micropolar Fluids: Theory and Applications*
