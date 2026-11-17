# Small-ball asymptotics of Gaussian matrix products: numerical checks

Code and results accompanying the paper

> J. Tian, *Small-ball asymptotics of Gaussian matrix products: Open Fermi shells and the Grassmannian* (2026).

For a product `P = A_N ⋯ A_1` of independent standard Gaussian matrices of fixed sizes `d = (d_0, …, d_N)`, the paper shows that

```
P(‖P‖_F² ≤ ε) ~ κ ε^λ (log 1/ε)^q,        E exp(−t ‖P‖_F²) ~ A t^(−λ) (log t)^q,    A = Γ(λ+1) κ,
```

with `λ = C` for complex and `λ = C/2` for real entries. It determines `C`, `q` and, for complex entries, `κ` and the full logarithmic polynomial (Theorems A and B). For real entries it proves the exponents (Theorem C), evaluates `κ` for every shell with at most one logarithm (Theorem D), and conjectures `κ = B·K_1(e)` in general (Conjecture E). This repository tests these statements numerically with two independent methods.

- **`quadrature/`**: a deterministic, high-precision computation for three 2×2 factors, `d = (2,2,2,2)`. An exact reduction in Bartlett coordinates turns the Laplace transform into a three-dimensional integral with an explicit Bessel kernel, which is evaluated to about 12 digits for `t` up to `10^16`.
- **`monte_carlo/`**: variance-reduced Monte Carlo for several size vectors, including four factors, `d = (2,2,2,2,2)`. It uses direct sampling of Gaussian matrices, exact Bartlett reduction with full-support importance sampling over logarithmic scales, and a separate estimator of the small-ball distribution function.

## Results

| Statement in the paper | Case | Method | Result |
|---|---|---|---|
| Theorem A (complex constant) | complex `(2,2,2,2)`, `A = 1/2` | quadrature | extrapolated `A = 0.500000071` (relative difference `1.4e-7`) |
| Theorem B (full complex polynomial) | complex `(2,2,2,2,2)`, `A = 1/12` | Monte Carlo | all 8 Laplace points agree with `P(L) = (L−3γ_E)⁴/12 + ζ(3)(L−3γ_E) + 25π⁴/144` within 1.51 standard errors |
| Theorem D, Proposition 6.8 (real, `q ≤ 1`) | real `(2,2,2)`, `κ = π/6`; `(2,2,3)`, `κ = 1/16` | Monte Carlo | exact finite-`t` calibrations and deep-tail estimates agree |
| Theorem C (real exponent and logarithmic power) | real `(2,2,2,2)`, real `(2,2,2,2,2)` | Monte Carlo | free fits give `λ = 1.50000017 ± 0.0000055`, `q = 1.9991 ± 0.0042` and `λ = 1.50000496 ± 0.0000064`, `q = 4.0056 ± 0.0051` |
| Conjecture E, `q = 2` | real `(2,2,2,2)`, `κ = √(2π)/48` | quadrature | extrapolated `A/A_pred = 1 + 2.6e-6`, i.e. `κ = 0.0522216` against `√(2π)/48 = 0.0522214` |
| Conjecture E, `q = 2` | real `(2,2,2,2)` | Monte Carlo | corrected fit `A/A_pred = 1.000125 ± 0.000659` |
| Conjecture E, `q = 4` | real `(2,2,2,2,2)`, `κ = 1/576` | Monte Carlo | corrected fit `A/A_pred = 1.001039 ± 0.000919` |

The two methods agree where they overlap. The quadrature also gives the subleading real coefficients `B ≈ 0.3208` and `C ≈ −1.683` in `t^(3/2) Z(t) = A L² + B L + C + o(1)`, which the paper does not determine. With them the quadrature reproduces the Monte Carlo ratios `t^(3/2) Z(t) / (A_pred L²)` at all eight scales `L = log t = 10, 20, …, 1280` within the Monte Carlo errors; at `L = 1280` it gives `1.00360` and the Monte Carlo `1.00359 ± 0.00233` (`quadrature/results/crosscheck.txt`).

Numerical agreement over a finite range of scales supports the asymptotic statements; it does not prove them. In particular, Conjecture E remains a conjecture.

## Conventions

- `Z(t) = E exp(−t ‖A_N ⋯ A_1‖_F²)`. Real entries are `N(0,1)`; complex entries are `(X + iY)/√2` with independent standard real normals `X`, `Y`.
- `L = log t` (natural logarithm), or `L = log(1/ε)` for distribution functions.
- `F_F(ε) = P(‖P‖_F² ≤ ε)`. The Laplace coefficient is `A = Γ(λ+1) κ`.
- A loss with a factor one half, `H = ‖P‖_F²/2`, has partition function `Z(t/2)`.

## Layout

```
quadrature/
    bartlett_quadrature_2222.py   exact reduction and quadrature for d = (2,2,2,2)
    results/                      validation, large-t sweep, grid refinement, extrapolation
monte_carlo/
    *.py, config*.json            estimators, experiment drivers, analysis and plotting
    results/                      per-seed records and pooled results (JSON, CSV, logs)
    plots/                        figures (PDF and PNG) with captions
    notes/                        estimator derivations and statistical audit
    REPORT.md                     full report of the Monte Carlo study
```

## Reproduction

The quadrature takes about six minutes on one core:

```bash
python quadrature/bartlett_quadrature_2222.py all
```

Single steps are `validate`, `validate_large`, `sweep`, `converge`, `summary` and `crosscheck`.

Re-analyse the recorded Monte Carlo results and replay three production seeds (minutes):

```bash
cd monte_carlo
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python audit_checks.py --production-only
python summarize_results.py
python make_plots.py
python check_reproducibility.py
```

The complete Monte Carlo experiment (several hours on four cores) is described in `monte_carlo/README.md`.

## Requirements

Python 3.10 or later with NumPy, SciPy, mpmath, SymPy and Matplotlib (`requirements.txt`). The exact versions used for the recorded Monte Carlo results are in `monte_carlo/requirements.txt`. Bitwise replay of seeds is expected in that environment, not promised across library versions.

## Citation

If you use this code, please cite the paper.
