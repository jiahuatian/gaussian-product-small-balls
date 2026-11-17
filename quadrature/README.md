# Deterministic quadrature for three 2×2 factors

`bartlett_quadrature_2222.py` computes `Z(t) = E exp(−t ‖A_3 A_2 A_1‖_F²)` for 2×2 Gaussian factors to about twelve digits, for `t` up to `10^16`. Real entries test Conjecture E of the paper at `q = 2`. Complex entries, where the constant is Theorem A, are the control.

## Exact reduction

In Bartlett coordinates the three factors are upper triangular, with diagonal entries `x_j`, `z_j` and off-diagonal entries `y_j`, and

```
‖P‖_F² = (x3 x2 x1)² + (z3 z2 z1)² + (x3 x2 y1 + x3 y2 z1 + y3 z2 z1)².
```

For real entries `x_j ~ χ_2`, `z_j ~ χ_1` and `y_j ~ N(0,1)`. The three Gaussian entries, `x1` and `z3` integrate out exactly. With `X = x3² x2²`, `Y = x3² z1²` and `W = z1² z2²`,

```
real:     Z(t) = E[(1+2tX)^(−1) (1+2tW)^(−1/2) (1+2t(X+Y+W))^(−1/2)],
complex:  Z(t) = E[(1+tX)^(−2)  (1+tW)^(−1)    (1+t(X+Y+W))^(−1)].
```

The joint density of `(X, Y, W)` is explicit:

```
real:     h(X,Y,W) = W^(−1/2) K_0( √((1+W/Y)(X+Y)) ) / (4π Y),
complex:  h(X,Y,W) = (2X/Y) K_0( 2√((1+W/Y)(X+Y)) ).
```

So `Z(t)` is a three-dimensional integral. It is evaluated by composite Gauss–Legendre quadrature in the variables `log X`, `log Y`, `log W`, over ranges that extend about `log t` below the small-ball scale. No coefficient predicted by the paper enters the computation.

## Prediction and extraction

Conjecture E gives `t^(3/2) Z_R(t) = A L² + B L + C + o(1)` with `L = log t` and

```
A = Γ(5/2) · √(2π)/48 = π√2/64 = 0.0694200459…
```

Theorem A gives `t³ Z_C(t) = A L² + …` with `A = κ_F Γ(4) = 1/2`.

The script computes `t^λ Z(t)` at `t = 10^k`. The local value of `A` is the second difference in `L`. It converges like `t^(−1/2)` for real entries and like `t^(−1)` for complex entries, and is extrapolated accordingly.

## Results (`results/`)

| file | content |
|---|---|
| `validate.txt` | densities integrate to 1 (to `1e-10`); quadrature against direct Monte Carlo of Gaussian matrix products at `t = 0.5, 2, 10` |
| `validate_large.txt` | the same with larger independent samples (`4·10^6` products, and `10^7` draws of the reduced expectation) |
| `sweep.txt` | `t^λ Z(t)` for `t = 10^4 … 10^14` (complex) and `10^6 … 10^16` (real), with fits |
| `converge.txt` | grid refinement at the largest `t`: relative changes below `1e-12` |
| `summary.txt` | extrapolated coefficients and the finite-scale ratios implied by the fitted subleading terms |
| `crosscheck.txt` | the ratio `t^(3/2) Z(t) / (A_pred L²)` at the eight scales `L = 10 … 1280` of the Monte Carlo study, by direct quadrature for `L ≤ 80` and from the fitted polynomial beyond, next to the Monte Carlo values |

Extrapolated coefficients:

| field | extrapolated `A` | predicted `A` | relative difference |
|---|---|---|---|
| complex (Theorem A) | 0.500000071 | 1/2 | `1.4e-7` |
| real (Conjecture E) | 0.069420227 | `π√2/64 = 0.069420046` | `2.6e-6` |

So `κ_R(2,2,2,2) = A/Γ(5/2) = 0.0522216`, against `√(2π)/48 = 0.0522214`.

## Running

```bash
python bartlett_quadrature_2222.py all         # about 6 minutes on one core
python bartlett_quadrature_2222.py summary     # re-analyse results/sweep.txt only
python bartlett_quadrature_2222.py crosscheck  # compare with ../monte_carlo/results
```
