# Monte Carlo verification

A reproducible numerical study of the paper's predictions for several size vectors, among them three and four real 2×2 factors, where Conjecture E is tested at `q = 2` and `q = 4`. It is numerical evidence, not a proof.

Start with `REPORT.md` for the methods, results and their interpretation. Per-seed records and pooled results are under `results/`, and all figures, in PNG and PDF with captions, are under `plots/`. `notes/` contains the estimator derivations and the statistical audit.

## Normalization

`Z(t) = E exp(−t ‖A_N ⋯ A_1‖_F²)`. Real entries have law `N(0,1)`; complex entries are `(X+iY)/√2` with independent standard real normals. Small-ball thresholds refer to the squared Frobenius or operator norm, as labelled. A loss with an extra factor 1/2 requires `Z(t/2)`.

## Environment and reproduction

The recorded results were produced with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, mpmath 1.3.0, SymPy 1.14.0 and Matplotlib 3.10.8 (`requirements.txt`). Random draws use NumPy `default_rng`/PCG64, with an explicit seed for every job, and parallelism is across independent jobs. Set BLAS threads to one:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

Re-analysis of the recorded observations takes minutes:

```bash
python audit_checks.py --production-only
python summarize_results.py
python make_plots.py
python check_reproducibility.py    # replays three production seeds and checks the log-domain kernel
```

The complete experiment takes several hours on four cores:

```bash
python analytic_reference.py
python run_experiments.py --mode direct --workers 4
python run_experiments.py --mode vr --workers 4
python run_experiments.py --mode vr --config config_ridge.json --workers 4
python run_experiments.py --mode efficiency --workers 4
python stratified_experiment.py
python run_cdf_experiments.py
python audit_checks.py
python summarize_results.py
python make_plots.py
```

`run_experiments.py` keeps existing per-job files, so interrupted runs can be resumed. To rerun from scratch, work in a fresh copy with the `results/` directory renamed, so that the recorded results are kept.

## Code map

| File | Role |
|---|---|
| `direct_mc.py` | Gaussian matrices sampled directly; direct Laplace and small-ball estimates; exact Gaussian conditioning on the last layer |
| `vr_estimators.py` | Bartlett reduction; exact integration of off-diagonal and endpoint variables; full-support importance proposals over logarithmic scales; fixed-allocation strata |
| `cdf_estimators.py` | Exact exponential tilt of the conditional three-Gamma distribution, with randomized correction, for the distribution function |
| `analytic_reference.py` | Separately implemented exact benchmarks and Mellin-residue polynomials, used only for comparison and never in importance weights |
| `run_experiments.py`, `config*.json` | Main job grid and seeds |
| `stratified_experiment.py` | IID mixture against fixed-allocation strata |
| `run_cdf_experiments.py` | Small-ball distribution function and its calibration |
| `audit_checks.py` | Normalization checks, aggregation of replicates and fit sensitivity |
| `summarize_results.py` | Comparisons, pooled uncertainties and CSV tables |
| `check_reproducibility.py` | Seed replay and extended-precision check of the kernel |
| `make_plots.py` | Figures from the recorded observations |

No coefficient predicted by the paper enters a sampling density, likelihood ratio, conditional kernel or distribution-function multiplier.

## Interpretation

- `L = log t` (or `log(1/ε)` for distribution functions), with natural logarithms. At `L = 1280` double precision cannot represent `t`, so the code works with `L` and scaled contributions throughout.
- Empirical standard errors and effective sample sizes cannot certify the absence of missed rare contributions. Ordinary sampling fails at the deeper thresholds and is kept only as a diagnostic.
- Reported 95% intervals are pointwise sampling intervals, not simultaneous bands, and they do not bound finite-`L` corrections.
- Fitted intercepts and exponents are unconstrained. A good fit over a finite range does not prove an asymptotic limit.
- The fixed-allocation variance is the sum of within-stratum variances.
- Direct Bernoulli estimates use Wilson intervals, with positive upper bounds when no events are observed.
