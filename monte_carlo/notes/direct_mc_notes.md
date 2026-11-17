# Independent direct and final-layer conditional sampling

`direct_mc.py` samples the original matrix entries. It does not invoke the paper's Mellin formulas, Bartlett representation, shell data, asymptotic exponents, or conjectured coefficients. This makes it an independent finite-range check of the variance-reduced Bartlett implementation.

## Normalization and Gaussian integration

The implemented target is the paper's

\[
Z(t)=\mathbb E\exp[-t\|A_N\cdots A_1\|_F^2].
\]

Real entries are independent `N(0,1)`. Complex entries are `(X+iY)/sqrt(2)` with independent real standard normals, so their density is `pi^(-1) exp(-|z|^2)` and `E|z|^2=1`. There is **no factor 1/2 in the implemented loss**. Bayesian evidence for loss `||P||^2/2` would require evaluating this function at `t/2`.

Write `B=A_(N-1)...A_1` and let `m=d_N`. Integrating the final matrix, which is an existing layer of the original model, gives

\[
\begin{aligned}
\mathbb E[ e^{-t\|A_NB\|_F^2}\mid B]
&=\det(I+2tB^TB)^{-m/2} &&\text{(real)},\\
&=\det(I+tB^*B)^{-m} &&\text{(complex)}.
\end{aligned}
\]

The code evaluates these through the singular values of `B`, avoiding cancellation in determinants of ill-conditioned Gram matrices. Algebraically zero singular directions associated with an earlier bottleneck width are discarded using the exact rank `min(widths[:-1])`. For the requested families this rank is two.

The direct calculation draws the final matrix and uses `exp(-t*sum(abs(product)**2))`. The two calculations share the preceding layers. Their paired difference is therefore a direct mean-zero diagnostic with the correct paired standard error. Separate calls with separately recorded seeds supply independent experiments.

## CDFs

The direct sampler records both

- Frobenius CDF `Pr(||P||_F^2 <= epsilon)`, and
- operator-norm CDF `Pr(||P||_op^2 <= epsilon)`.

These are distinct quantities. In particular, a real Frobenius constant must not be used to normalize an operator-norm probability.

For one positive prefix singular value, the final-layer conditional Frobenius law is an elementary gamma distribution. For real `m=2` and two positive squared singular values `lambda_1,lambda_2`, it is the sum of two independent exponential variables with means `a=2 lambda_1`, `b=2 lambda_2`:

\[
F(\epsilon\mid B)=1-\frac{a e^{-\epsilon/a}-b e^{-\epsilon/b}}{a-b}.
\]

The code uses this exact distributional formula, with its gamma limit at equal scales. A convergent 14-term small-argument series and stable hyperbolic reformulation avoid subtraction loss. Those numerical formulas were compared against 150-digit arithmetic on 64 extreme cases, including ratios as small as `1e-100` and almost equal scales. The maximum observed relative error was `2.201e-13`. This is a finite numerical accuracy check, not a uniform floating-point theorem.

For real final `m=3` and general rank-two complex laws the module deliberately does **not** substitute uncalibrated numerical CDF quadrature. Their conditional Laplace estimator is still exact, and their CDFs are sampled directly.

## API and reproduction

```python
from direct_mc import run_direct_case
result = run_direct_case(
    widths=[2,2,2,2,2], beta=1,
    t_values=[.01,.1,1,10,100,1000],
    eps_values=[.001,.01,.1,1],
    n=1_000_000, seed=873501, batch_size=65536)
```

`beta=1` means real and `beta=2` complex. The return value is JSON serializable. It includes one summary block per estimator, all recorded seeds, sample size, runtime and component timing. All scales within a call share samples; the full covariance matrix of the estimated means is included. Independent seed calls must not be treated as paired.

Each summary contains the empirical mean, sample variance, standard error, conventional normal 95% interval, nonzero contribution count, sums, squared sums, and covariance of the mean. Positive-contribution ESS is

\[
N_{\mathrm{contrib}}=\frac{(\sum_i Y_i)^2}{\sum_iY_i^2}.
\]

This measures concentration of the contributions; it is neither MCMC ESS nor a guarantee that an unobserved tail has been sampled. Direct CDFs also receive Wilson intervals, including a nonzero upper endpoint after zero events. A zero empirical standard error when every Laplace contribution underflows is **not** evidence of a vanishing expectation. Such points have `all_zero=true` and require separate treatment.

The covariance accumulator uses centered Chan updates to avoid subtracting nearly equal uncentered moments. Reported timing components separate prefix generation, final-layer conditional Laplace, direct final-layer generation/Laplace, CDF/spectral calculations, and summary accumulation. These are costs measured within one common-prefix run, not separately timed standalone algorithms.

Reproduce the validation outputs from the audit directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 direct_mc.py --validate results/direct_validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 direct_mc.py --validate-independent results/direct_independent_seed_validation.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 direct_mc.py --validate-cdf results/direct_cdf_roundoff_validation.json
```

The single-matrix exact references are `(1+2t)^(-2)` for a real `2x2` matrix and `(1+t)^(-4)` for a complex `2x2` matrix. The conditional estimator reproduces these to at most `1.11e-16` absolute difference in the recorded validation. These elementary gamma-law checks independently fix the factors of two.

Five 200,000-sample checks used seeds `873101`, `873102`, `873201`, `873202`, `873203`. The paired real `(2,2,3)` run has a largest absolute standardized difference of `2.93` among its correlated scale points; this fluctuation is retained rather than discarded. Separate 400,000-sample comparisons use direct/conditional seed pairs `(873401,873402)`, `(873403,873404)`, `(873405,873406)` for real `(2,2,2)`, real `(2,2,3)`, and complex `(2,2,2,2,2)`, respectively. Their largest absolute standardized difference is `1.833`. These checks support consistency of the finite-range estimators; they do not certify an asymptotic coefficient.

## Bias and scope

There is no self-normalized importance weighting, fitted asymptotic correction, finite Fourier cutoff, or acceptance/rejection truncation in this module. In exact arithmetic the original-entry and conditional estimators are unbiased for their finite-scale targets. Floating-point matrix multiplication/SVD and the stated stable CDF evaluation remain numerical approximations; direct `exp(-t*g)` can underflow at large `t`, and rare-event variance can make empirical errors misleading. The high-`q` asymptotic investigation must therefore use and independently validate the separate variance-reduced estimator. Successful agreement here is not a proof of Theorems A–D or Conjecture E.
