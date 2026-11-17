# Independent numerical and statistical audit

Date: 2026-11-17. Scope: estimator derivation, proposal densities, numerical stability, calibration, uncertainty and interpretation. This is a numerical audit, not mathematical certification.


## Findings

The numerical implementation is consistent with independently derived Gaussian conditioning and Bartlett formulas. Exact finite-parameter controls, two distinct sampling implementations and the full complex logarithmic polynomial agree within observed sampling uncertainty. The real q=2 and q=4 experiments show convergence compatible with the proposed constants after finite-log corrections. They do not prove Conjecture E, bound its remainder, or exclude arbitrarily close alternative limits.

| Claim | Numerical assessment |
|---|---|
| Real and complex Gaussian normalizations | Checked against the paper's definition; no missing factor of two found |
| Real 222 and 223 exact finite-t controls | Supported by independent quadrature and sampling |
| Complex 22222 full leading-power logarithmic polynomial | Supported over all eight tested log-scales; largest absolute standardized discrepancy 1.51 |
| Real q=2 and q=4 exponents and logarithmic degrees | Supported by unrestricted finite-range fits when subleading terms are included |
| Conjectured real q=2 and q=4 leading constants | Numerically supported at the tested resolution, conditional on stated extrapolation models |
| Exact asymptotic equivalence and unconditional high-precision identification of either real constant | Inconclusive; finite data provide no mathematical remainder bound |
| Zero-correction finite-range models | Contradicted by the data in several ranges; this is not a contradiction of the asymptotic claims |
| Narrow Schubert-only concentration as an adequate practical proposal | Inadequate in several efficiency runs; shifted and broad components materially improve coverage |

## Normalizations and independent analytic input

The target is `Z(t) = E exp(-t ||P||_F^2)`. The paper's separate Bayesian energy convention involving one half must not be substituted here. Real entries have variance one. Complex entries are `(X+iY)/sqrt(2)` and have mean squared modulus one.

The CDF coefficient kappa and Laplace coefficient A differ: `A = Gamma(lambda+1) kappa` for the leading regularly varying term. Thus real 2222 has Laplace coefficient `pi sqrt(2)/64`, and real 22222 has `sqrt(pi)/768`. Comparisons in this report use those Laplace coefficients. Their conversion is analytic input, not an independently simulated CDF limit.

In particular, dividing a large-t Laplace estimate by `Gamma(lambda+1)` yields a leading-coefficient proxy. At finite t that proxy is not a direct estimate of the rare-small-ball CDF, and its finite-log corrections need not equal the CDF corrections. Files labeled `CDF leading-coefficient proxy` in the initial stratification supplement must retain this qualification.

For a prefix matrix B and final Gaussian matrix with m rows, direct final-layer integration gives

`E[exp(-t ||G B||_F^2) | B] = det(I + (2/beta)t B*B)^(-beta m/2)`.

The SVD implementation matches this expression. Direct matrix sampling and this conditional estimator share prefixes within a run; paired differences must use their covariance. Independent-seed checks are also recorded. Direct Bernoulli CDF and conditional CDF results are separate from Laplace results.

For all production widths, the initial width is two and all later widths are at least two. A QR/Bartlett reduction therefore gives independent triangular factors with diagonal squares `a_j ~ Gamma(beta d_j/2, 2/beta)` and `b_j ~ Gamma(beta(d_j-1)/2, 2/beta)`. Conditional on diagonals, the squared scale of the product's off-diagonal Gaussian is

`V = sum_j product_{k>j} a_k product_{k<j} b_k`.

Integrating all off-diagonal Gaussians and the endpoint variables a_1 and b_N gives the implemented kernel

`h_t = (1+s t product a)^(-beta d_1/2) (1+s t product b)^(-beta(d_N-1)/2) (1+s t V)^(-beta/2)`, with `s=2/beta`.

This is finite-t Gaussian integration; neither the conjectured coefficient nor any asymptotic equivalence enters the kernel. Production does not permute arbitrary widths to invoke a minimum-width formula. Any future extension using permutation invariance must explicitly record that additional analytic input.

## Proposal and Jacobian audit

The proposals operate in logarithms of the positive Bartlett squares. The target log-coordinate density includes the Jacobian: for `z=log(x/scale)` it is `shape*z-exp(z)-log Gamma(shape)`. Sampling small Gamma shapes by `log Gamma(shape+1)+log(U)/shape` avoids underflow of the random positive variable itself.

For two independent uniform simplex points of dimension m-1, the prefix-order event used by the Schubert proposal has probability 1/m. Its conditioned density therefore includes the factor m. Transforming each simplex point and its positive total scale into log coordinates contributes `S^(m-1)` to the Jacobian. Both the `log(m)` and the two `-(m-1)log(S)` terms are present. Independent simulation of the order probability for m=1,2,3,4 agrees with 1/m; maximum deviation is 1.23 binomial standard errors.

The positive logistic total-scale density includes its truncation normalization. Numerical quadrature at centers 1,10,100 and width 3 gives integral one within floating-point quadrature error. The shifted simplex component uses `y=log(scale)-log(x)+4`, shifted total center `center+4m`, the same Jacobian, and no ordering factor. The two-factor ridge component is the product of a uniform log-coordinate and a shifted positive logistic coordinate; its implemented density matches its generator.

The main proposal uses 60% ordered scales, 30% shifted unordered scales, 5% broad tilted Gamma coordinates and 5% ordinary prior coordinates. Mixture labels are sampled independently for the principal runs. Every sample is weighted with the full mixture density, not its selected component density. The estimate is the ordinary mean of `h_t p/q`, not a self-normalized ratio. Since `q >= .05 p`, `p/q <= 20`; this guarantees finite contribution variance but does not itself guarantee useful relative accuracy for a rare target.

The optional fixed-allocation supplement uses actual component counts divided by total sample count in its balance-mixture denominator. Its correct estimated variance is `sum_k alpha_k^2 s_k^2/n_k`. Component labels and contiguous sample groups agree. The ordinary iid variance formula is not appropriate for that supplement. Its reported contribution ESS remains a concentration diagnostic, not a replacement for the stratified variance.

An earlier floating ordering tolerance and an exponential-density cap were removed during review. They are absent from the audited implementation. The narrow-window method is retained for comparison; poor finite-sample behavior is reported rather than hidden.

## Independent calibration

`audit_checks.py` independently derives positive one-dimensional integrals for real 222 and 223; it does not import the analytical-reference module for these checks. For real 222, one million ordinary Gamma-coordinate samples with seed 940012 agree with positive quadrature at t=0.1,1,10,100 within 0.65 standard errors. For real 223, positive quadrature and the exponential-integral closed form agree to better than 3e-15 relative error on that grid.

The actual shifted mixture was separately checked against these references at t=1,100,10000 using 300,000 samples per case and seeds recorded in `results/independent_is_calibration.json`. All six discrepancies are within 2.08 standard errors; five are within one. Importance weights average to one within 1.7 standard errors, and their observed maxima respect the bound of twenty. These checks calibrate finite-t implementation, not an asymptotic theorem.

The complex 22222 leading-power polynomial was also checked independently from the two-by-two determinant. Writing `L=log(t)` and `u=L-3 EulerGamma`, the result is

`t^3 Z(t) ~ u^4/12 + zeta(3) u + 25 pi^4/144`.

This polynomial includes all powers of log(t) at the leading power of t; it is not the exact finite-t transform. The empirical comparison uses the complete polynomial, not only `L^4/12`.

## Production uncertainty and convergence

The audited primary files comprise four families at eight log-scales, each with eight independent seeds and 250,000 samples per seed; real 223 has a five-scale ridge supplement with the same per-scale allocation. Direct runs use four seeds of 500,000 matrices in each of six families. All 704 main/direct/efficiency/ridge job seeds checked at this audit stage are distinct. Additional validation and stratification seeds are recorded separately.

Independent aggregation uses the larger of the between-seed standard error and the pooled within-seed standard error. Pointwise intervals use a Student t multiplier with seven degrees of freedom. This is a conservative diagnostic convention, not a rigorously guaranteed confidence procedure. Intervals are pointwise, not simultaneous across all plotted scales or selected models.

At L=1280, the following estimates are normalized by the proposed leading Laplace coefficient and by `t^(-lambda) L^q`:

| Family | Normalized estimate | Pointwise 95% half-width | Contribution ESS, out of 2 million |
|---|---:|---:|---:|
| Real 222 | 0.999782 | 0.002890 | about 825,000 |
| Real 223, ridge | 0.999364 | 0.001580 | about 1,332,000 |
| Real 2222, q=2 | 1.003585 | 0.002327 | about 685,000 |
| Real 22222, q=4 | 1.008331 | 0.002489 | about 628,000 |
| Complex 22222 | 0.992778 | 0.007091 | about 295,000 |

The complex ratio to the *full polynomial* at this scale is 0.998168. Across all eight scales, its largest standardized discrepancy is 1.51. The visibly different approach from above for real q=4 and below for complex q=4 makes leading-monomial-only assessment inappropriate.

In the primary L=1280 runs, the largest contribution within any individual seed is below 0.0003 of that seed's total, including the complex case. Seed dispersion and within-seed errors are of comparable order, though the complex largest-scale between-seed SE is about 1.78 times the within-seed SE. The larger SE is retained.

ESS here is `(sum contributions)^2 / sum(contributions^2)`. It describes concentration of the observed contributions. It is neither a count of independent Markov-chain draws nor a guarantee that an unobserved tail is harmless. The original importance-weight ESS is distinct from this target-specific contribution ESS.

## Controlled finite-range extrapolation

The independent analysis fits `Aeff/A_reference = a0 + sum b_j/L^j` with unconstrained intercept a0, varying lower cutoffs and polynomial degrees. Reported standard errors are conditional sampling errors for each chosen model; variation across acceptable models is reported separately. A fit is not allowed to force its intercept to one.

| Family | Example correction fit | Intercept ratio | Sampling SE |
|---|---|---:|---:|
| Real q=2 | L>=20, quadratic in 1/L | 1.000125 | 0.000659 |
| Real q=2 | L>=40, quadratic in 1/L | 1.000026 | 0.000826 |
| Real q=4 | L>=40, quadratic in 1/L | 1.001039 | 0.000919 |
| Real q=4 | L>=80, linear in 1/L | 1.000531 | 0.000805 |

Acceptable inspected real q=2 models using lower cutoffs from 20 to 160 and degrees one to three have intercepts approximately 0.99964 to 1.00069. The analogous q=4 range is approximately 0.99990 to 1.00148. A q=2 linear fit starting at L=20 has chi-square 161.7 on five degrees of freedom and is demonstrably inadequate; its intercept near 1.005 is not evidence against the conjecture. Such failed fits remain in the machine-readable output.

Exponents and logarithmic degrees were also fitted freely using `log Z = a - lambda L + q log L + c1/L + c2/L^2`. At L>=20, the real q=2 fit gives `lambda=1.50000017 +/- 0.00000551`, `q=1.99907 +/- 0.00416`; the real q=4 fit gives `lambda=1.50000496 +/- 0.00000644`, `q=4.00564 +/- 0.00507`. Here +/- denotes one conditional sampling SE. Changing correction order and cutoff changes these estimates; omission of corrections is strongly rejected. These fits therefore provide numerical support for the proposed exponents and multiplicities without assuming them in the estimator or fitting them as fixed values.

None of these polynomial corrections is a proved remainder expansion for the real q>=2 cases. The evidence distinguishes percent-scale alternatives within the displayed correction families, while alternatives differing by a few parts per thousand or with a later crossover cannot be reliably excluded. Exact identification of the limiting constants remains inconclusive. The inferential status must not be upgraded to mathematical PASS.

## Variance reduction and numerical bias limits

At L=12, the observed equal-sample variance reductions of shifted-window sampling relative to broad Gamma tilting are approximately 4.5 for real q=2, 4.7 for real q=4, and 3.9 for complex q=4. Runtime gains are smaller and depend on system load. Fixed component allocation offers an additional, generally modest change and uses its own variance estimator.

One supplemental comparison deserves explicit retention: for real q=4 at L=100, four iid replicates give a coefficient proxy 0.001894883 and four fixed-allocation replicates give 0.001912365. Their difference is 3.21 combined internal standard errors, or 2.57 combined between-replicate standard errors. This is a finite-sample discrepancy among six supplement comparisons; no density or stratified-variance defect was identified, but the comparison must not be described as universal agreement or a demonstrated bias reduction. The main conclusions use the separate eight-seed production and its fit sensitivity analysis.

Ordinary sampling at L>=30 frequently misses essentially all important scale regions. Some such runs report a moderate relative SE around a catastrophically small estimate. This is finite-sample rare-event undercoverage despite algebraic unbiasedness, not a contradiction of the paper. Variance ratios using those failed ordinary estimates are not reliable efficiency measurements. Narrow ordered-scale mixtures also show rare outliers or missed boundary contributions: at L=100 the real q=4 narrow proposal has a single-seed maximum contribution fraction around 0.418. The shifted mixture reduces this diagnostic substantially.

All principal weights and kernels are evaluated in log coordinates. The driver adds the deterministic reporting scale `lambda L-q log L` before exponentiation, so L=1280 does not require representing t itself or an underflowed Z. Those reference exponents affect units of output only; they do not alter proposal weights. The convenience unscaled summarizer can return zero once Z is below the floating-point range; its log estimate remains the meaningful result. The production driver avoids that path.

Floating arithmetic, pseudo-random generation and inverse-CDF endpoint rounding remain numerical approximations. Tiny direct Laplace contributions can underflow; raw zero CDF counts must be accompanied by an upper interval, not interpreted as zero probability. The real rank-two conditional CDF implementation uses a small-argument series and a stable near-equal-eigenvalue branch; independent high-precision roundoff checks are recorded separately. Nonlinear logarithms of noisy means have Jensen bias of order the squared relative SE; that is negligible for the primary small-error fits but does not disappear merely because the original arithmetic mean is unbiased.

## Actual deep-tail CDF supplement

`cdf_estimators.py` additionally estimates the actual probability `P(||P||_F^2 <= epsilon)`, without inferring it from a Laplace coefficient. Given the retained diagonals, the three terms in W are independent Gammas: the first endpoint times the product of first diagonals, the last endpoint times the product of second diagonals, and the squared off-diagonal Gaussian. The variance V contains neither integrated endpoint, which is essential for this independence.

For a conditional Gamma term with shape k and scale theta, exponential tilting by `exp(-t W)` changes the scale to `theta/(1+t theta)` and preserves k. Let `t=alpha/epsilon`, `U=tW`, and `G_t=E[exp(-tW)|diagonals]`. The identity

`F(epsilon | diagonals) = G_t E_t[exp(U) 1(U<=alpha)]`

then gives an unbiased estimator with multiplier bounded by `exp(alpha)`. The implemented scales of U are `logistic(log(t)+log(theta))`; the default alpha is one. The diagonal importance weight uses the same full mixture as before. The implementation uses one auxiliary triple of Gamma draws per diagonal sample. This is randomized tilted-Gamma conditioning, not full analytic CDF integration. No conjectural constant or CDF asymptotic enters its weights.

Independent calibration sets the two retained diagonal squares to one. The first two Gamma terms then have equal scales and combine, permitting a separate positive convolution quadrature for a two-Gamma sum. One million samples at epsilon=0.01,1,10 for each beta=1,2 agree with that quadrature within 2.25 standard errors. Seeds 889010,889011,889012,889020,889021,889022 and results are recorded in `results/independent_cdf_tilt_calibration.json`. Moderate-threshold comparisons with independent actual Gaussian matrices in `cdf_validation.json` have maximum absolute discrepancy 1.99 combined standard errors; the two real 223 exact-CDF controls differ by less than 1.94 standard errors.

The production CDF supplement uses four seeds of 250,000 samples at each of L=20,80,320,1280, where epsilon=exp(-L). With the same conservative SE rule and a t multiplier on three degrees of freedom, its deepest-scale coefficient ratios are:

| Family | Direct CDF coefficient ratio at L=1280 | Pointwise 95% half-width |
|---|---:|---:|
| Real q=2 | 1.004405 | 0.007804 |
| Real q=4 | 1.014828 | 0.008259 |
| Complex q=4 | 0.990737 | 0.025544 |

The real q=2 sequence is 1.242734,1.067854,1.016187,1.004405; the q=4 sequence is 1.665893,1.159374,1.030929,1.014828 on the four increasing scales. These direct probabilities provide a useful independent target check on the Laplace conclusions. Their Monte Carlo uncertainty is larger and finite-log corrections are still present. The q=4 deepest estimate alone does not establish a different limiting coefficient. Exact limiting-constant identification from this four-scale CDF supplement remains inconclusive.

## Reproduction

Run `python audit_checks.py` from this directory for independent calibration and production analysis, or `python audit_checks.py --production-only` to aggregate existing results and reproduce the fit sensitivity study. Results are in `results/independent_audit_checks.json`, `results/independent_is_calibration.json`, and `results/independent_production_analysis.json`. No fit output is used to change an estimator's normalization.

The independent ordinary-coordinate calibration seed is 940012; the ballot checks use 4001,4002,4003,4004. The six independent actual-mixture calibration seeds are 771600,771602,771604 for real 222 and 771700,771702,771704 for real 223. Main samples can be regenerated with `python run_experiments.py --mode direct --config config.json`, `python run_experiments.py --mode vr --config config.json`, `python run_experiments.py --mode efficiency --config config.json`, and `python run_experiments.py --mode vr --config config_ridge.json`. The driver retains existing result filenames; use a fresh output copy for an independent replay rather than silently overwriting audited results.

Run `python run_cdf_experiments.py` for the actual-CDF supplement and `python audit_checks.py --cdf-only` for the independent fixed-coordinate convolution calibration. The CDF experiment records its seeds in `results/cdf_config.json` and `results/cdf_replicates.json`.

