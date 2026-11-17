# Monte Carlo verification

**Date:** 17 November 2026. **Decision:** numerical support for the
tested predictions, with explicit finite-range qualifications; no mathematical
certification and no change to Conjecture E's unproved status.

## 1. Main findings

The real and complex Gaussian product estimators reproduce independently known
finite-parameter benchmarks. Importance sampling in Bartlett coordinates makes
the deep-tail regime accessible up to natural log parameter `L=1280`, without
representing exponentially large `t` or exponentially small probabilities directly.
The real q=2 and q=4 data approach the proposed constants after substantial
subleading logarithmic corrections. The complex q=4 data agree with the complete
leading-power logarithmic polynomial, whereas its leading monomial alone is a
poor finite-parameter approximation.

For the Laplace coefficient, an unconstrained quadratic fit in `1/L` gives:

| Real family | Fit window | Fitted coefficient / conjectured coefficient | Sampling SE | Decision |
|---|---|---:|---:|---|
| (2,2,2,2), q=2 | 20 ≤ L ≤ 1280 | 1.000125 | 0.000659 | Compatible within this correction model |
| (2,2,2,2), q=2 | 40 ≤ L ≤ 1280 | 1.000026 | 0.000826 | Stable on removing a low-L point |
| (2,2,2,2,2), q=4 | 40 ≤ L ≤ 1280 | 1.001039 | 0.000919 | Compatible within this correction model |

These are **sampling standard errors of descriptive fitted models**, not error
bounds on an asymptotic coefficient. Across the retained windows and polynomial
degrees with lower cutoffs Lmin≥20 and lack-of-fit p>0.01, the q=2 intercepts range approximately
0.99964–1.00069 and the q=4 intercepts 0.99990–1.00148. This selected-model spread
is a sensitivity diagnostic, not a rigorous systematic-error interval. Fits with
arbitrary slower corrections have not been excluded. Accordingly the numerical
evidence supports Conjecture E at the resolution of the tested models; proving its
exact constants, or distinguishing sufficiently close alternatives without
remainder assumptions, remains **inconclusive**.

## 2. Inputs and normalization

All formulas and normalizations follow the paper.

For independent matrices `A_j` of dimensions `d_j × d_{j-1}`, define

\[
W=\|A_N\cdots A_1\|_F^2,\qquad Z(t)=\mathbb E e^{-tW},\qquad
F_F(\varepsilon)=\mathbb P(W\le\varepsilon).
\]

Real entries are iid `N(0,1)`. Complex entries are `(X+iY)/sqrt(2)` with independent
standard real normals. There is **no factor 1/2 in the audited Laplace observable**.
For the different loss `W/2`, both powers and logarithmic shifts must be changed by
evaluating `Z(t/2)`.

We write `F_F(ε) ~ κ ε^λ (log(1/ε))^q`, so the leading Laplace coefficient is
`A=Γ(λ+1)κ`, and pole multiplicity is `m=q+1`. This coefficient conversion is
analytic input; it does not identify the two finite-L effective coefficients.

| Field / widths | λ | q | Frobenius κ | Laplace A | Status in the paper |
|---|---:|---:|---|---|---|
| Real (2,2,2) | 3/2 | 0 | π/6 | π^(3/2)/8 | Known benchmark |
| Real (2,2,3) | 2 | 1 | 1/16 | 1/8 | Known benchmark |
| Real (2,2,2,2) | 3/2 | 2 | √(2π)/48 | π√2/64 | Conjecture E |
| Real (2,2,2,2,2) | 3/2 | 4 | 1/576 | √π/768 | Conjecture E |
| Complex (2,2,2,2,2) | 3 | 4 | 1/72 | 1/12 | Theorem A/B comparator |
| Real (2,2,4), additional control | 2 | 0 | 1/16 | 1/8 | Additional width, moderate-range tests |

The complex operator-norm coefficient is `1/48`, distinct from the Frobenius
coefficient `1/72` and Laplace coefficient `1/12`. Direct operator-norm observations
are retained but no precise deep-tail operator-norm coefficient is claimed.

## 3. Independent estimators and analytic inputs

### Original Gaussian matrices and final-layer conditioning

`direct_mc.py` samples original Cartesian Gaussian entries and multiplies the
matrices. It records direct Laplace observations and Frobenius/operator small-ball
indicators. If `B=A_{N-1}...A_1`, exact final-layer Gaussian integration yields

\[
Z(t\mid B)=\begin{cases}
\det(I+2t B B^T)^{-d_N/2},&\text{real},\\
\det(I+t B B^*)^{-d_N},&\text{complex}.
\end{cases}
\]

The implementation uses the two nonzero singular-value squares. Direct and
conditional observations share the same prefix, so their difference is assessed
with its recorded **paired** variance. Conditional Frobenius CDF integration is
also implemented for the two-exponential real final-row-two case. Complex CDF
comparisons in the direct module remain direct indicator sampling.

### Exact Bartlett reduction

The separate importance implementation uses Gaussian rotational invariance and
thin QR/Bartlett coordinates. For a bottleneck-two product, write each triangular
factor as `R_j=[[x_j,y_j],[0,z_j]]`, with
`a_j=x_j² ~ Gamma(β d_j/2, 2/β)` and
`b_j=z_j² ~ Gamma(β(d_j−1)/2, 2/β)`, where β=1 or 2. Offdiagonal entries are real
or complex Gaussian. All coordinates are independent before forming the product.

Conditional on the diagonals, the product offdiagonal is Gaussian with variance

\[
V=\sum_{j=1}^N\left(\prod_{k>j}a_k\right)
                       \left(\prod_{k<j}b_k\right).
\]

Integrating that offdiagonal and endpoint diagonals `a_1,b_N` leaves the exact kernel

\[
K_t=(1+ct\!\prod_{j=2}^Na_j)^{-\beta d_1/2}
    (1+ct\!\prod_{j=1}^{N-1}b_j)^{-\beta(d_N-1)/2}
    (1+ctV)^{-\beta/2},\quad c=2/\beta.
\]

This is a distributional reduction of the original matrices, with no asymptotic
approximation. It was checked against Cartesian sampling and independent exact
finite-t formulas. All products, sums, kernels and likelihood ratios are evaluated
in log coordinates. Predicted λ and q are used only to rescale recorded outputs;
they do not enter `K_t` or importance weights.

### Importance sampling and scale stratification

Four strategies are compared: true-prior sampling of the remaining diagonals,
broad Gamma shape tilting, an ordered Schubert-window mixture, and an augmented
mixture adding shifted, unordered scale windows. The production mixture is 60%
ordered windows, 30% shifted windows, 5% broad Gamma tilt and 5% original prior.
The real (2,2,3) chain uses a separately normalized ridge proposal suited to its
one-log geometry, also retaining 5% original prior.

The ordered windows sample log-scale totals with normalized positive-truncated
logistic distributions and two uniform simplex points subject to an ordering
condition. Its normalization probability is exactly `1/(N−1)`; it is included in
the proposal density, as are all Jacobians. The shifted component covers boundary
and unordered contributions missed by an overly narrow geometric proposal.
Independent numerical checks test those normalizations. Full derivations and
proposal parameters are in `estimator_design.md` and the recorded batch metadata.

Each contribution is `K_t p/q_proposal` and is averaged without self-normalizing
the weights. The original-prior component gives `p/q_proposal ≤ 20` in exact
arithmetic. This ensures finite absolute variance for the bounded kernel, but does
not promise a small relative error for a rare quantity. A geometrically motivated
proposal therefore does not assume the conjectured coefficient: all strata have
their actual probability weights, and the omitted geometry remains covered.

The fixed-allocation supplement draws prescribed counts in each mixture component
and uses the actual allocation fractions in the balance density. Its variance is
`sum_k α_k² s_k²/n_k`; it is not treated as IID-mixture sampling. The contribution
ESS for stratified runs is descriptive only.

## 4. Calibration, uncertainty and numerical bias

The analytic-reference module uses separate finite-t formulas: positive
one-dimensional quadrature for real (2,2,2), an exponential-integral formula for
real (2,2,3), and a Meijer-G determinant for complex square-two products. For the
real q=1 case it also evaluates the exact CDF

\[
F_F(\varepsilon)=1-\varepsilon K_2(\sqrt\varepsilon)
                       +\varepsilon K_2(\sqrt{2\varepsilon}).
\]

These formulas are explicitly **analytic inputs**, not Monte Carlo outputs. The
sampler never reads their constants when constructing contributions. Their
cross-checks include quadrature versus closed form and high-precision evaluation.

Across the 18 pooled final-layer-conditional calibration points (three families,
six t values), the largest standardized difference from the independent exact
reference is 1.845 working SE. A separate six-point importance calibration, with
300,000 draws per point and independent seeds, is within 2.08 SE of the exact
values. Direct-versus-conditional comparisons use the recorded paired covariance.
An earlier exploratory paired comparison at 2.93 SE is retained in validation
records; no random seeds were discarded to improve agreement.

For the main eight-seed groups, the working SE is the larger of the pooled
within-seed SE and the between-seed SE. Displayed 95% intervals use the Student-t
quantile with seven degrees of freedom. These are pointwise empirical intervals,
not simultaneous bands. Observations at different L use distinct seeds. Direct
curves evaluated on the same matrix draws have covariance, which is retained.

Contribution ESS is `(sum w_i K_i)^2 / sum (w_i K_i)^2`, not a Markov-chain ESS.
Importance-weight ESS and maximum-contribution fractions are recorded separately.
For Bernoulli CDFs, Wilson intervals are used; zero events give an upper limit, not
zero uncertainty. The archive retains raw normal-interval fields as well, but their
zero-event values must not be used to infer exact probability zero.

The exact kernels and correctly normalized importance averages are unbiased in
exact arithmetic. Finite random samples, finite arithmetic and logarithmic
extrapolation are separate limitations. Three full seeded jobs reproduce their
recorded sums bit-for-bit in the recorded software environment. Comparing 200,000
sampled high-L kernels against extended precision gives maximum absolute log-kernel
error `1.07e-12`. This spot check is far smaller than observed sampling error but is
not a rigorous error bound for every density operation. Direct Gaussian products
are used only at moderate scales; deep-tail experiments avoid numeric underflow by
retaining log probabilities and rescaled contributions.

## 5. Convergence and logarithmic multiplicities

Define the plotted Laplace ratio

\[
R_Z(L)=\frac{e^{\lambda L}Z(e^L)}{A L^q}.
\]

The following uncertainties are pointwise 95% sampling half-widths, including the
between-seed safeguard:

| Family | L=40 | L=160 | L=1280 |
|---|---:|---:|---:|
| Real (2,2,2) | 1.00055 ± 0.00198 | 1.00026 ± 0.00199 | 0.99978 ± 0.00289 |
| Real (2,2,3) | 0.98632 ± 0.00147 | 0.99695 ± 0.00126 | 0.99936 ± 0.00158 |
| Real (2,2,2,2) | 1.10202 ± 0.00295 | 1.02822 ± 0.00245 | 1.00359 ± 0.00233 |
| Real (2,2,2,2,2) | 1.24456 ± 0.00601 | 1.05849 ± 0.00296 | 1.00833 ± 0.00249 |
| Complex (2,2,2,2,2) | 0.83834 ± 0.00586 | 0.95611 ± 0.00409 | 0.99278 ± 0.00709 |

The q=1 exact leading-power formula is `t^-2 (log t−γ_E)/8`, explaining its
negative finite-L offset. The real high-q offsets are positive. Their deviation
from one at finite L must not be called a contradiction of the limiting constant.

Unconstrained fits of
`log Z = a − λ L + q log L + c1/L + c2/L²` on 20 ≤ L ≤ 1280 give:

| Real family | Fitted λ ± sampling SE | Fitted q ± sampling SE |
|---|---:|---:|
| (2,2,2,2) | 1.50000017 ± 0.00000551 | 1.99907 ± 0.00416 |
| (2,2,2,2,2) | 1.50000496 ± 0.00000644 | 4.00564 ± 0.00507 |

λ and q are free fit parameters; these fits do not simply replot an imposed
power. Uncertainty is conditional on the tested correction family. A linear
`1/L` fit for the q=2 coefficient over L≥20 has χ²=161.7 with five degrees of
freedom and is rejected as a finite-range model; its shifted intercept is not
used as evidence against the conjecture. The complete fit grid, covariances,
conditioning diagnostics and lack-of-fit tests are retained.

At L=1280 the pooled main contribution ESS is approximately 685,000 for real q=2,
628,000 for real q=4, and 295,000 for complex q=4, each from 2,000,000 proposal
draws. No single recorded contribution dominates these final main estimates.

## 6. Complex full-polynomial comparison

For four complex square-two factors, independent residue reconstruction gives

\[
Z(t)=t^{-3}P(\log t)+\text{smaller-power terms},\qquad
P(L)=\frac{(L-3\gamma_E)^4}{12}
       +\zeta(3)(L-3\gamma_E)+\frac{25\pi^4}{144}.
\]

This includes every logarithmic term at leading power, rather than retaining only
`L^4/12`. At L=10,20,40,80 its ratio to that monomial is respectively
0.499607, 0.699025, 0.838054, 0.916221. At L=10 the exact finite-t Meijer-G value
differs from the full leading-power polynomial by about −0.463%; at L=20 the
relative correction is about −4.28e-7. Those corrections are separately labeled
analytic-reference calculations.

All eight complex importance estimates agree with the full polynomial within
1.506 working SE. The finite-t exact reference is used where its remainder matters.
The plot therefore displays both the full-polynomial ratio and the exact finite-t
comparison, with uncertainty. Agreement is numerical support for this benchmark,
not a proof of Theorem B or of its general-width formulation.

## 7. Efficiency and retained failure diagnostics

Final-layer conditioning reduces per-draw variance substantially at accessible
t, while retaining exact expectation. For example, at t=100 the measured direct
to conditional variance ratio is about 16.4 for real (2,2,2), 7.9 for real
(2,2,3), 4.7 for real q=2, 2.9 for real q=4 and 6.5 for complex q=4.

At L=12, the shifted-window estimator versus broad Gamma tilt reduces measured
per-draw variance by approximately 4.5× for real q=2, 4.6× for real q=4, and 4.0×
for complex q=4. Density evaluation and rejection sampling cost more per draw;
these are variance gains, not claims of equal wall-clock speedup. Per-job timings
are recorded and are machine- and concurrency-dependent.

The ordered-only window sometimes produces severe rare-weight outliers at moderate
L. The shifted component materially stabilizes those boundary contributions.
Fixed allocation adds only a modest roughly 0–11% variance reduction in the
supplement. One real q=4, L=100 IID-versus-stratified comparison differs by 3.2
internal SE, or 2.57 replicate-based SE; it is retained and flagged. An independent
code inspection found no balance-density or stratum-variance normalization error.
The discrepancy is a reason to keep replication-based uncertainty, not to erase
the run or claim perfect agreement.

At L≥30, ordinary sampling of the reduced integral can miss essentially all
important rare mass while reporting a deceptively small empirical SE. Such values
are **failed estimates**, not evidence against a theorem. We do not compute a
scientific speedup by comparing a valid importance estimator to the spurious
precision of those failures. Direct complex small-ball sampling has zero events
at the smallest two thresholds in the moderate-range grid; the associated upper
limits cannot resolve its asymptotic coefficient.

## 8. Actual small-ball CDF supplement

The separate CDF supplement estimates `F_F(e^-L)` itself. Given the retained
Bartlett diagonals, `W` is a sum of three independent Gamma variables. Their shapes
are `βd_1/2`, `β(d_N−1)/2`, `β/2`, and scales are `c prod(a)`, `c prod(b)`, `cV`.
The independence holds because the two integrated endpoints do not occur in V.
At `t=e^L`, exponentially tilting those Gammas yields `U=tW` with unchanged shapes
and scales `logistic(log t + log original_scale)`. Thus exactly

\[
F_F(1/t\mid\text{diagonals})=K_t\,
\mathbb E_t\left[e^U\mathbf 1_{U\le1}\right].
\]

The bounded multiplier is at most e. It is evaluated by additional randomized
Gamma draws and multiplied by the same exact likelihood ratio; this is **not** a
complete analytic CDF integration and adds sampling variance. It uses no conjectural
coefficient or asymptotic replacement of the distribution. Both field normalizations
and all five primary families are covered. Numerical tables and independent
validation are recorded in `results/cdf_summary.json` and `results/cdf_validation.json`.

The supplement uses four independent seeds of 250,000 draws per point, with
L=20,80,320,1280 (20 million proposal draws). At L=1280, the actual-CDF ratio
`F_F(e^-L)/(κ e^-λL L^q)` is:

| Family | Ratio ± conservative pointwise 95% half-width | Mean ESS per 250,000-draw replicate |
|---|---:|---:|
| R222 | 0.99898 ± 0.00844 | 43,295 |
| R223 | 0.99784 ± 0.00707 | 41,966 |
| R2222 | 1.00441 ± 0.00780 | 35,918 |
| R22222 | 1.01483 ± 0.00826 | 33,157 |
| C22222 | 0.99074 ± 0.02554 | 3,752 |

These intervals use `t(3) × max(within-seed SE, between-seed SE)`; the raw CDF
summary also retains normal/internal-SE intervals, which are not substituted for
these conservative displays. The real q=4 ratios decrease from about 1.666 at L=20
to 1.159 at L=80, 1.031 at L=320 and 1.015 at L=1280. A finite-L excess above one
persists and must be modeled before estimating the limit. Four points and this
larger CDF sampling error do not independently identify the exact high-q constants.
This actual-CDF-only test is **inconclusive for exact constant identification**,
while supporting the predicted direction of convergence.

Ten moderate-threshold comparisons with independent Cartesian matrix samples have
maximum |z|=1.982. For real (2,2,3), the exact finite-threshold Bessel CDF at all
four production L values agrees within 1.609 internal SE. Independent positive
convolution quadrature at fixed Bartlett diagonals validates the implemented tilt
for both fields within 2.25 SE. The known real q=1 CDF correction is
`[L+3/2−2γ_E]/16`, whereas its Laplace correction is `[L−γ_E]/8`: this explains why
the two finite-L coefficient ratios are not identical.

The complex Frobenius leading-power polynomial is separately reconstructed from
its exact density and from the finite polynomial transformation
`P_Lap = Γ(4−D_L) P_F`. Those implementations agree in the recorded reference
checks. Its leading coefficient is 1/72; it is a derived comparator, distinct from
Theorem B's operator-norm polynomial. The actual-CDF plot uses that full Frobenius
polynomial. No real conjectural coefficient is used in constructing the tilted
CDF estimator.


## 9. Summary of numerical conclusions

| Claim / question | Numerical decision | Limit |
|---|---|---|
| Gaussian normalization and exact conditioning | Supported by independent finite-t controls and seed replay | Numerical implementation checks |
| Known real q=0,1 leading coefficients | Supported in deep-tail estimates and exact finite-t calibration | No new theorem certification |
| Complex C=3, q=4, A=1/12 | Supported when the complete logarithmic polynomial is used | Tested square-two chain only |
| Real q=2,4 exponents and logarithmic multiplicities | Supported by unconstrained finite-range fits | Correction-model dependence remains |
| Conjecture E exact κ at q=2,4 | Numerically compatible, stable across tested fit windows | Exact asymptotic equality remains unproved; close alternatives inconclusive |
| Pure leading-monomial approximation at modest L | Inadequate for high q | This is not an asymptotic theorem contradiction |
| General real widths / all Conjecture E cases | Not established by these runs | Additional widths have limited coverage |

No analytical claim is reliably contradicted by the successful estimators in this
audit.

## 10. Reproducibility record and deliverables

Main Cartesian sampling uses 24 independent jobs of 500,000 products (12 million),
six families, four seeds each, with t from 0.1 to 10,000 and ε from 1 to 1e-4.
The primary deep-L Laplace grid has 256 jobs of 250,000 draws (64 million), four
families, eight L values and eight seeds. The q=1 ridge supplement adds 40 jobs
(10 million). The method-efficiency grid uses 384 jobs of 100,000 draws
(38.4 million). All 704 jobs have distinct recorded integer seeds. The fixed-strata
supplement and actual-CDF supplement have separate seeds and configuration files.
Sample counts refer to product or proposal draws; different estimators do not
necessarily have the same work per draw.

The archive includes executable source, exact configurations, per-seed sufficient
statistics and covariance matrices where applicable, pooled CSV/JSON results,
scientific PNG/PDF figures, detailed estimator derivations, audit notes and
environment versions. Individual matrices are regenerated from the recorded seeds rather
than stored; sufficient statistics permit the documented uncertainty calculations.

`README.md` lists the reproduction commands. Numerical agreement in a finite range of
scales supports, but does not prove, the asymptotic statements it is compared with.

