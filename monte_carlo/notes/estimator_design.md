# Independent variance-reduced Bartlett estimators

Implementation: `vr_estimators.py`. Target convention throughout:

\[
Z(t)=\mathbb E\exp\{-t\|A_N\cdots A_1\|_F^2\}.
\]

Real entries have distribution \(N(0,1)\); complex entries are
\((X+iY)/\sqrt2\), with independent standard real normals. This is distinct
from the population-loss convention \(K=\|P\|_F^2/2\): that convention uses
the present transform at \(t/2\).

## Exact final-matrix conditioning

Given \(H=A_{N-1}\cdots A_1\), the existing final Gaussian matrix can be
integrated without adding a model parameter:

\[
Z_{\mathbb R}(t\mid H)=\det(I+2tH^TH)^{-d_N/2},\qquad
Z_{\mathbb C}(t\mid H)=\det(I+tH^*H)^{-d_N}.
\]

`final_matrix_log_kernel` computes these formulas from singular values. Direct
sampling is separately implemented in `gaussian_products` and contains no
Bartlett reduction or asymptotic input.

## Stronger exact conditioning when the bottleneck is two

Assume \(d_0=2\) and \(d_j\ge2\). Successive thin QR decompositions yield
independent upper triangular matrices

\[
R_j=\begin{pmatrix}x_j&y_j\\0&z_j\end{pmatrix}.
\]

Set \(\beta=1\) for real matrices and \(\beta=2\) for complex matrices, and
\(c=2/\beta\). The diagonal squares are independent:

\[
a_j=x_j^2\sim\Gamma(\beta d_j/2,c),\qquad
b_j=z_j^2\sim\Gamma(\beta(d_j-1)/2,c).
\]

Every real offdiagonal is \(N(0,1)\); every complex offdiagonal is standard
variance-one complex Gaussian. The product has diagonal squares
\(\prod a_j,\prod b_j\). Conditional on all diagonal squares, its offdiagonal
is Gaussian with variance parameter

\[
V=\sum_{j=1}^N\prod_{k>j}a_k\prod_{k<j}b_k.
\]

In particular, neither \(a_1\) nor \(b_N\) enters \(V\). Integrating every
offdiagonal, then these two endpoint diagonal squares, gives the exact positive
integrand over only \(a_2,\ldots,a_N,b_1,\ldots,b_{N-1}\):

\[
G_t=(1+ct\prod_{j=2}^Na_j)^{-\beta d_1/2}
    (1+ct\prod_{j=1}^{N-1}b_j)^{-\beta(d_N-1)/2}
    (1+ctV)^{-\beta/2}.
\]

The target is exactly \(Z(t)=\mathbb E G_t\). No conjectured coefficient,
small-ball asymptotic, or polynomial truncation is used in this identity.
`conditional_log_kernel` uses sums, `logaddexp` and `logsumexp`; it never forms
\(t\) or extremely small diagonal products. Thus \(\log t=1280\) is
computationally accessible even though both \(t\) and \(Z(t)\) are outside
ordinary floating-point magnitude ranges.

## Importance proposals and their exact densities

Let \(M=N-1\), \(y_i^a=-\log(a_i/c)\) and
\(y_i^b=-\log(b_i/c)\). Target log-Gamma coordinate densities are evaluated
exactly as

\[
\log p(\log a)=\alpha(\log a-\log c)-a/c-\log\Gamma(\alpha).
\]

Gamma variables with shape below one are generated in log coordinates using
\(\log\Gamma_{\alpha+1}+\log U/\alpha\). No small-value clipping or
underflow-based point mass at zero is introduced.

### Original and broad-tail components

The original component is the exact joint Gamma prior. The broad component
replaces each shape \(\alpha_i\) with
\(k_i=\min(\alpha_i,1/\max(L,1))\), leaving scale unchanged, where
\(L=\max(\log t+N\log c,0.1)\). Its density is fully known. It has full
support and is not a truncated small-scale approximation.

### Ordered Schubert-window component

The two totals \(S_a=\sum y_i^a,S_b=\sum y_i^b\) are sampled independently
from logistic distributions of center \(L\) and width three, each conditioned
to be positive. Given the totals, uniform Dirichlet allocations \(u,v\) are
conditioned on

\[
\sum_{i=1}^jv_i\ge\sum_{i=1}^ju_i\quad(1\le j<M).
\]

The allocation event has probability exactly \(1/M\). This follows by
interleaving the two sets of \(M-1\) independent uniform order statistics:
the favorable ballot interleavings are a Catalan fraction \(1/M\). It is a
normalization of a proposal distribution, not Conjecture E or an assumption
about an asymptotic coefficient. The proposal density in the \(2M\) log
coordinates is

\[
q_{\rm ord}=
\frac{M\,\Gamma(M)^2 f_+(S_a)f_+(S_b)}
     {S_a^{M-1}S_b^{M-1}}\,\mathbf1_{\rm ordered},
\]

where \(f_+\) is the normalized positive-logistic density. The powers of the
totals are the two simplex-coordinate Jacobians. Samples are obtained by
rejection from two independent uniform Dirichlet vectors.

### Shifted unordered window

The ordered proposal alone can miss important finite-logarithm mass just
outside its window. To cover it efficiently, use shifted coordinates
\(\widetilde y_i=y_i+b\ge0\), with \(b=4\), and positive-logistic totals
of center \(L+Mb\). The allocations are independent uniform Dirichlet vectors
without any order restriction. The density is

\[
q_{\rm shift}=
\frac{\Gamma(M)^2f_+(\widetilde S_a)f_+(\widetilde S_b)}
     {\widetilde S_a^{M-1}\widetilde S_b^{M-1}},
\quad y_i^a,y_i^b\ge-b.
\]

Default `shifted_window_mixture` probabilities are ordered 0.60, shifted 0.30,
broad Gamma 0.05 and original Gamma 0.05. Assignment is independent categorical
sampling for every draw. The complete normalized mixture density is used in
the denominator regardless of the component that generated a sample.

The resulting contribution is \(G_t p/q_{\rm mix}\), never divided by an
estimated normalization. Since the mixture includes 5% of the original prior,
\(p/q_{\rm mix}\le20\). This bounds the ordinary likelihood ratio, but does
not guarantee a small relative error for an exponentially small expectation.

`schubert_mixture` is the comparison estimator with probabilities 0.90 ordered,
0.05 broad and 0.05 original. `gamma_tilt` uses 0.95 broad and 0.05 original.
`ordinary` samples the original diagonal Gamma variables without importance
sampling. All share the same exact conditional kernel.

### Two-factor ridge control

For the known \(d=(2,2,3)\) benchmark the logarithm arises from a ridge:
\(y_a\) ranges over approximately \([0,L]\), while \(y_b\) stays near \(L\).
`ridge_mixture` samples \(y_a\) uniformly from \([-b,L+b]\) and
\(y_b+b\) from a positive-logistic distribution centered at \(L+b\).
Its joint density is the product of the normalized uniform and logistic
densities. Probabilities are ridge 0.90, broad Gamma 0.05, original Gamma 0.05.
This proposal is restricted to two factors; the exact kernel remains unchanged.

## Uncertainty, ESS and bias

In the default APIs, each mixture draw is iid. Ordinary sample variance of the
*weighted contributions* estimates variance of the unnormalized sample mean.
The separate fixed-allocation API below uses a different variance formula.
Independent production seeds should be used to check replicate dispersion and
compare proposal choices.

Contribution ESS is \((\sum X_i)^2/\sum X_i^2\), where
\(X_i=G_t p/q_{\rm mix}\). Prior-weight ESS, computed from \(p/q\), measures
a different problem and should not be substituted. Neither ESS diagnoses
unseen rare high contributions by itself. Maximum contribution fraction,
seed-to-seed variation and comparisons of ordered-only versus shifted
proposals are necessary diagnostics.

In exact arithmetic the conditional expectations and every importance
estimator described here are unbiased for finite \(t\). There is no
self-normalization, fitted target density, asymptotic coefficient insertion,
rejection of low-weight target samples, or truncated logarithmic region.
Ordinary IEEE floating-point errors remain; they are not a formal interval
certificate. The implementation evaluates in log coordinates and avoids
forming exponentially large/small products. Unrounded log estimates must be
retained when ordinary estimates underflow to zero.

A ratio \(Z(t)t^\lambda/(\log t)^q\) at finite \(t\) is not an estimator
of the limiting coefficient without a subleading-error argument. Complex
results should be compared to their independently derived full logarithmic
polynomial. Real high-q finite-range extrapolation should be labeled model-
dependent and tested against different ranges and correction orders. Numerical
agreement is never a proof of asymptotic equivalence.

## Explicit fixed-allocation Bartlett scale stratification

`stratified_log_contributions` assigns fixed counts to the ordered window,
shifted window, original Gamma and broad Gamma components. Largest-remainder
rounding makes these counts sum to the requested sample size. The actual
fractions \(\alpha_k=n_k/n\), not the nominal probabilities, are used in the
balance denominator \(q_{\rm mix}=\sum_k\alpha_kq_k\). Within each component,
samples are independent. If \(X_{ki}=G_t p/q_{\rm mix}\), then

\[
\widehat Z=\sum_k\alpha_k\overline X_k,\qquad
\widehat{\operatorname{Var}}(\widehat Z)
=\sum_k\frac{\alpha_k^2s_k^2}{n_k}.
\]

`summarize_stratified` implements this formula and reports the pooled iid
standard error only as a comparison diagnostic. Treating all observations as
iid would include a between-stratum term that is absent in the fixed design.
No result is conditioned on passing a noisy normalization check.

The separate `stratified_experiment.py` uses four independent seeds and
200,000 observations per replication, for real and complex four-factor
two-by-two products at \(\log t=30,100,320\). It records a configuration,
every replication and aggregate internal/replicate errors in
`results/stratified_*.json`. Its iid and fixed-allocation streams have distinct
recorded seeds. This is stratification across explicitly normalized Bartlett
scale-window proposals, not a claim of an exhaustive geometric cell
decomposition of the random-matrix target law.

The real-valued summaries of this stratification experiment report the
**Laplace-inferred effective coefficient**
\(e^{\lambda L}Z(e^L)/[\Gamma(\lambda+1)L^q]\), not an actual finite-threshold
CDF. Its finite-logarithm corrections differ from those of
\(e^{\lambda L}F(e^{-L})/L^q\). The latter is evaluated only in the separate
`cdf_estimators.py` / `run_cdf_experiments.py` supplement.

## Pilot scope

Development pilots were used only to select proposal geometry and diagnose
variance, not as independent production evidence. They showed severe rare
outliers for ordered-only proposals at moderate logarithms; the shifted
component reduced this instability. The production experiment must use its own
recorded seeds. No benchmark leading coefficient appears in `vr_estimators.py`.
