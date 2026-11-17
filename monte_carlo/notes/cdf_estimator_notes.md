# Actual small-ball CDF estimator

`cdf_estimators.py` estimates the actual probability
\(F(\varepsilon)=\mathbb P(\|P\|_F^2\le\varepsilon)\), separately from
the Laplace-transform experiment. Its auxiliary conditional sampling is exact
in arithmetic; there is no asymptotic CDF substitution.

## Independent conditional Gamma variables

Use the remaining Bartlett diagonal squares
\(a_2,\ldots,a_N,b_1,\ldots,b_{N-1}\) from `vr_estimators.py`, and set

\[
A=\prod_{j=2}^N a_j,\quad B=\prod_{j=1}^{N-1}b_j,\quad
V=\sum_{j=1}^N\prod_{k>j}a_k\prod_{k<j}b_k.
\]

Conditional on these variables,
\[
W=\|P\|_F^2=Aa_1+Bb_N+|Y|^2
\]
is a sum of **three independent Gamma variables**, with shapes
\[
(r_1,r_2,r_3)=
\left(\frac{\beta d_1}{2},\frac{\beta(d_N-1)}2,\frac\beta2\right)
\]
and scales
\[
(\theta_1,\theta_2,\theta_3)=\frac2\beta(A,B,V).
\]

Independence follows because the Gaussian offdiagonal variance \(V\) excludes
both endpoint squares \(a_1,b_N\). The product offdiagonal \(Y\) is a linear
combination of independent standard Gaussian Bartlett offdiagonals whose
coefficients depend only on the remaining diagonals. Those offdiagonals are
independent of \(a_1,b_N\).

## Exponential-tilt identity

Let \(t=\alpha/\varepsilon\), with \(\alpha>0\). Exponentially tilting each
conditional Gamma variable by \(e^{-tW}\) preserves its shape and changes its
scale from \(\theta_j\) to \(\theta_j/(1+t\theta_j)\). If \(U=tW\) under
the tilted conditional law, its three summands therefore have scales
\[
v_j=\frac{t\theta_j}{1+t\theta_j}
=\operatorname{logistic}(\log t+\log\theta_j).
\]

Writing \(G_t=\prod_j(1+t\theta_j)^{-r_j}\), the exact conditional CDF is
\[
F(\varepsilon\mid a,b)
=G_t\,\mathbb E_t[e^U\mathbf1_{\{U\le\alpha\}}].
\]

The implementation samples one independent tilted-Gamma triple per diagonal
importance draw and multiplies by the complete diagonal likelihood ratio
\(p/q\). The resulting random contribution is
\[
X=\frac pqG_t e^U\mathbf1_{\{U\le\alpha\}}.
\]

It is unbiased for the finite-threshold CDF in exact arithmetic. The auxiliary
multiplier is bounded by \(e^\alpha\). Production uses \(\alpha=1\).
This is **randomized tilted-Gamma conditional Monte Carlo**, not an analytic
evaluation of the remaining CDF integral. Auxiliary randomness increases
variance; its effect is included in the sample variance of the full weighted
contributions. No separate binomial error estimate is substituted.

## Large logarithms and reporting

All inputs use \(L=\log(1/\varepsilon)\); neither \(\varepsilon\) nor
\(t\) needs to be representable. The estimator retains log contributions and
reports
\[
\widehat\kappa(L)=\widehat F(e^{-L})\,e^{\lambda L}/L^q.
\]

The known/predicted \(\lambda,q\) are used only to rescale the output;
the target coefficient is never passed into the sampler. For complex products
the comparison is the Frobenius-CDF coefficient \(\kappa_F\), not the
spectral coefficient or Laplace coefficient. Real (q\ge2\) constants remain
conjectural, and finite-range agreement does not prove an asymptotic limit.

Double precision is used for logistic scales, Gamma sampling, log sums and
contributions. Very small logistic scales may underflow to zero and very large
ones round to one; no rigorous interval bound is claimed for these floating-
point effects. There is no imposed cutoff on Bartlett log coordinates or
omission of target regions. The full-support importance mixture is retained.

## Reproduction and validation

Run `python run_cdf_experiments.py`. Configuration and all independent seeds
are written to `results/cdf_config.json` and `results/cdf_replicates.json`.
The five required real/complex benchmarks are sampled at
\(L=20,80,320,1280\), with four independent seeds and 250,000 draws per seed.
`R223` uses the ridge mixture; the others use the shifted-window mixture.

`results/cdf_summary.json` records internal and between-seed errors,
contribution ESS and maximum contribution fractions. At moderate thresholds
\(\varepsilon=1,0.1\), separate direct Gaussian-matrix samples provide an
independent validation. The real \((2,2,3)\) case is also compared with the
independently derived finite-threshold Bessel CDF from
`analytic_reference.exact_real_223_cdf`. Those checks are recorded in
`results/cdf_validation.json`. The same exact Bessel representation, evaluated
with automatically increased precision, is also compared at every production
logarithm in `results/cdf_exact_r223_validation.json`; this is a finite-threshold
control, not a replacement by the leading monomial.
