# Numerical-audit figure captions

All figures are available as 300 dpi PNG and vector PDF. Rebuild with
`python make_plots.py`. The input manifest and plotted data are in
`plot_data.json`, including source SHA-256 hashes. Natural logarithms are used.
Every Laplace observable is Z(t)=E exp(-t ||P||_F^2), with real N(0,1) entries
and complex (X+iY)/sqrt(2) entries. Numerical agreement is not proof.

## 01_leading_normalized_ratios

Leading-term normalization Z(t)t^lambda/(A L^q), with L=log(t) on the
logarithmic horizontal axis. The horizontal reference is one. Real q=0 and
q=1 use the known benchmark coefficients; q=2 and q=4 use the coefficients
predicted by Conjecture E. The complex panel uses lambda=3, q=4, A=1/12.
The complete leading-power logarithmic polynomial P4(L), divided by L^4/12,
is overlaid as analytic input. No finite-L offsets are removed.

Points use the shifted-window mixture importance sampler with the final
Gaussian factor integrated out; they are not direct Monte Carlo. Each
point pools eight independent seeds, totaling 2,000,000 samples. Bars are
pointwise 95% t(7) intervals with the larger of pooled within-seed and
between-seed standard errors. These are empirical Monte Carlo uncertainty
estimates, not uniform or rigorous coverage guarantees. They do not include
finite-range asymptotic error.

## 02_complex_full_polynomial

The same complex estimates divided by t^-3 P4(L), where
P4(L)=(L-3 gamma)^4/12+zeta(3)(L-3 gamma)+25 pi^4/144. Bars propagate the
same conservative pointwise sampling intervals. Hollow diamonds show the
exact finite-t polynomial-ensemble determinant evaluated with 75 decimal
digits at L=10,20,40. This determinant is analytic input and is not a second
Monte Carlo estimator. Values and evaluation precision are recorded in
plot_data.json. The normalization is an asymptotic polynomial, not an
assertion that the finite-t transform equals it.

## 03_real_intercept_window_sensitivity

Unconstrained fits of Aeff/A_reference = intercept + sum_j b_j (Lmin/L)^j,
with degree 1, 2, or 3 and Lmax=1280. Only fits with lack-of-fit p>0.01 are
shown. Degree markers are slightly displaced horizontally for legibility;
their window endpoints remain the labeled values 10,20,40,80,160. Rejected
fits remain in independent_production_analysis.json. The fit constant is
not fixed to Conjecture E. Bars are 1.96 times the WLS intercept standard
error, using the recorded working standard errors without rescaling them
to force reduced chi-square to one. They quantify sampling uncertainty
within the specified polynomial model; they do not bound omitted
subleading terms, model selection, or asymptotic extrapolation bias.
The p-value filter is descriptive and not a multiple-testing certification.

## 04_direct_frobenius_cdf

Original-entry real and complex Gaussian product sampling, pooled over
four independent seeds (2,000,000 products per case), at epsilon=1,0.1,
0.01,0.001,0.0001. The event is ||P||_F^2 <= epsilon, not ||P||_F <= epsilon.
Blue circles show direct hit frequencies and two-sided 95% Wilson intervals.
When no hits occur, a downward arrow is placed at the upper endpoint of
that Wilson interval; zero-hit estimates are not plotted as zero probability.
Orange squares show exact final-layer conditional CDF integration where
implemented (real final width two, rank two); epsilon positions are shifted
by 11% for readability. Conditional bars use t(3) times the larger of
within-seed and between-seed standard errors. Direct and conditional
estimators share Gaussian prefixes, so plotted marginal error bars are
not a paired-difference test. The green curve for real (2,2,3) is an exact
Bessel-function CDF comparator, explicitly analytic input. The additional
real (2,2,4) width is included as an exploratory benchmark. The plot shows
rare-event sampling limitations without interpreting zero hits as evidence
against a positive small-ball probability.

## 05_actual_deep_tail_cdf

The actual Frobenius small-ball CDF F(epsilon)=Pr(||P||_F^2 <= epsilon)
at L=log(1/epsilon)=20,80,320,1280, divided by kappa exp(-lambda L)L^q.
This is a separately sampled CDF observable, not a Laplace proxy. The
estimator combines Bartlett-coordinate importance sampling with randomized
exponentially tilted conditional Gamma variables; the remaining conditional
CDF is not replaced by its asymptotic formula. Each point pools four
independent seeds with 250,000 samples per seed (1,000,000 total).
Bars are t(3) pointwise 95% intervals using the larger of pooled within-seed
and between-seed standard errors, including auxiliary Gamma randomness.
The original summary's normal intervals are not used in this plot.

The horizontal line is the leading coefficient prediction; real q=2 and
q=4 references are conjectural. The dotted real q=1 curve is the known
two-term asymptotic ratio 1+(3/2-2 gamma)/L, not the exact finite-epsilon
Bessel CDF. The complex coefficient is the Frobenius value kappa=1/72,
distinct from the Laplace value A=1/12. The orange complex curve is the
derived full Frobenius logarithmic polynomial divided by kappa L^4,
computed by inversion of the finite polynomial Laplace operator. It is
analytic input, not a newly certified form of Theorem B or an exact
finite-threshold CDF. No four-point extrapolation fit is made. Sampling
intervals do not bound unknown asymptotic remainders or prove Conjecture E.
