# Independent analytic comparators for the numerical audit

These calculations provide comparison targets for numerical estimators. They are
not an independent mathematical certification of the paper or of Conjecture E.
No comparison coefficient is used to define an estimator or its importance weights.

## Source scope

The normalization and theorem structure follow the paper (Eq. (1) and Theorem B); the exact
Mellin-residue formulas are those used in the paper's proof of Theorem B.

## Normalization and distinct observables

Real entries are independent `N(0,1)`. Complex entries are `(X+iY)/sqrt(2)`, with
independent `X,Y~N(0,1)`, and therefore have density `pi^-1 exp(-|z|^2)`.
Let `g=||A_N ... A_1||_F^2`. The paper's primary Laplace observable is

\[
Z(t)=E e^{-t g}.
\]

It does **not** contain a factor of one half. For a Frobenius CDF coefficient
`kappa` and exponent `lambda`, the Laplace coefficient is

\[
A=\Gamma(\lambda+1)\kappa.
\]

For the distinct Bayesian loss `g/2`, one must evaluate `Z(t/2)`, including the
shift `log(t/2)` in every logarithmic polynomial. Merely multiplying the leading
power by `2^lambda` loses lower logarithmic terms.

| Field and widths | lambda | q | Frobenius CDF kappa | Laplace A | Status of coefficient |
|---|---:|---:|---|---|---|
| Real (2,2,2) | 3/2 | 0 | pi/6 | pi^(3/2)/8 | Proved benchmark |
| Real (2,2,3) | 2 | 1 | 1/16 | 1/8 | Proved benchmark |
| Real (2,2,2,2) | 3/2 | 2 | sqrt(2 pi)/48 | pi sqrt(2)/64 | Conjecture E |
| Real (2,2,2,2,2) | 3/2 | 4 | 1/576 | sqrt(pi)/768 | Conjecture E |
| Complex (2,2,2,2,2) | 3 | 4 | 1/72 | 1/12 | Theorem A |

For the last complex example the **operator-norm** CDF coefficient is `1/48`,
which is different from both `1/72` and `1/12`.

## Full complex logarithmic polynomial

The code computes the polynomial from the determinant of principal Mellin
residues, using truncated formal exponential series. It does not insert the
leading coefficient as an input. For equal square size two and `N` factors,
the column transforms are `Gamma(s)^N` and `s Gamma(s)^N`, and the determinant
normalization is one. Define `h_k` as the coefficients of

\[
e^{Lu}\Gamma(1+u)^N\Gamma(1-u)=\sum_{k\ge0}h_k(L)u^k.
\]

For `N>=2`, the residue determinant simplifies to

\[
P_{\rm Lap}(L)=h_{N-2}(L)^2-h_{N-1}(L)h_{N-3}(L),\qquad h_{-1}=0.
\]

Thus for four factors, writing `x=L-3 gamma_E`,

\[
\boxed{P_{\rm Lap}(L)=\frac{x^4}{12}+\zeta(3)x+\frac{25\pi^4}{144}.}
\]

The code also independently constructs the operator-CDF polynomial. With
`y=L-4 gamma_E`, it is

\[
P_{\rm op}(L)=\frac{y^4}{48}+\frac{y^3}{8}+\frac{9y^2}{32}
+\left(\frac3{16}+\frac{\zeta(3)}3\right)y
+\frac1{16}+\frac{\zeta(3)}2+\frac{5\pi^2}{48}+\frac{\pi^4}{36}.
\]

Theorem B directly supplies Laplace and operator-CDF polynomials, not a statement
that the Frobenius CDF has the same polynomial. Those observables must remain
separate.

The finite-log correction is substantial:

| L=log t | P_Lap(L)/(L^4/12) | Exact Z(t)/(t^-3 P_Lap(L)) |
|---:|---:|---:|
| 5 | 0.582701588 | 0.602499757 |
| 10 | 0.499606764 | 0.995369094 |
| 20 | 0.699024876 | 0.999999572 |
| 40 | 0.838053936 | 1 to shown precision |
| 80 | 0.916221010 | 1 to shown precision |

The last column is an analytic-input calculation, not Monte Carlo. It shows why
a measured ratio near one half at `log t=10` would not contradict `A=1/12`.

## Independent exact finite-t calibration formulas

### Real two-factor calibrations

For a real `2x2` first factor, let `S=||A_1||_F^2~Gamma(2,2)`. Its Wishart
eigenvalues can be parameterized as `S(1+u)/2,S(1-u)/2`. The angular density is
`u/sqrt(1-u^2)`, so `v=sqrt(1-u^2)` is uniform on `[0,1]`, independently of S.
After integrating the last real matrix with p rows, the kernel is

\[
(1+2tS+t^2S^2v^2)^{-p/2}.
\]

For p=2 (widths (2,2,2)), integration over v gives the positive exact integral

\[
Z(t)=\frac1{4t}\int_0^\infty
\frac{e^{-s/2}\arctan\{ts/\sqrt{1+2ts}\}}{\sqrt{1+2ts}}\,ds.
\]

The implementation integrates `t^(3/2)Z(t)` at large t to avoid a misleading
tiny absolute quadrature error on an extremely small integral.

For p=3 (widths (2,2,3)),

\[
Z(t)=E\frac1{(1+tS)(1+2tS)}=\frac{I(t)-I(2t)}{4t},\qquad
I(a)=\frac{e^{1/(2a)}E_1(1/(2a))}{a}.
\]

Its full leading-power logarithm is

\[
Z(t)=\frac{\log t-\gamma_E}{8t^2}+\text{smaller-power corrections}.
\]

The same calculation yields the distributional representation
`g=S(U+2V)`, where `U,V` are independent mean-one exponentials independent of S.
Consequently its exact Frobenius CDF is

\[
F_F(\varepsilon)=1-\varepsilon K_2(\sqrt\varepsilon)
+\varepsilon K_2(\sqrt{2\varepsilon}).
\]

The CDF code automatically raises working precision at small epsilon to handle
the cancellation of terms of order one. This exact reference does not use the
proposed high-q constants.

### Complex size-two chains

For any positive N and t, the exact finite-t reference is the determinant

\[
Z(t)=\det_{i,j=0,1}\left[
t^{-i-1}G_{1,N}^{N,1}\left(t^{-1}\;\middle|\;{-i\atop j,0,\ldots,0}\right)
\right].
\]

This follows directly by integrating the exact polynomial-ensemble weights
against `exp(-tx)`. It is evaluated with arbitrary-precision Meijer-G functions,
separately from the principal-residue asymptotic. It can detect both finite-t
remainder and Monte Carlo disagreement. It remains an analytic-input comparison,
not an independent sampling estimator.

## Executable checks and APIs

Run `python analytic_reference.py`. Checks cover the independently known real
constants, the conjecture formula's stated target values (only arithmetic, not
conjecture validation), scalar complex chains, the exact one-matrix transform,
the full four-factor polynomial, positive real quadrature versus its E1 form,
and 50- versus 80-digit Meijer-G evaluations.

- `shell_data(widths)` returns C, q and shell occupation data.
- `leading_constants(widths)` returns separately named complex constants and
  `conjectured_real_*` values with a proved-family flag.
- `complex_log_polynomial(widths, observable)` returns a symbolic exact polynomial.
- `complex_log_polynomial_value(widths, log_t)` evaluates it at high precision.
- `exact_complex_square2_laplace(N,t)` evaluates the exact finite-t comparator.
- `exact_real_twofactor_laplace(widths,t)` handles the two real calibration cases.
- `exact_real_223_cdf(epsilon)` supplies the exact real q=1 Frobenius CDF.

The package's empirical report should distinguish MC uncertainty, importance
weight degeneracy, floating-point/quad error, and finite-t asymptotic bias.
Agreement with any comparator does not prove asymptotic equivalence or Conjecture E.

## Supplement: Frobenius-CDF logarithmic comparisons

For a leading-power CDF expansion `F(e)=e^lambda Q(log(1/e))`, integration by
parts and `u=tx` give the finite-polynomial coefficient relation

\[
P_{\rm Lap}(L)=\Gamma(\lambda+1-D_L)Q(L),\qquad
Q(L)=\Gamma(\lambda+1-D_L)^{-1}P_{\rm Lap}(L).
\]

`complex_frobenius_log_polynomial` implements the inverse using the Taylor
coefficients of `1/Gamma(lambda+1-u)` and exact symbolic polynomial derivatives.
The forward transform reproduces the Laplace polynomial exactly. This is a
**derived Frobenius comparator**, distinct from Theorem B's directly stated
operator-norm CDF polynomial. A general Tauberian theorem alone does not justify
transferring every lower logarithmic coefficient or a quantitative remainder.

For the complex four-factor size-two benchmark let
`z=L+11/6-4 gamma_E`. The result is

\[
\begin{aligned}
Q(L)={}&\frac{z^4}{72}
+\left(\frac{49}{432}-\frac{\pi^2}{72}\right)z^2
+\left(\frac{\zeta(3)}{18}+\frac{251}{1944}\right)z\\
&+\frac{1729}{10368}-\frac{49\pi^2}{2592}+\frac{7\pi^4}{240}.
\end{aligned}
\]

The leading coefficient is `1/72`. The ratios `Q(L)/(L^4/72)` at
`L=10,20,40,60,80,100` are respectively
`0.8412297103, 0.9071660328, 0.9525462787, 0.9682878229,
0.9762041033, 0.9809614650`.

An additional independent check integrates the principal **joint-density**
Mellin polynomials directly over `u,v>=0`, `u+v<=1`, with Vandermonde factor
`v-u`. It evaluates logarithmic moments as mixed derivatives of
`Gamma(a+1)Gamma(b+1)/Gamma(a+b+3)` and does not use the Laplace polynomial.
At `L=12` it agrees with the transformed comparator to better than forty digits.
This is implemented in
`complex_square2_frobenius_polynomial_simplex_value(N,L)` and included in the
executable reference checks. It verifies the coefficient algebra within the
same analytic polynomial-ensemble input; it is not numerical certification of
Theorem B or of a new remainder theorem.

For the independently soluble real `(2,2,3)` calibration, the Bessel expansion
and the coefficient transform both give

\[
F_F(\varepsilon)=\frac{\varepsilon^2}{16}
\left[\log(1/\varepsilon)+\frac32-2\gamma_E\right]
+O\bigl(\varepsilon^3\log(1/\varepsilon)\bigr).
\]

Use `real_223_cdf_asymptotic(epsilon)` for this full leading-power expression;
it is checked against `exact_real_223_cdf` at `epsilon=10^-20`. No real high-q
coefficient is assumed or fitted in any of these calculations.
