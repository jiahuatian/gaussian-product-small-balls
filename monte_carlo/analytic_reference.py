"""Independent analytic reference calculations for the QIP Monte Carlo audit.

These routines are comparison targets, NEVER part of a sampling weight.  All
Laplace transforms use Z(t)=E exp(-t*||A_N...A_1||_F**2), with real N(0,1)
entries or complex (X+iY)/sqrt(2).  The real `conjectured_*` outputs remain
conjectural outside the paper's proved families.

Dependencies: mpmath, sympy, numpy.  Run this file for independent normalization
and polynomial self-checks.  Symbolic polynomial generation is cached and intended
for small n (the benchmarks have n=1 or 2).
"""
from functools import lru_cache
from math import factorial, comb
import json
import mpmath as mp
import numpy as np
import sympy as sp

LOG = sp.Symbol("L", real=True)


def shell_data(widths):
    """Compute shell invariants without looking up benchmark constants."""
    d = tuple(int(x) for x in widths)
    if len(d) < 2 or min(d) < 1:
        raise ValueError("At least two positive integer widths are required")
    ds = sorted(d)
    n, N = ds[0], len(ds) - 1
    nu = tuple(x - n for x in ds[1:])
    levels = sorted((v+k, j, k) for j, v in enumerate(nu) for k in range(n))
    chosen = levels[:n]
    LF = chosen[-1][0]
    mu = sum(v <= LF for v in nu)
    h = sum(level == LF for level, _, _ in chosen)
    e = [0]*N
    occupations = {}
    for level, j, _ in chosen:
        e[j] += 1
        occupations[level] = occupations.get(level, 0)+1
    C = n*(n+1)//2 + sum(level for level, _, _ in chosen)
    return dict(widths=d, n=n, N=N, nu=nu, L_F=LF, mu=mu, h=h,
                C=C, q=h*(mu-h), multiplicity=1+h*(mu-h),
                theta=comb(mu,h), occupation=e, occupied_levels=occupations)


def leading_constants(widths, dps=60):
    """Complex theorem constants and separately labelled real conjecture target.

    kappa_* are CDF coefficients. laplace_* multiply t^-lambda (log t)^q.
    The conjecture value is independently assembled from the particle Gamma
    product, not hard-coded into any estimator.
    """
    s = shell_data(widths)
    n,N,nu,e,C = (s[k] for k in ("n","N","nu","occupation","C"))
    with mp.workdps(dps):
        B = mp.fprod(mp.factorial(j)/mp.factorial(s["mu"]-1-j)
                     for j in range(s["h"]))
        G = mp.fprod(mp.gamma(i+v+1) for i in range(n) for v in nu)
        Sn = mp.fprod(mp.factorial(i) for i in range(n))
        occ = s["occupied_levels"]
        V = mp.fprod(mp.mpf(M-L)**(occ[L]*occ[M])
                     for L in occ for M in occ if L<M)
        A = B*V/(G*Sn)
        kop = B*V/G
        for L, count in occ.items():
            R = mp.fprod(mp.gamma(v-L) for v in nu if v>L)
            A *= (R*mp.gamma(L+1))**count
            kop *= (R*mp.gamma(L+1)/mp.gamma(L+n+1))**count
        # Particle expression with beta=1, with zero-gap pairs omitted.
        S1 = lambda k: mp.fprod(mp.gamma(mp.mpf(j)/2) for j in range(1,k+1))
        value = mp.power(2,-mp.mpf(N*C)/2)/mp.gamma(mp.mpf(C)/2+1)
        value *= mp.fprod(S1(j) for j in e)/S1(n)
        f = [nu[j]+e[j] for j in range(N)]
        for j in range(N):
            for k in range(e[j]):
                energy=nu[j]+k
                for i in range(N):
                    if i != j and f[i] != energy:
                        gap=f[i]-energy
                        if gap < 0: raise ArithmeticError("Nonminimal occupation")
                        value *= mp.gamma(mp.mpf(gap)/2)
        value /= mp.fprod(mp.gamma(mp.mpf(nu[j]+u)/2)
                         for j in range(N) for u in range(e[j]+1,n+1))
        kr = B*value
        return {**s, "B":+B, "complex_laplace_A":+A,
                "complex_cdf_frobenius_kappa":+(A/mp.gamma(C+1)),
                "complex_cdf_operator_kappa":+kop,
                "conjectured_real_cdf_kappa":+kr,
                "conjectured_real_laplace_A":+(kr*mp.gamma(mp.mpf(C)/2+1)),
                "real_formula_proved_family":s["q"]<=1 or n==1 or N==2}


def _exp_series(log_coefficients, order):
    """Taylor coefficients of exp(sum_{k>=1} a[k]u^k), exact symbolic."""
    out=[sp.Integer(1)]
    for m in range(1,order+1):
        out.append(sp.expand(sum(k*log_coefficients[k]*out[m-k]
                                 for k in range(1,m+1))/m))
    return out


@lru_cache(maxsize=128)
def complex_log_polynomial(widths, observable="laplace"):
    """Full leading-power logarithmic polynomial, via independent residues.

    Supported observables are 'laplace' and 'operator'.  This computes the
    determinant of residue entries directly, without assembling the paper's
    Plucker minors or inserting its leading constant.  It is an implementation
    of the analytic polynomial-ensemble/Mellin input, not independent proof of it.
    """
    if observable not in ("laplace","operator"):
        raise ValueError("observable must be laplace or operator")
    s=shell_data(widths)
    n,N,nu=s["n"],s["N"],s["nu"]
    columns=[]
    for level,count in sorted(s["occupied_levels"].items()):
        mu=sum(v<=level for v in nu)
        offsets=[v-level for v in nu if v>level]
        R=sp.prod(sp.gamma(v) for v in offsets)
        for j in range(count):
            order=mu-j-1
            col=[]
            for i in range(n):
                beta=i+level+1
                aa={}
                for k in range(1,order+1):
                    base=(-mu*sp.EulerGamma if k==1 else
                          mu*(-1)**k*sp.zeta(k)/k)
                    base += sum(sp.polygamma(k-1,v)/sp.factorial(k) for v in offsets)
                    if observable=="laplace":
                        base += (-1)**k*sp.polygamma(k-1,beta)/sp.factorial(k)
                    else:
                        base += sp.Rational(1,k*beta**k)
                    aa[k]=sp.expand(base+(LOG if k==1 else 0))
                coeff=_exp_series(aa,order)[order]
                pref=sp.gamma(beta) if observable=="laplace" else sp.Rational(1,beta)
                col.append(sp.expand(R*pref*coeff))
            columns.append(col)
    D=sp.prod(sp.factorial(i) for i in range(n))*sp.prod(
        sp.gamma(i+v+1) for i in range(n) for v in nu)
    result=sp.Poly(sp.expand(sp.Matrix(n,n,lambda i,j:columns[j][i]).det()/D),LOG)
    if result.degree()!=s["q"]:
        raise ArithmeticError("Cancellation gave an unexpected degree")
    return result.as_expr()


def complex_polynomial_coefficients(widths, observable="laplace", dps=60):
    """Ascending-order coefficients as high-precision mpmath numbers."""
    p=sp.Poly(complex_log_polynomial(tuple(widths),observable),LOG)
    with mp.workdps(dps):
        return [mp.mpf(str(sp.N(p.nth(i),dps))) for i in range(p.degree()+1)]


def complex_log_polynomial_value(widths, log_t, observable="laplace", dps=60):
    with mp.workdps(dps):
        coeff=complex_polynomial_coefficients(tuple(widths),observable,dps)
        return +mp.polyval(list(reversed(coeff)),mp.mpf(log_t))


def complex_laplace_asymptotic(widths,t,full_polynomial=True,dps=60):
    """Analytic asymptotic comparison, NOT the exact finite-t transform."""
    with mp.workdps(dps):
        t=mp.mpf(t);s=shell_data(widths)
        if full_polynomial:
            poly=complex_log_polynomial_value(tuple(widths),mp.log(t),dps=dps)
        else:
            poly=leading_constants(widths,dps)["complex_laplace_A"]*mp.log(t)**s["q"]
        return +(poly*t**(-s["C"]))


def _laplace_to_cdf_polynomial(polynomial, exponent):
    """Invert Gamma(exponent+1-D) on a finite polynomial, exactly.

    This is a finite coefficient transform. Existence and a remainder bound for
    a complete CDF expansion do not follow from an unquantified Tauberian theorem
    alone; the caller must keep that analytic qualification separate.
    """
    p=sp.Poly(sp.expand(polynomial),LOG)
    degree=p.degree();beta=sp.sympify(exponent)+1
    # Taylor series of 1/Gamma(beta-u), factored by 1/Gamma(beta).
    log_coeff={k:-(-1)**k*sp.polygamma(k-1,beta)/sp.factorial(k)
               for k in range(1,degree+1)}
    inverse_coeff=_exp_series(log_coeff,degree)
    return sp.expand(sum(inverse_coeff[k]*sp.diff(p.as_expr(),LOG,k)
                         for k in range(degree+1))/sp.gamma(beta))


@lru_cache(maxsize=128)
def complex_frobenius_log_polynomial(widths):
    """Analytically derived Frobenius-CDF leading-power polynomial.

    If F(e)=e^C Q(log(1/e)) plus lower-power terms, integration by parts gives
    Z(t)=t^-C Gamma(C+1-D_L)Q(L), L=log(t).  This function inverts that finite
    polynomial operator on the independently computed Laplace polynomial.

    Theorem B itself states a full polynomial for Laplace and operator-norm CDF,
    not Frobenius CDF. This output is therefore labelled a derived comparator;
    it is not a newly certified theorem or a general Tauberian remainder claim.
    """
    widths=tuple(widths)
    return _laplace_to_cdf_polynomial(complex_log_polynomial(widths),shell_data(widths)['C'])


def complex_frobenius_log_polynomial_value(widths, log_inverse_epsilon, dps=60):
    with mp.workdps(dps):
        p=sp.Poly(complex_frobenius_log_polynomial(tuple(widths)),LOG)
        coeff=[mp.mpf(str(sp.N(p.nth(i),dps))) for i in range(p.degree()+1)]
        return +mp.polyval(list(reversed(coeff)),mp.mpf(log_inverse_epsilon))


def complex_frobenius_cdf_asymptotic(widths,epsilon,full_polynomial=True,dps=60):
    """Derived asymptotic comparator, NOT an exact finite-epsilon CDF."""
    with mp.workdps(dps):
        e=mp.mpf(epsilon)
        if e<=0:raise ValueError('epsilon must be positive')
        s=shell_data(widths);L=-mp.log(e)
        if full_polynomial:
            poly=complex_frobenius_log_polynomial_value(tuple(widths),L,dps)
        else:
            poly=leading_constants(widths,dps)['complex_cdf_frobenius_kappa']*L**s['q']
        return +(e**s['C']*poly)


def complex_square2_frobenius_polynomial_simplex_value(N,log_inverse_epsilon,dps=60):
    """Independent direct-simplex check of the CDF polynomial for N>=2.

    Insert the principal Mellin polynomials f_0(L-log u),f_1(L-log v) into
    (1/2) integral_{u+v<=1}(v-u)[f0(u)f1(v)-f1(u)f0(v)]du dv.
    Log moments are mixed derivatives of Gamma(a+1)Gamma(b+1)/Gamma(a+b+3).
    This does not invert the Laplace polynomial or insert its coefficients.
    """
    if N<2:raise ValueError('This principal-shell check requires N>=2')
    with mp.workdps(dps):
        L=mp.mpf(log_inverse_epsilon)
        aa={k:(-N*sp.EulerGamma if k==1 else N*(-1)**k*sp.zeta(k)/k)
            for k in range(1,N)}
        aa[1]+=LOG
        series=_exp_series(aa,N-1)
        polys=[series[N-1],series[N-2]]
        def coefficient(p,j):
            symbolic=sp.diff(p,LOG,j)/sp.factorial(j)
            return mp.mpf(str(sp.N(symbolic.subs(LOG,sp.Float(str(L),dps)),dps)))
        ac=[coefficient(polys[0],j) for j in range(N)]
        bc=[coefficient(polys[1],k) for k in range(N-1)]
        moment=lambda a,b:mp.gamma(a+1)*mp.gamma(b+1)/mp.gamma(a+b+3)
        answer=mp.mpf(0)
        for j in range(N):
            for k in range(N-1):
                if j==k:continue
                difference=(-1)**(j+k)*(mp.diff(moment,(0,1),(j,k))-mp.diff(moment,(1,0),(j,k)))
                answer+=ac[j]*bc[k]*difference
        return +answer


def real_223_frobenius_log_polynomial():
    """Full leading-power polynomial, independently checked by exact Bessel CDF.

    F(e)=e^2*(log(1/e)+3/2-2*gamma_E)/16 + O(e^3*log(1/e)).
    The qualification is specific to this exactly soluble q=1 calibration.
    """
    return (LOG+sp.Rational(3,2)-2*sp.EulerGamma)/16


def real_223_cdf_asymptotic(epsilon,dps=60):
    with mp.workdps(dps):
        e=mp.mpf(epsilon)
        if e<=0:raise ValueError('epsilon must be positive')
        return +(e**2*(-mp.log(e)+mp.mpf('1.5')-2*mp.euler)/16)


def exact_real_twofactor_laplace(widths,t,dps=60):
    """Exact positive-integral references for d=(2,2,2) and permutations of (2,2,3).

    S=||A_1||_F^2~Gamma(2,2) for A_1 a 2x2 real Gaussian matrix.
    Conditional Wishart angular variable v is Uniform[0,1], and the final-layer
    kernel is (1+2tS+t^2S^2 v^2)^(-p/2), p=2 or 3. Integration over v yields
    these one-dimensional formulas. No conjectured asymptotic is used.
    """
    widths=tuple(sorted(widths))
    if widths not in [(2,2,2),(2,2,3)]:
        raise ValueError("Only the two independently tractable benchmarks supported")
    with mp.workdps(dps):
        t=mp.mpf(t)
        if t<0:raise ValueError("t must be nonnegative")
        if t==0:return mp.mpf(1)
        if widths==(2,2,3):
            I=lambda a:mp.exp(1/(2*a))*mp.e1(1/(2*a))/a
            # Avoid catastrophic cancellation of large asymptotic E1 factors for tiny t.
            if t>mp.mpf('.01'):
                return +((I(t)-I(2*t))/(4*t))
            return +mp.quad(lambda s:s*mp.exp(-s/2)/(4*(1+t*s)*(1+2*t*s)),[0,1,mp.inf])
        # Integrate t^(3/2) Z(t), an order-one quantity at large t, to avoid
        # falsely accepting tiny absolute quadrature errors as small relative errors.
        f=lambda s:mp.exp(-s/2)*mp.atan(t*s/mp.sqrt(1+2*t*s))/(4*mp.sqrt(1/t+2*s))
        if t>1:return +(mp.quad(f,[0,1/t,1,mp.inf])*t**(-mp.mpf('1.5')))
        return +(mp.quad(f,[0,1,mp.inf])*t**(-mp.mpf('1.5')))


def exact_real_223_cdf(epsilon,dps=70):
    """Exact real (2,2,3) Frobenius CDF, from the same angular integration.

    In law g=S*(U+2V), S~Gamma(2,2), U,V~Exp(1), independent.  Hence
    F(e)=1-e*K_2(sqrt(e))+e*K_2(sqrt(2e)).  Very small e causes cancellation;
    working precision is increased by 2*(-log10(e)) digits automatically.
    """
    e=mp.mpf(epsilon)
    if e<0:raise ValueError("epsilon must be nonnegative")
    if e==0:return mp.mpf(0)
    needed=dps+max(0,int(-2*mp.log10(e))+20)
    with mp.workdps(needed):
        return +(1-e*mp.besselk(2,mp.sqrt(e))+e*mp.besselk(2,mp.sqrt(2*e)))


def exact_complex_square2_laplace(N,t,dps=70):
    """Exact finite-t 2x2 complex-product determinant evaluated with mpmath.

    This is an analytic-input comparator, not Monte Carlo. The entry is
    t^(-i-1) G_{1,N}^{N,1}(1/t | -i ; j,0,...,0), i,j in {0,1}.
    It follows by directly integrating the exact polynomial-ensemble weights
    against exp(-tx).  No asymptotic coefficient is inserted. The determinant
    normalization is exactly one for square n=2, any N>=1.
    """
    if N<1:raise ValueError("N must be positive")
    with mp.workdps(dps):
        t=mp.mpf(t)
        if t<0:raise ValueError("t must be nonnegative")
        if t==0:return mp.mpf(1)
        matrix=mp.matrix(2)
        for i in range(2):
            for j in range(2):
                matrix[i,j]=t**(-i-1)*mp.meijerg([[-i],[]],[[j]+[0]*(N-1),[]],1/t)
        return +mp.det(matrix)


def benchmark_table(dps=60):
    specs=[("real",(2,2,2)),("real",(2,2,3)),("real",(2,2,2,2)),
           ("real",(2,2,2,2,2)),("complex",(2,2,2,2,2))]
    out=[]
    for field,widths in specs:
        x=leading_constants(widths,dps)
        out.append(dict(field=field,widths=widths,C=x['C'],q=x['q'],
            exponent=x['C']/2 if field=='real' else x['C'],
            cdf_kappa=str(x['conjectured_real_cdf_kappa'] if field=='real' else x['complex_cdf_frobenius_kappa']),
            laplace_A=str(x['conjectured_real_laplace_A'] if field=='real' else x['complex_laplace_A']),
            coefficient_status='proved_family' if field=='complex' or x['real_formula_proved_family'] else 'conjecture_E'))
    return out


def self_checks():
    """Independent arithmetic/reference checks; never a theorem certificate."""
    checks={}
    expected=[((2,2,2),mp.pi/6),((2,2,3),mp.mpf(1)/16),
              ((2,2,2,2),mp.sqrt(2*mp.pi)/48),((2,2,2,2,2),mp.mpf(1)/576)]
    for d,target in expected:
        got=leading_constants(d)["conjectured_real_cdf_kappa"]
        checks['real_constant_'+str(d)]=abs(float(got/target)-1)<1e-14
    checks['one_complex_2x3_matrix']=sp.simplify(complex_log_polynomial((2,3))-1)==0
    checks['one_complex_scalar']=sp.simplify(complex_log_polynomial((1,1))-1)==0
    checks['two_complex_scalars']=sp.simplify(complex_log_polynomial((1,1,1))-(LOG-sp.EulerGamma))==0
    checks['two_complex_2x2']=sp.simplify(complex_log_polynomial((2,2,2))-1)==0
    x=LOG-3*sp.EulerGamma
    expected_poly=x**4/sp.Integer(12)+sp.zeta(3)*x+25*sp.pi**4/144
    checks['four_complex_2x2_full_polynomial']=sp.simplify(complex_log_polynomial((2,2,2,2,2))-expected_poly)==0
    checks['four_complex_2x2_leading_A']=abs(float(leading_constants((2,2,2,2,2))['complex_laplace_A'])*12-1)<1e-14
    Q=complex_frobenius_log_polynomial((2,2,2,2,2))
    checks['complex_frobenius_leading_1_over_72']=sp.Poly(Q,LOG).LC()==sp.Rational(1,72)
    # Apply the forward Gamma(4-D) operator; no leading coefficient is fitted.
    forward_logs={k:(-1)**k*sp.polygamma(k-1,4)/sp.factorial(k) for k in range(1,5)}
    forward=_exp_series(forward_logs,4)
    restored=sp.expand(sp.gamma(4)*sum(forward[k]*sp.diff(Q,LOG,k) for k in range(5)))
    checks['complex_frobenius_laplace_transform_roundtrip']=sp.simplify(restored-expected_poly)==0
    checks['real_q1_cdf_coefficient_transform']=sp.simplify(
        _laplace_to_cdf_polynomial((LOG-sp.EulerGamma)/8,2)-real_223_frobenius_log_polynomial())==0
    # Positive quadrature vs closed form for the q=1 real calibration.
    with mp.workdps(50):
        t=mp.mpf(11)
        quad=mp.quad(lambda s:s*mp.exp(-s/2)/(4*(1+t*s)*(1+2*t*s)),[0,1,mp.inf])
        closed=exact_real_twofactor_laplace((2,2,3),t,dps=50)
        checks['real_q1_closed_vs_positive_integral']=abs(quad/closed-1)<mp.mpf('1e-45')
        # Scalar two-factor complex exact Laplace has the predicted full leading polynomial.
        t=mp.mpf('1e15');exact=mp.exp(1/t)*mp.e1(1/t)/t
        approx=complex_laplace_asymptotic((1,1,1),t)
        checks['complex_scalar_exact_vs_residue']=abs(exact/approx-1)<mp.mpf('1e-13')
        t=mp.mpf(7)
        checks['one_complex_2x2_exact_meijerg']=abs(exact_complex_square2_laplace(1,t)/(1+t)**(-4)-1)<mp.mpf('1e-45')
        a=exact_complex_square2_laplace(4,mp.mpf('1e8'),dps=50)
        b=exact_complex_square2_laplace(4,mp.mpf('1e8'),dps=80)
        checks['complex_exact_precision_50_vs_80']=abs(a/b-1)<mp.mpf('1e-40')
        e=mp.mpf('1e-20')
        checks['real_q1_full_cdf_vs_exact_bessel']=abs(exact_real_223_cdf(e)/real_223_cdf_asymptotic(e)-1)<mp.mpf('1e-18')
        direct=complex_square2_frobenius_polynomial_simplex_value(4,12,dps=50)
        transformed=complex_frobenius_log_polynomial_value((2,2,2,2,2),12,dps=50)
        checks['complex_frobenius_direct_simplex_vs_transform']=abs(direct/transformed-1)<mp.mpf('1e-40')
    return checks


if __name__=='__main__':
    checks=self_checks()
    result={'checks':checks,'all_checks_passed':all(checks.values()),'benchmarks':benchmark_table(),
            'complex_four_2x2_laplace_polynomial':str(complex_log_polynomial((2,2,2,2,2))),
            'complex_four_2x2_operator_polynomial':str(complex_log_polynomial((2,2,2,2,2),'operator')),
            'complex_four_2x2_frobenius_derived_polynomial':str(complex_frobenius_log_polynomial((2,2,2,2,2))),
            'qualification':'Arithmetic/reference checks only; not mathematical certification.'}
    print(json.dumps(result,indent=2))
    if not all(checks.values()):raise SystemExit(1)
