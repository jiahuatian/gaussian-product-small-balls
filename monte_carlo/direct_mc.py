#!/usr/bin/env python3
"""Independent Gaussian-product direct and final-layer conditional Monte Carlo.

Target: Z(t) = E[exp(-t ||A_N ... A_1||_F**2)].
Real entries have variance 1. Complex entries have E|z|**2 = 1.
No small-ball theorem, asymptotic exponent or conjectured amplitude enters sampling.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import gammainc


class _Moments:
    def __init__(self, dimension: int):
        self.n = 0
        self.mean = np.zeros(dimension, dtype=np.float64)
        self.m2 = np.zeros((dimension, dimension), dtype=np.float64)
        self.positive = np.zeros(dimension, dtype=np.int64)

    def update(self, values: np.ndarray):
        count = len(values)
        batch_mean = values.mean(axis=0)
        # Make truly constant columns exact (e.g. N=1 conditional integration).
        constant = np.ptp(values, axis=0) == 0
        batch_mean[constant] = values[0, constant]
        centered = values - batch_mean
        batch_m2 = centered.T @ centered
        new_n = self.n + count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + np.outer(delta, delta) * (self.n * count / new_n)
        self.mean += delta * (count / new_n)
        self.n = new_n
        self.positive += np.count_nonzero(values, axis=0)

    def report(self, *, bernoulli: bool = False, nonnegative: bool = True):
        mean = self.mean
        # Chan's centered online covariance avoids subtracting nearly equal
        # uncentered moments for low-variance conditional estimates.
        covariance = self.m2 / (self.n - 1)
        diagonal = np.maximum(np.diag(covariance), 0.0)
        covariance[np.diag_indices_from(covariance)] = diagonal
        se = np.sqrt(diagonal / self.n)
        low, high = mean - 1.959963984540054 * se, mean + 1.959963984540054 * se
        if nonnegative:
            low = np.maximum(low, 0.0)
        result = {
            "n": self.n,
            "mean": mean.tolist(),
            "sample_variance": diagonal.tolist(),
            "standard_error": se.tolist(),
            "normal_95_interval": np.stack([low, high], axis=1).tolist(),
            "sum": (self.n * mean).tolist(),
            "sum_squares": (np.diag(self.m2) + self.n * mean**2).tolist(),
            "covariance_of_mean": (covariance / self.n).tolist(),
            "nonzero_contributions": self.positive.tolist(),
            "all_zero": (self.positive == 0).tolist(),
        }
        if nonnegative:
            total2 = np.diag(self.m2) + self.n * mean**2
            ess = np.divide((self.n*mean)**2, total2, out=np.zeros_like(mean), where=total2 > 0)
            result["contribution_ess"] = np.minimum(ess, self.n).tolist()
            result["ess_definition"] = "(sum contributions)^2 / sum(contributions^2); not MCMC ESS"
        if bernoulli:
            z2 = 1.959963984540054**2
            center = (mean + z2 / (2 * self.n)) / (1 + z2 / self.n)
            half = np.sqrt(z2 * mean * (1 - mean) / self.n + z2**2 / (4 * self.n**2)) / (1 + z2 / self.n)
            result["wilson_95_interval"] = np.stack([np.maximum(0, center - half), np.minimum(1, center + half)], axis=1).tolist()
            result["event_count"] = np.rint(self.n * mean).astype(np.int64).tolist()
        return result


def gaussian_matrices(rng: np.random.Generator, count: int, rows: int, cols: int, beta: int):
    real = rng.standard_normal((count, rows, cols))
    if beta == 1:
        return real
    return (real + 1j * rng.standard_normal((count, rows, cols))) / np.sqrt(2.0)


def conditional_laplace(squared_singular_values: np.ndarray, final_rows: int, beta: int, t_values):
    """Integrate the actual final Gaussian matrix, exactly in floating point.

    beta=1: det(I+2t B*B)^(-m/2)
    beta=2: det(I+t B*B)^(-m)
    SVD avoids cancellation in a computed 2x2 Gram determinant.
    """
    t = np.asarray(t_values, dtype=np.float64)
    log_det = np.log1p((2.0 / beta) * squared_singular_values[:, None, :] * t[None, :, None]).sum(axis=2)
    return np.exp(-0.5 * beta * final_rows * log_det)


def _two_exponential_cdf(x: np.ndarray, a: np.ndarray, b: np.ndarray):
    """CDF of independent exponentials with means a>=b>=0.

    This evaluates the exact hypoexponential CDF. A convergent small-argument
    series avoids catastrophic cancellation; its 14 terms have negligible
    truncation when max(x/a,x/b)<0.1. Equal-scale values use Gamma(2).
    """
    x, a, b = np.broadcast_arrays(np.asarray(x, float), np.asarray(a, float), np.asarray(b, float))
    out = np.zeros_like(x)
    zero = a == 0
    out[zero] = (x[zero] >= 0).astype(float)
    one = (a > 0) & (b == 0)
    out[one] = -np.expm1(-x[one] / a[one])
    active = (a > 0) & (b > 0)
    if not np.any(active):
        return out
    u = x[active] / a[active]
    ratio = b[active] / a[active]
    with np.errstate(over="ignore", divide="ignore"):
        v = x[active] / b[active]
    val = np.empty_like(u)
    small = v < 0.1
    if np.any(small):
        r, vv = ratio[small], v[small]
        # F = u*v sum_{j>=0} (-v)^j (1+r+...+r^j)/(j+2)!
        geom = np.ones_like(vv)
        rpower = np.ones_like(vv)
        factor = np.full_like(vv, 0.5)
        series = factor.copy()
        for j in range(1, 14):
            rpower *= r
            geom += rpower
            factor *= -vv / (j + 2)
            series += factor * geom
        val[small] = u[small] * vv * series
    equal = (~small) & (ratio == 1)
    val[equal] = gammainc(2.0, u[equal])
    close = (~small) & (~equal) & (ratio > 0.999)
    if np.any(close):
        d = (1 - ratio[close]) / (1 + ratio[close])
        um = 2 * u[close] / (1 + ratio[close])
        w = um / (1 - d*d)
        y = w * d
        result = np.ones_like(w)
        safe = w < 700
        yy = y[safe]
        sinhc = np.ones_like(yy)
        nonzero = yy != 0
        sinhc[nonzero] = np.sinh(yy[nonzero]) / yy[nonzero]
        result[safe] = -np.expm1(-w[safe]) - np.exp(-w[safe]) * (2*np.sinh(yy/2)**2 + w[safe]*sinhc)
        val[close] = result
    normal = (~small) & (~equal) & (~close)
    val[normal] = (-np.expm1(-u[normal]) + ratio[normal] * np.expm1(-v[normal])) / (1 - ratio[normal])
    out[active] = np.clip(val, 0.0, 1.0)
    return out


def conditional_frobenius_cdf(squared_singular_values, final_rows, beta, eps_values):
    """Conditional CDF without numerical quadrature for supported cases.

    Rank 1: Gamma(beta*m/2, scale=2*lambda/beta).
    Real final m=2, rank 2: a sum of two exponential variables.
    Other shapes return None, explicitly declining an uncontrolled quadrature.
    """
    lam = np.asarray(squared_singular_values)
    eps = np.asarray(eps_values)[None, :]
    if lam.shape[1] == 1:
        with np.errstate(divide="ignore", invalid="ignore"):
            z = beta * eps / (2 * lam[:, :1])
        return gammainc(beta * final_rows / 2, z)
    if beta == 1 and final_rows == 2 and lam.shape[1] == 2:
        return _two_exponential_cdf(eps, 2 * lam[:, :1], 2 * lam[:, 1:2])
    return None


def run_direct_case(widths, beta, t_values, eps_values, n, seed, batch_size=65536):
    """Generate direct products and Rao–Blackwell estimates on common prefixes.

    Returns JSON-ready summary dictionaries. All t and epsilon points share
    draws, and their covariance matrices are retained. Separate calls/seeds
    give independent replicates. CI/ESS diagnostics do not certify tail control.
    """
    widths = tuple(int(x) for x in widths)
    if len(widths) < 2 or min(widths) < 1 or beta not in (1, 2) or n < 2:
        raise ValueError("positive widths, beta in {1,2}, and n>=2 are required")
    t = np.atleast_1d(np.asarray(t_values, float))
    eps = np.atleast_1d(np.asarray(eps_values, float))
    if np.any(t < 0) or np.any(eps < 0) or np.any(~np.isfinite(t)) or np.any(~np.isfinite(eps)):
        raise ValueError("finite nonnegative t and epsilon values required")
    rng = np.random.default_rng(seed)
    blocks = {name: _Moments(len(t)) for name in ["direct_laplace", "conditional_laplace", "paired_laplace_difference"]}
    blocks.update({name: _Moments(len(eps)) for name in ["direct_frobenius_cdf", "direct_spectral_cdf", "conditional_frobenius_cdf", "paired_frobenius_cdf_difference"]})
    conditional_cdf_available = None
    start = time.perf_counter()
    timings = {k:0.0 for k in ['prefix_sampling','conditional_laplace','direct_laplace_after_prefix','spectral_and_cdf','accumulation']}
    for offset in range(0, n, batch_size):
        count = min(batch_size, n-offset)
        tick = time.perf_counter()
        if len(widths) == 2:
            prefix = np.broadcast_to(np.eye(widths[0]), (count, widths[0], widths[0]))
        else:
            prefix = gaussian_matrices(rng, count, widths[1], widths[0], beta)
            for layer in range(2, len(widths)-1):
                prefix = gaussian_matrices(rng, count, widths[layer], widths[layer-1], beta) @ prefix
        timings['prefix_sampling'] += time.perf_counter()-tick
        tick = time.perf_counter()
        # The exact rank is the smallest preceding width. Discard roundoff
        # singular values of algebraically rank-deficient rectangular products.
        singular = np.linalg.svd(prefix, compute_uv=False)[:, :min(widths[:-1])]
        conditional = conditional_laplace(singular**2, widths[-1], beta, t)
        timings['conditional_laplace'] += time.perf_counter()-tick
        tick = time.perf_counter()
        final = gaussian_matrices(rng, count, widths[-1], widths[-2], beta)
        product = final @ prefix
        frobenius_squared = np.sum(np.abs(product)**2, axis=(1, 2))
        direct = np.exp(-frobenius_squared[:, None] * t[None, :])
        timings['direct_laplace_after_prefix'] += time.perf_counter()-tick
        tick = time.perf_counter()
        spectral_squared = np.linalg.svd(product, compute_uv=False)[:, 0]**2
        f_cdf = (frobenius_squared[:, None] <= eps).astype(float)
        op_cdf = (spectral_squared[:, None] <= eps).astype(float)
        c_cdf = conditional_frobenius_cdf(singular**2, widths[-1], beta, eps_values)
        timings['spectral_and_cdf'] += time.perf_counter()-tick
        tick = time.perf_counter()
        blocks["direct_laplace"].update(direct)
        blocks["conditional_laplace"].update(conditional)
        blocks["paired_laplace_difference"].update(direct - conditional)
        blocks["direct_frobenius_cdf"].update(f_cdf)
        blocks["direct_spectral_cdf"].update(op_cdf)
        conditional_cdf_available = c_cdf is not None
        if conditional_cdf_available:
            blocks["conditional_frobenius_cdf"].update(c_cdf)
            blocks["paired_frobenius_cdf_difference"].update(f_cdf-c_cdf)
        timings['accumulation'] += time.perf_counter()-tick
    result = {
        "widths": list(widths), "beta": beta, "n": int(n), "seed": int(seed),
        "rng": "numpy.random.default_rng / PCG64", "batch_size": int(batch_size),
        "t_values": t.tolist(), "eps_values": eps.tolist(),
        "normalization": "Z(t)=E exp(-t ||P||_F^2); real N(0,1); complex (X+iY)/sqrt(2)",
        "runtime_seconds": time.perf_counter() - start,
        "component_timing_seconds": timings,
        "timing_scope": "Component costs in one common-prefix run; Laplace-only cost excludes CDF/spectral work and summary accumulation. Not independently timed standalone executables.",
        "conditional_cdf_available": conditional_cdf_available,
        "analytic_inputs": ["Gaussian final-layer integral", "SVD singular-value identity"],
        "notes": ["No importance sampling in this module.",
                  "Direct and conditional estimators share prefixes; use paired-difference standard errors for their comparison.",
                  "Zero contributions do not establish zero probability; use Wilson bounds for direct CDF events.",
                  "All normal confidence intervals are asymptotic Monte Carlo diagnostics, not rigorous coverage guarantees."],
    }
    for name, block in blocks.items():
        if block.n:
            result[name] = block.report(bernoulli=name in ["direct_frobenius_cdf", "direct_spectral_cdf"], nonnegative=not name.startswith("paired_"))
    return result


def validation_run(output):
    """Small reproducible estimator validation; records results without a theorem PASS."""
    cases = []
    for beta in [1, 2]:
        r = run_direct_case([2, 2], beta, [0.1, 1, 10], [0.1, 1, 10], 200000, 873100+beta)
        expected = (1+(2/beta)*np.asarray(r["t_values"]))**(-2*beta)
        r["independent_exact_laplace"] = expected.tolist()
        r["conditional_exact_max_absolute_error"] = float(np.max(np.abs(np.asarray(r["conditional_laplace"]["mean"])-expected)))
        cases.append(r)
    for beta, widths, seed in [(1,[2,2,2],873201),(1,[2,2,3],873202),(2,[2,2,2,2,2],873203)]:
        r = run_direct_case(widths,beta,[0.01,0.1,1,10],[0.01,0.1,1,10],200000,seed)
        difference = np.asarray(r["paired_laplace_difference"]["mean"])
        se = np.asarray(r["paired_laplace_difference"]["standard_error"])
        r["paired_laplace_z_score"] = (difference/se).tolist()
        if r["conditional_cdf_available"]:
            a=r["paired_frobenius_cdf_difference"]
            r["paired_cdf_z_score"]=(np.asarray(a["mean"])/np.asarray(a["standard_error"])).tolist()
        cases.append(r)
    Path(output).write_text(json.dumps({"description":"Independent validation of exact distributional identities; not asymptotic certification", "cases":cases},indent=2)+"\n")
    return cases


def independent_seed_validation(output):
    cases = []
    for beta, widths, seed in [(1,[2,2,2],873401),(1,[2,2,3],873403),(2,[2,2,2,2,2],873405)]:
        a = run_direct_case(widths,beta,[0.01,0.1,1,10],[0.01,0.1,1,10],400000,seed)
        b = run_direct_case(widths,beta,[0.01,0.1,1,10],[0.01,0.1,1,10],400000,seed+1)
        x, y = np.asarray(a['direct_laplace']['mean']), np.asarray(b['conditional_laplace']['mean'])
        se = np.hypot(a['direct_laplace']['standard_error'],b['conditional_laplace']['standard_error'])
        cases.append({'widths':widths,'beta':beta,'direct_seed':seed,'conditional_seed':seed+1,
                      'n_per_seed':400000,'t':[0.01,0.1,1,10],
                      'independent_difference_z_score':((x-y)/se).tolist(),
                      'direct_run':a,'independent_conditional_run':b})
    Path(output).write_text(json.dumps({'cases':cases},indent=2)+'\n')
    return cases


def cdf_roundoff_validation(output):
    import itertools
    import mpmath as mp
    mp.mp.dps = 150
    cases = []
    for ratio,u in itertools.product([1.,1-1e-12,.9995,.99,.5,1e-3,1e-12,1e-100],
                                      [1e-40,1e-12,1e-5,.01,.1,1.,10.,1000.]):
        a,b,x=mp.mpf(1),mp.mpf(float(ratio)),mp.mpf(float(u))
        ref=1-(1+x)*mp.exp(-x) if a==b else 1-(a*mp.exp(-x/a)-b*mp.exp(-x/b))/(a-b)
        value=float(_two_exponential_cdf(np.array([u]),np.array([1.]),np.array([ratio]))[0])
        error=abs(mp.mpf(value)-ref)
        cases.append({'ratio':ratio,'x_over_a':u,'reference':float(ref),'value':value,
                      'relative_error':float(error/ref) if ref else float(error)})
    Path(output).write_text(json.dumps({'mpmath_digits':150,'cases':cases},indent=2)+'\n')
    return cases


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", metavar="OUTPUT_JSON")
    parser.add_argument("--validate-independent", metavar="OUTPUT_JSON")
    parser.add_argument("--validate-cdf", metavar="OUTPUT_JSON")
    parser.add_argument("--widths", nargs="+", type=int, default=[2,2,2])
    parser.add_argument("--beta", type=int, choices=[1,2], default=1)
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=873301)
    parser.add_argument("--t", nargs="+", type=float, default=[0.01,0.1,1,10,100])
    parser.add_argument("--eps", nargs="+", type=float, default=[0.001,0.01,0.1,1])
    parser.add_argument("--output", default="direct_results.json")
    args=parser.parse_args()
    if args.validate:
        cases=validation_run(args.validate)
        print(json.dumps([{k:r[k] for k in ["widths","beta","runtime_seconds","conditional_exact_max_absolute_error","paired_laplace_z_score","paired_cdf_z_score"] if k in r} for r in cases],indent=2))
    elif args.validate_independent:
        cases=independent_seed_validation(args.validate_independent)
        print(json.dumps([{k:r[k] for k in ['widths','beta','independent_difference_z_score']} for r in cases],indent=2))
    elif args.validate_cdf:
        cases=cdf_roundoff_validation(args.validate_cdf)
        print(json.dumps({'maximum_relative_error':max(r['relative_error'] for r in cases)}))
    else:
        result=run_direct_case(args.widths,args.beta,args.t,args.eps,args.n,args.seed)
        Path(args.output).write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({"output":args.output,"runtime_seconds":result["runtime_seconds"]}))
