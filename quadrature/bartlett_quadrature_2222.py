#!/usr/bin/env python3
"""Deterministic high-precision test for three 2x2 Gaussian factors, d = (2,2,2,2).

Real entries test Conjecture E of the paper at q = 2; complex entries, where the
constant is a theorem (Theorem A), serve as a control.

Reduction (exact).  In Bartlett coordinates the three factors are upper triangular,
with diagonal entries x_j, z_j and off-diagonal entries y_j, and
    ||P||_F^2 = (x3 x2 x1)^2 + (z3 z2 z1)^2 + (x3 x2 y1 + x3 y2 z1 + y3 z2 z1)^2 .
Integrating out y1, y2, y3, x1 and z3 exactly gives, with
X = x3^2 x2^2, Y = x3^2 z1^2, W = z1^2 z2^2,
    real:    Z(t) = E[(1+2tX)^-1 (1+2tW)^-1/2 (1+2t(X+Y+W))^-1/2]
    complex: Z(t) = E[(1+tX)^-2  (1+tW)^-1   (1+t(X+Y+W))^-1  ]
where Z(t) = E exp(-t ||P||_F^2).  The joint density of (X, Y, W) is
    real:    h(X,Y,W) = W^(-1/2) K0( sqrt((1+W/Y)(X+Y)) ) / (4 pi Y)
    complex: h(X,Y,W) = (2X/Y) K0( 2 sqrt((1+W/Y)(X+Y)) )
so Z(t) is an exact three-dimensional integral.  It is evaluated by composite
Gauss-Legendre quadrature in logarithmic coordinates.

Prediction.  t^(3/2) Z_R(t) = A L^2 + B L + C + o(1) with L = log t and
A = Gamma(5/2) sqrt(2 pi)/48 = pi sqrt(2)/64 (Conjecture E, kappa_R = sqrt(2 pi)/48);
t^3 Z_C(t) = A L^2 + ... with A = kappa_F Gamma(4) = 1/2 (Theorem A).

Usage:  python bartlett_quadrature_2222.py [validate|validate_large|sweep|converge|summary|crosscheck|all]
Each mode prints its output and writes it to results/<mode>.txt.  summary and crosscheck
read results/sweep.txt; crosscheck also reads ../monte_carlo/results/laplace_convergence.csv.
Approximate single-core run times: validate 15 s, validate_large 15 s, sweep 1 min,
converge 3 min, summary < 1 s, crosscheck 1 min.
"""
import re
import sys
import time
from math import gamma, pi, sqrt
from pathlib import Path

import numpy as np
from scipy.special import k0

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
A_REAL = pi * sqrt(2) / 64          # Gamma(5/2) * sqrt(2 pi) / 48
A_CPLX = 0.5                        # kappa_F * Gamma(4) = (1/12) * 6


class Tee:
    def __init__(self, name):
        RESULTS.mkdir(exist_ok=True)
        self.f = open(RESULTS / f'{name}.txt', 'w')

    def __call__(self, line=''):
        print(line, flush=True)
        self.f.write(line + '\n')

    def close(self):
        self.f.close()


def gl_composite(a, b, panel, n):
    m = max(1, int(np.ceil((b - a) / panel)))
    edges = np.linspace(a, b, m + 1)
    x, w = np.polynomial.legendre.leggauss(n)
    lo, hi = edges[:-1, None], edges[1:, None]
    return (0.5 * (hi - lo) * x + 0.5 * (hi + lo)).ravel(), (0.5 * (hi - lo) * w).ravel()


def Z(t, field, panel=2.0, n=8, G_one=False):
    """Z(t) = E exp(-t ||P||_F^2) as the 3-D integral; G_one=True integrates h alone."""
    L = np.log(max(t, 1.0))
    if field == 'R':
        a1, a2, a3 = -L - 34, -2 * L - 48, -L - 64
    else:
        a1, a2, a3 = -L - 22, -2 * L - 48, -L - 34
    b = 7.5
    x1, w1 = gl_composite(a1, b, panel, n)
    x2, w2 = gl_composite(a2, b, panel, n)
    x3, w3 = gl_composite(a3, b, panel, n)
    Y = np.exp(x2)[:, None]
    W = np.exp(x3)[None, :]
    WY = W / Y
    W2 = w2[:, None] * w3[None, :]
    total = 0.0
    for xi, wi in zip(x1, w1):
        X = np.exp(xi)
        arg = (1.0 + WY) * (X + Y)
        if field == 'R':
            dens = X * np.sqrt(W) * k0(np.sqrt(arg)) / (4 * np.pi)       # h * X * Y * W
            G = 1.0 if G_one else 1.0 / (1 + 2 * t * X) / np.sqrt((1 + 2 * t * W) * (1 + 2 * t * (X + Y + W)))
        else:
            dens = 2 * X * X * W * k0(2 * np.sqrt(arg))
            G = 1.0 if G_one else 1.0 / (1 + t * X) ** 2 / ((1 + t * W) * (1 + t * (X + Y + W)))
        total += wi * np.sum(W2 * dens * G)
    return total


def mc_direct(ts, field, batches, batch, seed):
    """Monte Carlo of E exp(-t ||A3 A2 A1||_F^2) from Gaussian matrices; returns means and SEs."""
    rng = np.random.default_rng(seed)
    S = np.zeros(len(ts)); S2 = np.zeros(len(ts)); N = 0
    for _ in range(batches):
        if field == 'R':
            A = rng.standard_normal((3, batch, 2, 2))
        else:
            A = (rng.standard_normal((3, batch, 2, 2)) + 1j * rng.standard_normal((3, batch, 2, 2))) / np.sqrt(2)
        P = A[2] @ A[1] @ A[0]
        g = np.sum(np.abs(P) ** 2, axis=(1, 2))
        for i, t in enumerate(ts):
            e = np.exp(-t * g); S[i] += e.sum(); S2[i] += (e * e).sum()
        N += batch
    m = S / N
    return m, np.sqrt(S2 / N - m ** 2) / np.sqrt(N)


def mc_reduced(ts, batches, batch, seed):
    """Monte Carlo of the reduced real expectation over (x3^2, x2^2, z1^2, z2^2)."""
    rng = np.random.default_rng(seed)
    S = np.zeros(len(ts)); S2 = np.zeros(len(ts)); N = 0
    for _ in range(batches):
        u = rng.chisquare(2, batch); v = rng.chisquare(2, batch)
        p = rng.chisquare(1, batch); r = rng.chisquare(1, batch)
        X, Y, W = u * v, u * p, p * r
        for i, t in enumerate(ts):
            e = 1 / (1 + 2 * t * X) / np.sqrt((1 + 2 * t * W) * (1 + 2 * t * (X + Y + W)))
            S[i] += e.sum(); S2[i] += (e * e).sum()
        N += batch
    m = S / N
    return m, np.sqrt(S2 / N - m ** 2) / np.sqrt(N)


def validate():
    out = Tee('validate')
    out('Normalization of the closed-form densities, and quadrature against direct Monte Carlo')
    out('(400k Gaussian matrix triples per field; one sample reused for all t, so deviations are correlated).')
    for f in ('R', 'C'):
        out(f'[{f}] integral of h: {Z(1.0, f, G_one=True):.12f}  (should be 1)')
    ts = (0.5, 2.0, 10.0)
    for f in ('R', 'C'):
        m, s = mc_direct(ts, f, 4, 100_000, 1)
        for i, t in enumerate(ts):
            q = Z(t, f)
            out(f'[{f}] t={t:5.1f}  quadrature {q:.6f}   Monte Carlo {m[i]:.6f} +- {s[i]:.6f}   diff/sigma {(q - m[i]) / s[i]:+.2f}')
    out.close()


def validate_large():
    out = Tee('validate_large')
    out('Quadrature against larger independent Monte Carlo samples: direct matrix sampling (4e6 triples)')
    out('and the reduced four-variable expectation (1e7 draws, real field only).')
    ts = (0.5, 2.0, 10.0, 50.0)
    mr, sr = mc_reduced(ts, 20, 500_000, 7)
    for f, seed in (('R', 20261117), ('C', 20261118)):
        md, sd = mc_direct(ts, f, 20, 200_000, seed)
        for i, t in enumerate(ts):
            q = Z(t, f)
            red = f' | reduced MC {mr[i]:.6f}+-{sr[i]:.6f} ({(q - mr[i]) / sr[i]:+.2f} sig)' if f == 'R' else ''
            out(f'[{f}] t={t:5.1f}  quadrature {q:.6f}{red} | direct MC {md[i]:.6f}+-{sd[i]:.6f} ({(q - md[i]) / sd[i]:+.2f} sig)')
    out.close()


def sweep():
    out = Tee('sweep')
    for f, expo, pred, ks in (('C', 3.0, A_CPLX, range(4, 15)), ('R', 1.5, A_REAL, range(6, 17))):
        Ls, Fs = [], []
        for k in ks:
            t = 10.0 ** k; t0 = time.time()
            F = t ** expo * Z(t, f)
            Ls.append(np.log(t)); Fs.append(F)
            out(f'[{f}] t=1e{k:<3d} t^{expo} Z(t) = {F:.12f}   ({time.time() - t0:.1f}s)')
        Ls, Fs = np.array(Ls), np.array(Fs)
        out(f'[{f}] prediction for the L^2 coefficient: {pred:.10f}')
        for lo in (0, 2, 4):
            c = np.polyfit(Ls[lo:], Fs[lo:], 2)
            out(f'[{f}] quadratic fit using t >= 1e{list(ks)[lo]}: A = {c[0]:.10f}  B = {c[1]:.8f}  C = {c[2]:.8f}   A/pred = {c[0] / pred:.8f}')
        d2 = (Fs[2:] - 2 * Fs[1:-1] + Fs[:-2]) / (np.log(10.0) ** 2) / 2
        out(f'[{f}] local A from second differences (last five): {np.array2string(d2[-5:], precision=8)}')
    out.close()


def converge():
    out = Tee('converge')
    out('Grid refinement at the largest t (panel width, Gauss-Legendre nodes per panel).')
    for f, t, ex in (('R', 1e15, 1.5), ('R', 1e16, 1.5), ('C', 1e13, 3.0)):
        vals = []
        for panel, n in ((2.0, 8), (2.0, 12), (1.0, 8)):
            t0 = time.time(); F = t ** ex * Z(t, f, panel, n); vals.append(F)
            out(f'[{f}] t={t:.0e} panel={panel} nodes/panel={n}: t^{ex} Z = {F:.12f}  ({time.time() - t0:.0f}s)')
        out(f'[{f}]   max relative spread: {(max(vals) - min(vals)) / abs(vals[0]):.2e}')
    out.close()


def coefficients(tag):
    """Read results/sweep.txt and return (k, F, local A, extrapolated A, B, C) for one field.

    The local A is the second difference of F = t^lambda Z(t) in L = log t; it converges like
    t^(-1/2) (real) or t^(-1) (complex) and is extrapolated with two correction terms.  B and C
    are then fitted to F - A L^2 for t >= 1e10, with A fixed at its extrapolated value."""
    txt = (RESULTS / 'sweep.txt').read_text()
    ks, Fs = [], []
    for m in re.finditer(r'\[' + tag + r'\] t=1e(\d+)\s+t\^[\d.]+ Z\(t\) = ([\d.]+)', txt):
        ks.append(int(m.group(1))); Fs.append(float(m.group(2)))
    k, F = np.array(ks), np.array(Fs)
    eta = 0.5 if tag == 'R' else 1.0
    d = (F[2:] - 2 * F[1:-1] + F[:-2]) / np.log(10.0) ** 2 / 2; kc = k[1:-1]
    M = np.vstack([np.ones(4), 10.0 ** (-eta * kc[-4:]), 10.0 ** (-2 * eta * kc[-4:])]).T
    A = np.linalg.lstsq(M, d[-4:], rcond=None)[0][0]
    sel = k >= 10
    Lk = np.log(10.0) * k[sel]
    B, C = np.polyfit(Lk, F[sel] - A * Lk ** 2, 1)
    return k, F, kc, d, A, B, C


def summary():
    out = Tee('summary')
    for tag, pred, name in (('C', A_CPLX, 'complex (Theorem A): A = kappa_F*Gamma(4) = 1/2'),
                            ('R', A_REAL, 'real (Conjecture E): A = Gamma(5/2)*sqrt(2 pi)/48 = pi*sqrt(2)/64')):
        k, F, kc, d, A, B, C = coefficients(tag)
        out(name)
        out('  local A at t = 1e%d..1e%d: ' % (kc[-5], kc[-1]) + ' '.join(f'{x:.8f}' for x in d[-5:]))
        out(f'  extrapolated A = {A:.9f}   predicted {pred:.9f}   relative difference {(A - pred) / pred:+.1e}')
        if tag == 'R':
            out(f'  kappa_R(2,2,2,2) = A/Gamma(5/2) = {A / gamma(2.5):.9f}   sqrt(2 pi)/48 = {sqrt(2 * pi) / 48:.9f}')
            out(f'  subleading coefficients (A fixed, fit for t >= 1e10): B = {B:.6f}, C = {C:.6f}')
            for L in (320.0, 640.0, 1280.0):
                out(f'  implied ratio t^(3/2) Z(t) / (A_pred L^2) at L = {L:6.0f}: {(A * L * L + B * L + C) / (pred * L * L):.6f}')
    out.close()


def crosscheck():
    """Compare with the independent Monte Carlo study (../monte_carlo/results/laplace_convergence.csv).

    The Monte Carlo reports, for real d = (2,2,2,2), the ratio t^(3/2) Z(t) / (A_pred L^2) at
    L = log t = 10, 20, ..., 1280.  Here the same ratio is computed by direct quadrature for
    L <= 80 and, for larger L, from A L^2 + B L + C with the coefficients of coefficients('R'),
    which use the quadrature alone; there the O(t^(-1/2)) corrections are negligible."""
    import csv
    rows = csv.DictReader(open(HERE.parent / 'monte_carlo' / 'results' / 'laplace_convergence.csv'))
    mc = {float(r['L']): (float(r['ratio_to_leading']), float(r['ci95_ratio_halfwidth'])) for r in rows if r['case_id'] == 'R2222'}
    _, _, _, _, A, B, C = coefficients('R')
    c = (A, B, C)
    out = Tee('crosscheck')
    out('Real d=(2,2,2,2): ratio t^(3/2) Z(t) / (A_pred L^2), A_pred = pi*sqrt(2)/64, against the Monte Carlo study.')
    for L in sorted(mc):
        if L <= 80:
            ratio = np.exp(1.5 * L) * Z(np.exp(L), 'R') / (A_REAL * L * L); how = 'direct quadrature'
        else:
            ratio = (c[0] * L * L + c[1] * L + c[2]) / (A_REAL * L * L); how = 'fitted polynomial'
        r, hw = mc[L]
        out(f'L={L:6.0f}  quadrature {ratio:.6f} ({how:17s}) | Monte Carlo {r:.6f} +- {hw:.6f} (95%) | difference / half-width {(ratio - r) / hw:+.2f}')
    out.close()


if __name__ == '__main__':
    modes = {'validate': validate, 'validate_large': validate_large, 'sweep': sweep, 'converge': converge, 'summary': summary,
             'crosscheck': crosscheck}
    arg = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    for name in (modes if arg == 'all' else [arg]):
        modes[name]()
