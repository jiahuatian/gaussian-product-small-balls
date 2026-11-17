#!/usr/bin/env python3
"""Rebuild the numerical-audit figures from recorded results, without resampling.

Usage: python make_plots.py [--results PATH] [--output PATH]
Requires numpy, scipy, matplotlib, mpmath, sympy.  Exact finite-t comparisons
use analytic_reference.py and are explicitly distinguished from Monte Carlo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.stats import norm, t as student_t

from analytic_reference import (complex_polynomial_coefficients,
                                complex_log_polynomial_value,
                                complex_frobenius_log_polynomial_value,
                                exact_complex_square2_laplace,
                                exact_real_223_cdf)

ROOT = Path(__file__).resolve().parent
COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7"]
LABELS = {
    "R222": r"$d=(2,2,2),\ q=0$ (known)",
    "R223": r"$d=(2,2,3),\ q=1$ (known)",
    "R2222": r"$d=(2,2,2,2),\ q=2$ (conjectured)",
    "R22222": r"$d=(2,2,2,2,2),\ q=4$ (conjectured)",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_figure(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def log_l_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks([10, 40, 160, 640, 1280])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel(r"$L=\log t$ (natural logarithm)")
    ax.grid(alpha=.2, which="major")
    ax.set_xlim(8, 1550)


def leading_ratios(data, output, coeff):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for cid, color, marker in zip(LABELS, COLORS, ["o", "s", "^", "D"]):
        rows = sorted((r for r in data["aggregates"] if r["case_id"] == cid), key=lambda r:r["L"])
        axes[0].errorbar([r["L"] for r in rows], [r["ratio_to_leading"] for r in rows],
                         yerr=[r["ci95_ratio_halfwidth"] for r in rows], color=color,
                         marker=marker, ms=4.2, lw=1, capsize=2.5, label=LABELS[cid])
    axes[0].set_ylim(.90, 2.13)
    axes[0].set_title("Real products: leading-term normalization", loc="left")
    axes[0].legend(loc="upper right", fontsize=8.4)
    rows = sorted((r for r in data["aggregates"] if r["case_id"] == "C22222"), key=lambda r:r["L"])
    axes[1].errorbar([r["L"] for r in rows], [r["ratio_to_leading"] for r in rows],
                     yerr=[r["ci95_ratio_halfwidth"] for r in rows], fmt="o", color=COLORS[0],
                     ms=4.2, capsize=3, label="Conditional importance sampling")
    dense = np.geomspace(10, 1280, 400)
    axes[1].plot(dense, np.polynomial.polynomial.polyval(dense, coeff)/(dense**4/12),
                 color=COLORS[2], lw=1.5, label="Full logarithmic polynomial / leading term")
    axes[1].set_ylim(.45, 1.055)
    axes[1].set_title(r"Complex $d=(2,2,2,2,2)$: $q=4$, $A=1/12$", loc="left")
    axes[1].legend(loc="lower right", fontsize=8.2)
    for ax in axes:
        ax.axhline(1, color=".35", ls="--", lw=1, zorder=0)
        ax.set_ylabel(r"$\widehat Z(t)t^{\lambda}/(A L^q)$")
        log_l_axis(ax)
    fig.suptitle("Finite-range convergence; bars are pointwise 95% Monte Carlo intervals", fontsize=11)
    save_figure(fig, output, "01_leading_normalized_ratios")


def complex_full_polynomial(data, output, coeff):
    rows = sorted((r for r in data["aggregates"] if r["case_id"] == "C22222"), key=lambda r:r["L"])
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    half = [r["ci95_ratio_halfwidth"]/(12*r["full_polynomial_scaled"]) for r in rows]
    ax.errorbar([r["L"] for r in rows], [r["full_polynomial_ratio"] for r in rows],
                yerr=half, fmt="o", color=COLORS[0], capsize=3, ms=5,
                label="Conditional importance sampling / full polynomial")
    exact = []
    with mp.workdps(75):
        for L in [10, 20, 40]:
            t = mp.exp(L)
            value = exact_complex_square2_laplace(4, t, dps=75)
            polynomial = complex_log_polynomial_value((2,2,2,2,2), L, dps=75)
            ratio = value * mp.exp(3*L) / polynomial
            exact.append({"L":L, "exact_Z":str(value), "exact_to_full_polynomial":float(ratio), "precision_decimal_digits":75})
    ax.plot([r["L"] for r in exact], [r["exact_to_full_polynomial"] for r in exact],
            "D", color=COLORS[2], markerfacecolor="none", ms=7,
            label="Exact finite-t determinant / full polynomial (analytic input)")
    ax.axhline(1, color=".35", ls="--", lw=1)
    ax.set_ylim(.975, 1.025)
    log_l_axis(ax)
    ax.set_ylabel(r"$\widehat Z(t)\,t^3/P_4(L)$")
    ax.set_title(r"Complex $d=(2,2,2,2,2)$: full logarithmic polynomial", loc="left")
    ax.legend(loc="lower right", fontsize=8.6)
    ax.text(.035, .94,
            r"$P_4(L)=(L-3\gamma)^4/12+\zeta(3)(L-3\gamma)+25\pi^4/144$",
            transform=ax.transAxes, va="top", fontsize=10)
    save_figure(fig, output, "02_complex_full_polynomial")
    return exact


def intercept_windows(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained", sharey=True)
    selected=[]
    for ax, cid, q in zip(axes, ["R2222", "R22222"], [2, 4]):
        for degree, color, marker, shift in zip([1,2,3], COLORS, ["o","s","^"], [.93,1,1.075]):
            rows = sorted((r for r in data["intercept_fits"] if r["case_id"]==cid and
                           r["degree"]==degree and r["lack_of_fit_p"]>.01), key=lambda r:r["min_L"])
            selected.extend(rows)
            ax.errorbar([r["min_L"]*shift for r in rows], [r["intercept_ratio"] for r in rows],
                        yerr=[norm.ppf(.975)*r["intercept_ratio_se"] for r in rows],
                        fmt=marker, color=color, capsize=3, ms=4.7, label=f"Degree {degree}")
        ax.axhline(1, color=".35", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([10,20,40,80,160])
        ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlabel(r"Fit window lower endpoint $L_{\min}$")
        ax.set_title(rf"Real $q={q}$: free intercept, $L_{{\max}}=1280$", loc="left")
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim(7.5,210)
    axes[0].set_ylabel(r"Extrapolated intercept / conjectured $A$")
    axes[0].set_ylim(.9958,1.0048)
    fig.suptitle("Accepted finite-range fits (lack-of-fit p > 0.01); 95% sampling-only WLS intervals", fontsize=11)
    save_figure(fig, output, "03_real_intercept_window_sensitivity")
    return selected


def wilson(count, total):
    z = norm.ppf(.975)
    p = count/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return np.maximum(center-half,0), np.minimum(center+half,1)


def direct_cdfs(results, output):
    files = sorted((results / "direct").glob("*.json"))
    raw = [json.loads(p.read_text()) for p in files]
    ids = ["R222", "R223", "R2222", "R22222", "C22222", "R224"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), layout="constrained", sharex=True, sharey=True)
    pooled=[]
    for ax, cid in zip(axes.flat, ids):
        cases=[r for r in raw if r["case_id"]==cid]
        if not cases:
            ax.set_visible(False)
            continue
        eps=np.array(cases[0]["eps_values"])
        total=sum(r["n"] for r in cases)
        count=np.sum([r["direct_frobenius_cdf"]["event_count"] for r in cases],axis=0)
        mean=count/total
        low,high=wilson(count,total)
        positive=count>0
        ax.errorbar(eps[positive],mean[positive],yerr=np.vstack([mean[positive]-low[positive],high[positive]-mean[positive]]),
                    fmt="o", color=COLORS[0], capsize=2, ms=4, label="Direct: Wilson 95% interval")
        zero=~positive
        if zero.any():
            ax.errorbar(eps[zero],high[zero],yerr=high[zero]*.45,uplims=True,fmt="o",ms=3,
                        color=COLORS[0],capsize=3,label="Zero hits: Wilson upper endpoint")
        record={"case_id":cid,"n_total":total,"replicates":len(cases),"epsilon":eps.tolist(),
                "direct_count":count.tolist(),"direct_mean":mean.tolist(),
                "direct_wilson95":np.column_stack([low,high]).tolist()}
        if all(r["conditional_cdf_available"] for r in cases):
            means=np.array([r["conditional_frobenius_cdf"]["mean"] for r in cases])
            within=np.sqrt(np.sum([np.array(r["conditional_frobenius_cdf"]["standard_error"])**2 for r in cases],axis=0))/len(cases)
            between=np.std(means,axis=0,ddof=1)/np.sqrt(len(cases))
            cm=means.mean(axis=0)
            half=student_t.ppf(.975,len(cases)-1)*np.maximum(within,between)
            ax.errorbar(eps*1.11,cm,yerr=np.minimum(half,cm*(1-1e-10)),fmt="s",color=COLORS[2],
                        capsize=2,ms=3.7,label="Final-layer conditional: 95% interval")
            record.update(conditional_mean=cm.tolist(),conditional_ci95_halfwidth=half.tolist())
        if cid=="R223":
            dense=np.geomspace(1e-4,1,65)
            analytic=np.array([float(exact_real_223_cdf(e)) for e in dense])
            ax.plot(dense,analytic,color=COLORS[1],lw=1,label="Exact CDF (analytic input)")
        field="Complex" if cid.startswith("C") else "Real"
        widths=",".join(str(x) for x in cases[0]["widths"])
        ax.set_title(f"{field} d=({widths}); n={total:,}",loc="left",fontsize=9.7)
        ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlim(7e-5,1.6);ax.set_ylim(2e-9,.5)
        ax.grid(alpha=.18,which="major")
        pooled.append(record)
    for ax in axes[1]:ax.set_xlabel(r"$\epsilon$ in $\Pr(\|P\|_F^2\leq\epsilon)$")
    for ax in axes[:,0]:ax.set_ylabel(r"Frobenius CDF $F(\epsilon)$")
    handles=[Line2D([],[],color=COLORS[0],marker="o",ls="none",label="Direct + Wilson 95% intervals"),
             Line2D([],[],color=COLORS[2],marker="s",ls="none",label="Final-layer conditional + 95% intervals"),
             Line2D([],[],color=COLORS[1],lw=1,label="Exact real (2,2,3) CDF: analytic input")]
    fig.legend(handles=handles,loc="outside lower center",ncol=3,fontsize=8.5)
    fig.suptitle("Original Gaussian-entry sampling: Frobenius small-ball probabilities\nDownward arrows show Wilson upper endpoints after zero direct hits",fontsize=11)
    save_figure(fig,output,"04_direct_frobenius_cdf")
    return pooled, files


def deep_tail_cdfs(results, output):
    """Plot the separately sampled CDF itself, with conservative seed errors."""
    source=results/"cdf_summary.json"
    data=json.loads(source.read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    rows_for_manifest=[]
    for cid,color,marker in zip([*LABELS,"C22222"],[*COLORS,COLORS[0]],["o","s","^","D","o"]):
        rows=sorted((r for r in data if r["case"]==cid),key=lambda r:r["L"])
        half=[student_t.ppf(.975,3)*max(r["pooled_internal_standard_error"],r["replicate_standard_error"])/r["reference_coefficient"] for r in rows]
        for r,h in zip(rows,half):
            rows_for_manifest.append({**r,"plotted_conservative_t3_ci95_ratio_halfwidth":float(h)})
        ax=axes[1] if cid.startswith("C") else axes[0]
        label="Actual-CDF importance sampling" if cid.startswith("C") else LABELS[cid]
        ax.errorbar([r["L"] for r in rows],[r["ratio"] for r in rows],yerr=half,
                    marker=marker,ms=4.5,color=color,lw=1,capsize=3,label=label)
    dense=np.geomspace(20,1280,240)
    r223=1+(1.5-2*float(mp.euler))/dense
    axes[0].plot(dense,r223,color=COLORS[1],ls=":",lw=1.4,
                 label=r"Known $q=1$ correction: $1+(3/2-2\gamma)/L$")
    full=np.array([float(complex_frobenius_log_polynomial_value((2,2,2,2,2),float(L))) for L in dense])
    axes[1].plot(dense,72*full/dense**4,color=COLORS[2],lw=1.5,
                 label="Derived full Frobenius polynomial / leading term")
    for ax in axes:
        ax.axhline(1,color=".35",ls="--",lw=1)
        ax.set_xscale("log")
        ax.set_xticks([20,80,320,1280]);ax.get_xaxis().set_major_formatter(ScalarFormatter())
        ax.set_xlim(16,1600);ax.grid(alpha=.2)
        ax.set_xlabel(r"$L=\log(1/\epsilon)$ (natural logarithm)")
        ax.set_ylabel(r"$\widehat F(e^{-L})\,e^{\lambda L}/(\kappa L^q)$")
    axes[0].set_ylim(.95,1.75)
    axes[0].set_title("Real products: actual Frobenius CDF",loc="left")
    axes[0].legend(loc="upper right",fontsize=8)
    axes[1].set_ylim(.82,1.05)
    axes[1].set_title(r"Complex $d=(2,2,2,2,2)$: $\kappa=1/72$",loc="left")
    axes[1].legend(loc="lower right",fontsize=8)
    fig.suptitle("Actual deep-tail small-ball probabilities; pointwise conservative 95% Monte Carlo intervals",fontsize=11)
    save_figure(fig,output,"05_actual_deep_tail_cdf")
    return {"source_file":str(source),"source_sha256":sha256(source),"points":rows_for_manifest,
            "complex_derived_curve":{"L":dense.tolist(),"ratio_to_leading":(72*full/dense**4).tolist()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results")
    parser.add_argument("--output",type=Path,default=ROOT/"plots")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":10.5,
                         "axes.labelsize":10,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"ps.fonttype":42,"savefig.facecolor":"white"})
    source=args.results/"independent_production_analysis.json"
    data=json.loads(source.read_text())
    coeff=[float(x) for x in complex_polynomial_coefficients((2,2,2,2,2),dps=75)]
    leading_ratios(data,args.output,coeff)
    exact=complex_full_polynomial(data,args.output,coeff)
    accepted=intercept_windows(data,args.output)
    cdfs,directfiles=direct_cdfs(args.results,args.output)
    deep_cdf=deep_tail_cdfs(args.results,args.output)
    manifest={"source_file":str(source),"source_sha256":sha256(source),
              "analytic_reference_sha256":sha256(ROOT/"analytic_reference.py"),
              "plot_script_sha256":sha256(__file__),"complex_polynomial_coefficients_ascending":coeff,
              "exact_complex_finite_t_comparisons":exact,"accepted_intercept_fits":accepted,
              "direct_cdf_pools":cdfs,"direct_source_hashes":{p.name:sha256(p) for p in directfiles},
              "actual_deep_tail_cdf":deep_cdf}
    (args.output/"plot_data.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (args.output/"CAPTIONS.md").write_text(CAPTIONS)
    print(json.dumps({"output_directory":str(args.output),"figures":5,"exact_complex_comparisons":exact},indent=2))


CAPTIONS=r"""# Numerical-audit figure captions

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
"""


if __name__=="__main__":main()
