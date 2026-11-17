"""Independent numerical spot checks of estimator calibration and proposal density.

This script tests numerical implementation, not an asymptotic theorem.  It uses
neither Conjecture E nor the sampling driver's precomputed reference values.
"""
from pathlib import Path
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import exp1
import vr_estimators as vr
from scipy.stats import t as student_t, chi2


def main(output=None):
    records={"qualification":"Numerical implementation checks only; no mathematical theorem certification.",
             "independent_seeds":{"calibration":940012,"ballot_base":4000},
             "calibration_n":1000000,"calibration":[],"positive_logistic":[],"ballot":[]}
    # Derive the 2x2,2x2 reduction independently in ordinary Cartesian Gamma
    # coordinates; do not call the VR kernel or analytic-reference module.
    rng=np.random.default_rng(940012)
    a=rng.gamma(1.,2.,1000000)
    b=rng.gamma(.5,2.,1000000)
    for t in [.1,1.,10.,100.]:
        f=lambda s: math.exp(-s/2)*math.atan(t*s/math.sqrt(1+2*t*s))/math.sqrt(1+2*t*s)/(4*t)
        exact,quad_error=quad(f,0,np.inf,epsabs=1e-13,epsrel=1e-12)
        values=(1+2*t*a)**-1*(1+2*t*b)**-.5*(1+2*t*(a+b))**-.5
        mean=float(values.mean());se=float(values.std(ddof=1)/math.sqrt(len(values)))
        records["calibration"].append(dict(widths=[2,2,2],t=t,positive_integral=exact,
            quadrature_error_estimate=quad_error,mc_mean=mean,mc_se=se,z_score=(mean-exact)/se))
    # Two independent forms of the exactly soluble q=1 reference.
    records['q1_reference_comparison']=[]
    for t in [.1,1.,10.,100.]:
        q,e=quad(lambda s:s*math.exp(-s/2)/(4*(1+t*s)*(1+2*t*s)),0,np.inf,epsabs=1e-13,epsrel=1e-12)
        I=lambda c:math.exp(1/(2*c))*exp1(1/(2*c))/c
        closed=(I(t)-I(2*t))/(4*t)
        records['q1_reference_comparison'].append(dict(t=t,quadrature=q,closed=closed,relative_difference=closed/q-1,quadrature_error_estimate=e))
    for center,width in [(1.,3.),(10.,3.),(100.,3.)]:
        q,e=quad(lambda s:math.exp(vr._positive_logistic_logpdf(s,center,width)),0,np.inf,epsabs=2e-12)
        records['positive_logistic'].append(dict(center=center,width=width,integral=q,quadrature_error_estimate=e))
    for m in [1,2,3,4]:
        rng=np.random.default_rng(4000+m);n=1000000
        a=rng.dirichlet(np.ones(m),n);b=rng.dirichlet(np.ones(m),n)
        event=np.all(np.cumsum(b,axis=1)[:,:-1]>=np.cumsum(a,axis=1)[:,:-1],axis=1)
        p=float(event.mean());se=math.sqrt((1/m)*(1-1/m)/n)
        records['ballot'].append(dict(m=m,n=n,seed=4000+m,observed=p,target=1/m,bernoulli_se_at_target=se,z_score=(p-1/m)/se if se else 0))
    records['all_checks_within_thresholds']=(all(abs(x['z_score'])<4 for x in records['calibration'])
        and all(abs(x['relative_difference'])<1e-10 for x in records['q1_reference_comparison'])
        and all(abs(x['integral']-1)<1e-9 for x in records['positive_logistic'])
        and all(abs(x['z_score'])<4 for x in records['ballot']))
    if output is None:output=Path(__file__).resolve().parent/'results'/'independent_audit_checks.json'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps(records,indent=2))
    return records

def importance_calibration(output=None):
    """Check the actual four-component mixture against independent exact cases."""
    rows=[]
    for widths in [(2,2,2),(2,2,3)]:
        for t in [1.,100.,10000.]:
            seed=771000+sum(widths)*100+int(math.log10(t))
            values,lw,meta=vr.log_contributions(300000,widths,1,math.log(t),
                np.random.default_rng(seed),method='shifted_window_mixture')
            summary=vr.summarize_log_contributions(values)
            if widths[-1]==2:
                f=lambda x:math.exp(-x/2)*math.atan(t*x/math.sqrt(1+2*t*x))/math.sqrt(1+2*t*x)/(4*t)
                exact,_=quad(f,0,np.inf,epsabs=1e-16,epsrel=1e-10)
            else:
                I=lambda a:math.exp(1/(2*a))*exp1(1/(2*a))/a
                exact=(I(t)-I(2*t))/(4*t)
            w=np.exp(lw)
            rows.append(dict(widths=widths,t=t,seed=seed,mean=summary['estimate'],
                se=summary['standard_error'],exact=exact,
                zscore=(summary['estimate']-exact)/summary['standard_error'],
                contribution_ess=summary['contribution_ess'],weight_mean=float(w.mean()),
                weight_se=float(w.std(ddof=1)/math.sqrt(len(w))),max_weight=float(w.max()),weight_upper_bound=20))
    if output is None:output=Path(__file__).resolve().parent/'results'/'independent_is_calibration.json'
    Path(output).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    return rows

def production_analysis(output=None):
    """Independent seed aggregation and model-sensitivity audit of saved runs.

    Fits are descriptive finite-range statistical models, not remainder bounds.
    All intercepts, lambda and q estimates are unconstrained in the relevant fit.
    The conservative working SE is max(between-seed SE, pooled within-seed SE).
    """
    root=Path(__file__).resolve().parent
    groups={}
    for f in sorted((root/'results'/'vr').glob('*.json')):
        d=json.loads(f.read_text())
        groups.setdefault((d['case_id'],d['method'],d['log_t']),[]).append(d)
    constants={'R222':math.pi**1.5/8,'R223':1/8,
        'R2222':math.pi*math.sqrt(2)/64,'R22222':math.sqrt(math.pi)/768,'C22222':1/12}
    aggregates=[]
    for (case,method,L),ds in sorted(groups.items()):
        if len(ds)<2:continue
        means=np.array([d['scaled_estimate'] for d in ds]);r=len(ds)
        between=float(means.std(ddof=1)/math.sqrt(r))
        within=math.sqrt(sum(d['scaled_standard_error']**2 for d in ds))/r
        se=max(within,between);mean=float(means.mean());A=constants[case]
        sv=sum(d['scaled_sum'] for d in ds);sv2=sum(d['scaled_sum_squares'] for d in ds)
        rec=dict(case_id=case,method=method,L=L,replicates=r,n_total=sum(d['n'] for d in ds),
            mean=mean,se_working=se,se_within=within,se_between=between,
            ratio_to_leading=mean/A,ci95_ratio_halfwidth=float(student_t.ppf(.975,r-1))*se/A,
            contribution_ess=sv*sv/sv2,
            largest_single_seed_contribution_fraction=max(d['maximum_contribution_fraction'] for d in ds),
            log_Z=math.log(mean)-ds[0]['rescaling_log_factor'])
        if case=='C22222':
            u=L-3*float(np.euler_gamma)
            full=(u**4/12+1.2020569031595942854*u+25*math.pi**4/144)/L**4
            rec.update(full_polynomial_scaled=full,full_polynomial_z=(mean-full)/se,
                full_polynomial_ratio=mean/full)
        aggregates.append(rec)
    def fit(X,y,sigma):
        X=np.asarray(X,float);y=np.asarray(y,float);sigma=np.asarray(sigma,float)
        W=X/sigma[:,None];v=y/sigma
        norms=np.linalg.norm(W,axis=0)
        scaled=W/norms
        coef_scaled=np.linalg.lstsq(scaled,v,rcond=None)[0]
        coef=coef_scaled/norms
        cov=np.linalg.inv(scaled.T@scaled)/np.outer(norms,norms)
        resid=(y-X@coef)/sigma
        dof=len(y)-X.shape[1];chisq=float(resid@resid)
        return {'coefficients':coef.tolist(),'standard_errors':np.sqrt(np.diag(cov)).tolist(),
            'covariance':cov.tolist(),'chi_square':chisq,'degrees_of_freedom':dof,
            'lack_of_fit_p':float(chi2.sf(chisq,dof)) if dof>0 else None,
            'design_condition_number':float(np.linalg.cond(scaled))}
    fits=[];exponent_fits=[]
    for case in constants:
        data=[d for d in aggregates if d['case_id']==case]
        for start in [10,20,40,80,160,320]:
            rows=[d for d in data if d['L']>=start]
            if not rows:continue
            L=np.array([d['L'] for d in rows]);y=np.array([d['mean'] for d in rows])/constants[case]
            sigma=np.array([d['se_working'] for d in rows])/constants[case]
            for degree in [1,2,3]:
                if len(rows)<degree+3:continue
                X=np.column_stack([(start/L)**j for j in range(degree+1)])
                f=fit(X,y,sigma);f.update(case_id=case,min_L=start,max_L=float(L.max()),degree=degree,
                    intercept_ratio=f['coefficients'][0],intercept_ratio_se=f['standard_errors'][0],
                    model='Aeff/A_reference = unconstrained_intercept + sum b_j*(min_L/L)^j')
                fits.append(f)
            for corrections in [0,1,2]:
                if len(rows)<5+corrections:continue
                X=np.column_stack([np.ones(len(L)),-L,np.log(L)]+
                    [(start/L)**j for j in range(1,corrections+1)])
                logZ=np.array([d['log_Z'] for d in rows]);logsigma=sigma/y
                f=fit(X,logZ,logsigma);f.update(case_id=case,min_L=start,max_L=float(L.max()),
                    corrections=corrections,lambda_hat=f['coefficients'][1],q_hat=f['coefficients'][2],
                    lambda_se=f['standard_errors'][1],q_se=f['standard_errors'][2],
                    model='log Z = a - unconstrained_lambda*L + unconstrained_q*log L + sum c_j*(min_L/L)^j')
                exponent_fits.append(f)
    records={'qualification':'Descriptive finite-range fits; no proven asymptotic remainder bound. SEs quantify sampling error only.',
        'aggregation':'Equal-sized independent seeds. Working SE=max(between-seed SE, pooled within-seed SE); pointwise t(r-1) intervals. Fits use working SE without forcing chi-square/dof=1.',
        'aggregates':aggregates,'intercept_fits':fits,'exponent_fits':exponent_fits}
    if output is None:output=root/'results'/'independent_production_analysis.json'
    Path(output).write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'production_groups':len(aggregates),'intercept_fits':len(fits),
                      'exponent_fits':len(exponent_fits),'output':str(output)},indent=2))
    return records

def cdf_tilt_calibration(output=None):
    """Validate the implemented tilted-Gamma CDF at fixed Bartlett coordinates.

    Set a=b=1 for two factors. The first two conditional Gamma terms then
    share a scale and combine, giving an independent positive convolution
    quadrature for a two-Gamma sum. This tests the actual CDF function without
    invoking the asymptotic formulas, determinant reference, or prior proposal.
    """
    from unittest.mock import patch
    from scipy.special import gammainc, gamma
    import cdf_estimators as cdf
    rows=[];n=1000000
    for beta in [1,2]:
        scale=2/beta
        combined_shape=1.5*beta;last_shape=.5*beta
        for ei,eps in enumerate([.01,1.,10.]):
            seed=889000+10*beta+ei
            exact,err=quad(lambda x:(x**(combined_shape-1)*math.exp(-x/scale)
                /(gamma(combined_shape)*scale**combined_shape))
                *gammainc(last_shape,(eps-x)/(2*scale)),0,eps,
                epsabs=1e-17,epsrel=1e-11)
            fixed=(np.zeros((n,1)),np.zeros((n,1)),np.zeros(n),{'audit_fixed_diagonals':True})
            with patch.object(cdf,'sample_log_diagonals',return_value=fixed):
                logs,_,meta=cdf.tilted_cdf_log_contributions(n,[2,2,2],beta,-math.log(eps),
                    np.random.default_rng(seed),method='ordinary')
            result=cdf.summarize_cdf_logs(logs)
            rows.append(dict(beta=beta,epsilon=eps,seed=seed,n=n,positive_convolution=exact,
                quadrature_error_estimate=err,estimate=result['estimate'],se=result['standard_error'],
                zscore=(result['estimate']-exact)/result['standard_error'],
                indicator_acceptance_fraction=meta['indicator_acceptance_fraction']))
    if output is None:output=Path(__file__).resolve().parent/'results'/'independent_cdf_tilt_calibration.json'
    Path(output).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--production-only',action='store_true')
    p.add_argument('--cdf-only',action='store_true')
    a=p.parse_args()
    if a.cdf_only:
        cdf_tilt_calibration()
    elif not a.production_only:
        main()
        importance_calibration()
        production_analysis()
        cdf_tilt_calibration()
    else:
        production_analysis()
