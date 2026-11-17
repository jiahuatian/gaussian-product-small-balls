#!/usr/bin/env python3
"""Pool independent replicates and produce machine-readable audit tables.

Intervals are empirical, pointwise sampling intervals. They do not bound an
asymptotic remainder or protect an estimator which misses rare contributions.
"""
from pathlib import Path
import json, math, csv
import numpy as np
from scipy.stats import t as student_t
from analytic_reference import exact_real_twofactor_laplace, exact_complex_square2_laplace, exact_real_223_cdf

ROOT=Path(__file__).resolve().parent

def pooled(ds, key):
    blocks=[d[key] for d in ds];r=len(blocks);n=sum(d['n'] for d in blocks)
    s=sum(np.asarray(d['sum']) for d in blocks)
    ss=sum(np.asarray(d['sum_squares']) for d in blocks)
    mean=s/n;within=np.sqrt(np.maximum((ss-s*s/n)/(n-1),0)/n)
    between=np.std([d['mean'] for d in blocks],axis=0,ddof=1)/math.sqrt(r)
    se=np.maximum(within,between)
    ans=dict(n=n,mean=mean.tolist(),se_within=within.tolist(),se_between=between.tolist(),
             se_working=se.tolist(),ci95_halfwidth=(student_t.ppf(.975,r-1)*se).tolist(),
             contribution_ess=np.divide(s*s,ss,out=np.zeros_like(s),where=ss>0).tolist())
    if 'event_count' in blocks[0]:
        count=sum(np.array(d['event_count']) for d in blocks);z=1.959963984540054
        center=(mean+z*z/(2*n))/(1+z*z/n)
        half=z*np.sqrt(mean*(1-mean)/n+z*z/(4*n*n))/(1+z*z/n)
        ans.update(event_count=count.tolist(),wilson95_lower=(center-half).tolist(),wilson95_upper=(center+half).tolist(),
                   zero_events_one_sided95_upper=1-.05**(1/n))
    return ans

def main():
    direct=[]
    groups={}
    for f in sorted((ROOT/'results/direct').glob('*.json')):
        d=json.loads(f.read_text());groups.setdefault(d['case_id'],[]).append(d)
    for case,ds in sorted(groups.items()):
        rec={'case_id':case,'widths':ds[0]['widths'],'beta':ds[0]['beta'],'t_values':ds[0]['t_values'],
             'eps_values':ds[0]['eps_values'],'seeds':[d['seed'] for d in ds]}
        for key in ['direct_laplace','conditional_laplace','paired_laplace_difference','direct_frobenius_cdf',
                    'direct_spectral_cdf','conditional_frobenius_cdf','paired_frobenius_cdf_difference']:
            if key in ds[0] and ds[0][key] is not None:rec[key]=pooled(ds,key)
        exact=[]
        for t in rec['t_values']:
            if case in ('R222','R223'):v=float(exact_real_twofactor_laplace(tuple(rec['widths']),t))
            elif case=='C22222':v=float(exact_complex_square2_laplace(4,t))
            else:break
            exact.append(v)
        if exact:
            rec['analytic_exact_laplace']=exact
            for key in ['direct_laplace','conditional_laplace']:
                rec[key]['exact_zscore']=((np.array(rec[key]['mean'])-exact)/rec[key]['se_working']).tolist()
        if case=='R223':rec['analytic_exact_frobenius_cdf']=[float(exact_real_223_cdf(e)) for e in rec['eps_values']]
        rec['empirical_conditional_variance_gain']=(np.array(rec['direct_laplace']['se_within'])**2/np.array(rec['conditional_laplace']['se_within'])**2).tolist()
        direct.append(rec)
    efficiencies=[];groups={}
    for f in sorted((ROOT/'results/efficiency').glob('*.json')):
        d=json.loads(f.read_text());groups.setdefault((d['case_id'],d['method'],d['log_t']),[]).append(d)
    for (case,method,L),ds in sorted(groups.items()):
        r=len(ds);mean=np.mean([d['scaled_estimate'] for d in ds]);n=sum(d['n'] for d in ds)
        within=math.sqrt(sum(d['scaled_standard_error']**2 for d in ds))/r
        between=np.std([d['scaled_estimate'] for d in ds],ddof=1)/math.sqrt(r)
        s=sum(d['scaled_sum'] for d in ds);ss=sum(d['scaled_sum_squares'] for d in ds)
        efficiencies.append(dict(case_id=case,method=method,L=L,n=n,mean=float(mean),
            se_within=within,se_between=float(between),se_working=max(within,between),
            contribution_ess=s*s/ss if ss else 0,run_seconds=sum(d['runtime_seconds'] for d in ds),
            variance_per_draw=within**2*n,
            max_single_seed_contribution_fraction=max(d['maximum_contribution_fraction'] or 0 for d in ds),
            caution='Empirical uncertainty can miss unsampled rare mass; do not infer efficiency from failed ordinary estimates.'))
    out={'uncertainty':'Working SE=max(pooled within-seed SE, independent replicate SE). Pointwise t intervals. Direct Bernoulli use Wilson interval or zero-event upper limit.',
         'direct':direct,'efficiency':efficiencies}
    (ROOT/'results/pooled_direct_and_efficiency.json').write_text(json.dumps(out,indent=2)+'\n')
    with (ROOT/'results/efficiency_table.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(efficiencies[0]));w.writeheader();w.writerows(efficiencies)
    data=json.loads((ROOT/'results/independent_production_analysis.json').read_text())['aggregates']
    fields=['case_id','method','L','replicates','n_total','mean','se_within','se_between','se_working','ratio_to_leading','ci95_ratio_halfwidth','contribution_ess','log_Z']
    with (ROOT/'results/laplace_convergence.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
    if (ROOT/'results/cdf_summary.json').exists():
        rows=json.loads((ROOT/'results/cdf_summary.json').read_text())
        for d in rows:
            d['se_working']=max(d['pooled_internal_standard_error'],d['replicate_standard_error'])
            d['ci95_ratio_halfwidth_working']=float(student_t.ppf(.975,3))*d['se_working']/d['reference_coefficient']
            d['qualification']='Actual CDF; working t3 interval differs from raw normal/internal-SE ci95 field.'
        (ROOT/'results/cdf_conservative_summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        with (ROOT/'results/cdf_convergence.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'direct_families':len(direct),'efficiency_groups':len(efficiencies),'laplace_groups':len(data)}))

if __name__=='__main__':main()
