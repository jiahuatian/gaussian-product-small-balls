"""Reproduce actual small-ball CDF tests, separate from Laplace comparisons."""
from pathlib import Path
import json,time,math,sys
import numpy as np
import mpmath as mp
from cdf_estimators import tilted_cdf_log_contributions,summarize_cdf_logs
from vr_estimators import gaussian_products
from analytic_reference import exact_real_223_cdf

ROOT=Path(__file__).resolve().parent
CONFIG={'samples_per_seed':250000,'replicates':4,'log_inverse_epsilon':[20,80,320,1280],
 'seed_base':27000000,'tilt_factor':1.0,
 'benchmarks':[
  {'id':'R222','widths':[2,2,2],'beta':1,'lambda':1.5,'q':0,'coefficient':math.pi/6,'status':'proved'},
  {'id':'R223','widths':[2,2,3],'beta':1,'lambda':2.,'q':1,'coefficient':1/16,'status':'proved'},
  {'id':'R2222','widths':[2,2,2,2],'beta':1,'lambda':1.5,'q':2,'coefficient':math.sqrt(2*math.pi)/48,'status':'conjectured'},
  {'id':'R22222','widths':[2,2,2,2,2],'beta':1,'lambda':1.5,'q':4,'coefficient':1/576,'status':'conjectured'},
  {'id':'C22222','widths':[2,2,2,2,2],'beta':2,'lambda':3.,'q':4,'coefficient':1/72,'status':'proved'}],
 'scope':'Actual Frobenius small-ball CDF, not a Laplace amplitude. Coefficients used only for reporting ratios.'}

def main():
 out=ROOT/'results';out.mkdir(exist_ok=True)
 (out/'cdf_config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
 rows=[]
 for ci,case in enumerate(CONFIG['benchmarks']):
  for li,L in enumerate(CONFIG['log_inverse_epsilon']):
   offset=case['lambda']*L-case['q']*math.log(L)
   for r in range(CONFIG['replicates']):
    seed=CONFIG['seed_base']+ci*100000+li*100+r
    method='ridge_mixture' if case['id']=='R223' else 'shifted_window_mixture'
    start=time.perf_counter()
    logs,lw,meta=tilted_cdf_log_contributions(CONFIG['samples_per_seed'],case['widths'],case['beta'],L,np.random.default_rng(seed),method=method,tilt_factor=CONFIG['tilt_factor'])
    result=summarize_cdf_logs(logs,offset)
    row={'case':case['id'],'L':L,'seed':seed,'replicate':r,'elapsed_seconds':time.perf_counter()-start,
         'method':method,'reference_coefficient':case['coefficient'],'reference_status':case['status'],
         'metadata':meta,**result}
    row['ratio']=row['estimate']/case['coefficient'];row['ratio_standard_error']=row['standard_error']/case['coefficient']
    rows.append(row)
    print(case['id'],L,r,row['ratio'],row['ratio_standard_error'],row['contribution_ess'],flush=True)
    (out/'cdf_replicates.json').write_text(json.dumps(rows,indent=2)+'\n')
 summary=[]
 for case in CONFIG['benchmarks']:
  for L in CONFIG['log_inverse_epsilon']:
   rr=[r for r in rows if r['case']==case['id'] and r['L']==L]
   means=np.array([r['estimate'] for r in rr]);ses=np.array([r['standard_error'] for r in rr]);k=len(rr)
   mean=float(means.mean());se=float(np.sqrt(np.sum(ses**2))/k)
   summary.append({'case':case['id'],'L':L,'estimate':mean,'pooled_internal_standard_error':se,
     'replicate_standard_error':float(means.std(ddof=1)/math.sqrt(k)),
     'reference_coefficient':case['coefficient'],'reference_status':case['status'],
     'ratio':mean/case['coefficient'],'ratio_standard_error':se/case['coefficient'],
     'ci95':[max(0.,mean-1.959963984540054*se),mean+1.959963984540054*se],
     'mean_contribution_ess':float(np.mean([r['contribution_ess'] for r in rr])),
     'largest_maximum_fraction':max(r['maximum_fraction'] for r in rr),
     'total_elapsed_seconds':sum(r['elapsed_seconds'] for r in rr)})
 (out/'cdf_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Moderate-threshold independent validation: true matrix draws and the
 # independently derived exact R223 Bessel CDF; no asymptotic comparator.
 validation=[]
 for ci,case in enumerate(CONFIG['benchmarks']):
  for ei,eps in enumerate([1.,.1]):
   L=-math.log(eps);n=500000;seed=27900000+ci*100+ei
   p=gaussian_products(n,case['widths'],case['beta'],np.random.default_rng(seed))
   indicator=np.sum(abs(p)**2,axis=(1,2))<=eps
   direct=float(indicator.mean());directse=float(indicator.std(ddof=1)/math.sqrt(n))
   method='ridge_mixture' if case['id']=='R223' else 'shifted_window_mixture'
   logs,lw,meta=tilted_cdf_log_contributions(n,case['widths'],case['beta'],L,np.random.default_rng(seed+100000),method=method)
   sm=summarize_cdf_logs(logs)
   item={'case':case['id'],'epsilon':eps,'direct_seed':seed,'tilted_seed':seed+100000,'n_each':n,
         'direct_estimate':direct,'direct_standard_error':directse,'tilted_estimate':sm['estimate'],
         'tilted_standard_error':sm['standard_error'],
         'independent_difference_z':(sm['estimate']-direct)/math.hypot(sm['standard_error'],directse)}
   if case['id']=='R223':
    item['exact_cdf']=float(exact_real_223_cdf(eps));item['tilted_minus_exact_z']=(sm['estimate']-item['exact_cdf'])/sm['standard_error']
   validation.append(item)
   (out/'cdf_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 # Exact finite-threshold control even at extreme L, using automatically
 # increased precision for the cancelling Bessel representation.
 exact_rows=[]
 with mp.workdps(80):
  for L in CONFIG['log_inverse_epsilon']:
   value=exact_real_223_cdf(mp.exp(-L),dps=70)*mp.exp(2*L)/L
   row=next(x for x in summary if x['case']=='R223' and x['L']==L)
   exact_rows.append({'L':L,'exact_scaled_cdf':str(value),'exact_ratio_to_kappa':str(16*value),
      'monte_carlo_minus_exact_z':(row['estimate']-float(value))/row['pooled_internal_standard_error']})
 (out/'cdf_exact_r223_validation.json').write_text(json.dumps(exact_rows,indent=2)+'\n')

if __name__=='__main__':main()
