"""Reproduce the separate fixed-allocation Bartlett-window comparison."""
from pathlib import Path
import json,time,math
import numpy as np
from scipy.special import gammaln
from vr_estimators import log_contributions,stratified_log_contributions,summarize_log_contributions,summarize_stratified

ROOT=Path(__file__).resolve().parent
CONFIG={'samples_per_replication':200000,'log_t':[30,100,320],
        'seeds':[735219,735229,735239,735251],
        'widths':[2,2,2,2,2],'betas':[1,2],
        'proposal':{'method':'shifted_window_mixture','face_probability':0.6,
                    'shifted_probability':0.3,'ordinary_probability':0.05,
                    'log_coordinate_shift':4.0,'logistic_width':3.0},
        'scope':'Fixed-allocation stratified versus iid importance mixture. Target Z(t)=E exp(-t||P||F²).'}

def main():
    output=ROOT/'results';output.mkdir(exist_ok=True)
    (output/'stratified_config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    rows=[]
    for beta in CONFIG['betas']:
      lam=1.5*beta;q=4
      for L in CONFIG['log_t']:
       offset=lam*L-q*math.log(L)-(gammaln(lam+1) if beta==1 else 0.)
       for seed in CONFIG['seeds']:
        for strat in [False,True]:
         # Method-specific seeds prevent incidental same-stream covariance.
         actual_seed=seed+beta*1000000+L*100+int(strat)*10000000
         rng=np.random.default_rng(actual_seed);started=time.perf_counter()
         if strat:
          values,lw,labels,meta=stratified_log_contributions(CONFIG['samples_per_replication'],CONFIG['widths'],beta,L,rng,**CONFIG['proposal'])
          summary=summarize_stratified(values+offset,labels)
         else:
          values,lw,meta=log_contributions(CONFIG['samples_per_replication'],CONFIG['widths'],beta,L,rng,**CONFIG['proposal'])
          summary=summarize_log_contributions(values+offset)
         elapsed=time.perf_counter()-started
         row={'beta':beta,'log_t':L,'seed':actual_seed,'method':'fixed_stratified' if strat else 'iid_mixture',
              'offset_applied':offset,'scaled_quantity':'Laplace-inferred effective kappa: e^(lambda L) Z(e^L)/(Gamma(lambda+1)L^q); NOT actualCDF' if beta==1 else 'Laplace leading-coefficient proxy: e^(lambda L) Z(e^L)/L^q; NOT actualCDF',
              'elapsed_seconds':elapsed,'metadata':meta,**summary}
         rows.append(row)
         print(beta,L,actual_seed,row['method'],summary['estimate'],summary['standard_error'],flush=True)
         (output/'stratified_replicates.json').write_text(json.dumps(rows,indent=2)+'\n')
    aggregates=[]
    for beta in CONFIG['betas']:
     for L in CONFIG['log_t']:
      for method in ['iid_mixture','fixed_stratified']:
       selected=[r for r in rows if r['beta']==beta and r['log_t']==L and r['method']==method]
       means=np.array([r['estimate'] for r in selected]);ses=np.array([r['standard_error'] for r in selected]);k=len(means)
       aggregates.append({'beta':beta,'log_t':L,'method':method,'replications':k,
           'scaled_quantity':selected[0]['scaled_quantity'],
           'estimate':float(means.mean()),'pooled_internal_standard_error':float(np.sqrt(np.sum(ses**2))/k),
           'replicate_standard_error':float(means.std(ddof=1)/math.sqrt(k)),
           'mean_contribution_ess':float(np.mean([r['contribution_ess'] for r in selected])),
           'largest_maximum_fraction':max(r['maximum_fraction'] for r in selected),
           'total_elapsed_seconds':sum(r['elapsed_seconds'] for r in selected)})
    (output/'stratified_summary.json').write_text(json.dumps(aggregates,indent=2)+'\n')

if __name__=='__main__':main()
