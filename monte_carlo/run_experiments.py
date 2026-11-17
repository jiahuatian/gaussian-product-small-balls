#!/usr/bin/env python3
"""Reproduce all samples from config.json; no conjectured coefficient in weights."""
from pathlib import Path
import argparse, concurrent.futures, json, math, time, hashlib, os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy.special import logsumexp
from direct_mc import run_direct_case
from vr_estimators import log_contributions

ROOT=Path(__file__).resolve().parent

def run_job(job):
    path=ROOT/'results'/job['mode']/f"{job['name']}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return {'name':job['name'],'status':'retained_existing'}
    start=time.perf_counter();b=job['benchmark'];n=job['n'];seed=job['seed']
    if job['mode']=='direct':
        result=run_direct_case(b['widths'],b['beta'],job['t_values'],job['eps_values'],n,seed,job['batch_size'])
        result.update(case_id=b['id'],job=job)
    else:
        L=job['log_t'];lam=b['lambda'];q=b['q'];shift=lam*L-q*math.log(L)
        rng=np.random.default_rng(seed)
        sv=sv2=sw=sw2=0.;maximum=0.;n_nonzero=0;metas=[]
        # Counts and unnormalized sums are retained for pooled IID estimates.
        for off in range(0,n,job['batch_size']):
            size=min(job['batch_size'],n-off)
            lv,lw,meta=log_contributions(size,b['widths'],b['beta'],L,rng,method=job['method'])
            values=np.exp(lv+shift)
            if not np.all(np.isfinite(values)):raise FloatingPointError('Nonfinite contribution')
            weights=np.exp(lw)
            sv+=float(values.sum());sv2+=float(values@values)
            sw+=float(weights.sum());sw2+=float(weights@weights)
            maximum=max(maximum,float(values.max()));n_nonzero+=int(np.count_nonzero(values))
            metas.append(meta)
        mean=sv/n;variance=max((sv2-sv*sv/n)/(n-1),0.);se=math.sqrt(variance/n)
        result={'case_id':b['id'],'widths':b['widths'],'beta':b['beta'],'method':job['method'],'n':n,'seed':seed,
            'log_t':L,'scaled_estimate':mean,'scaled_standard_error':se,'scaled_sample_variance':variance,
            'scaled_sum':sv,'scaled_sum_squares':sv2,'rescaling_log_factor':shift,
            'log_Z_estimate':math.log(mean)-shift if mean>0 else None,
            'contribution_ess':sv*sv/sv2 if sv2 else 0.,'importance_weight_ess':sw*sw/sw2 if sw2 else 0.,
            'importance_weight_mean':sw/n,'maximum_contribution_fraction':maximum/sv if sv else None,
            'nonzero_contributions':n_nonzero,'first_batch_proposal_metadata':metas[0],
            'face_samples':sum(m.get('face_count',0) for m in metas),
            'ordinary_samples':sum(m.get('ordinary_count',0) for m in metas),
            'runtime_seconds':time.perf_counter()-start,'job':job,
            'estimator':'ordinary arithmetic mean of p/q times exact conditional kernel; NOT self-normalized',
            'rescaling_note':'lambda,q rescale output only, and do not enter proposal density or conditional kernel.'}
    path.write_text(json.dumps(result,indent=2)+'\n')
    return {'name':job['name'],'status':'completed','seconds':round(time.perf_counter()-start,2)}

def jobs(config,mode):
    s=config[mode if mode=='direct' else ('variance_reduced' if mode=='vr' else 'efficiency')]
    out=[]
    for ci,b in enumerate(config['benchmarks']):
        if 'case_ids' in s and b['id'] not in s['case_ids']:continue
        if mode=='direct':
            for r in range(s['replicates']):
                out.append(dict(mode=mode,name=f"{b['id']}_seed{r}",benchmark=b,n=s['n_per_seed'],
                    seed=s['seed_base']+ci*1000+r,batch_size=s['batch_size'],t_values=s['t_values'],eps_values=s['eps_values']))
        else:
            for mi,method in enumerate(s.get('methods',[s.get('method')])):
                for li,L in enumerate(s['log_t_values']):
                    for r in range(s['replicates']):
                        out.append(dict(mode=mode,name=f"{b['id']}_{method}_L{L}_seed{r}",benchmark=b,n=s['n_per_seed'],
                            seed=s['seed_base']+ci*100000+mi*10000+li*100+r,batch_size=s['batch_size'],method=method,log_t=L))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['direct','vr','efficiency'],required=True)
    p.add_argument('--workers',type=int);p.add_argument('--config',default=str(ROOT/'config.json'))
    a=p.parse_args();config=json.loads(Path(a.config).read_text());work=jobs(config,a.mode)
    record={'config_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),'mode':a.mode,'jobs':work}
    (ROOT/'results'/f'{a.mode}_{Path(a.config).stem}_jobs.json').write_text(json.dumps(record,indent=2)+'\n')
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers or config['workers']) as pool:
        for i,r in enumerate(pool.map(run_job,work),1):print(json.dumps({'completed':i,'total':len(work),**r}),flush=True)
