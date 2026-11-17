#!/usr/bin/env python3
"""Replay saved seeds and compare extended precision kernels."""
from pathlib import Path
import json,math,hashlib,platform,sys,importlib.metadata
import numpy as np
from scipy.special import logsumexp
from vr_estimators import sample_log_diagonals,conditional_log_kernel,log_contributions
from direct_mc import run_direct_case
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    checks={'meaning':'Implementation reproducibility only; not mathematical certification.','replays':[],'extended_precision':[]}
    p=ROOT/'results/direct/R222_seed0.json';d=json.loads(p.read_text());j=d['job']
    fresh=run_direct_case(d['widths'],d['beta'],d['t_values'],d['eps_values'],d['n'],d['seed'],d['batch_size'])
    differences={k:float(np.max(np.abs(np.array(fresh[k]['mean'])-d[k]['mean']))) for k in ['direct_laplace','conditional_laplace','direct_frobenius_cdf']}
    checks['replays'].append(dict(file=str(p.relative_to(ROOT)),seed=d['seed'],max_absolute_mean_differences=differences,bitwise_equal=all(v==0 for v in differences.values())))
    for case in ['R22222','C22222']:
        p=ROOT/f'results/vr/{case}_shifted_window_mixture_L1280_seed0.json';d=json.loads(p.read_text());j=d['job']
        rng=np.random.default_rng(d['seed']);s=ss=0.
        for off in range(0,d['n'],j['batch_size']):
            lv,lw,meta=log_contributions(min(j['batch_size'],d['n']-off),d['widths'],d['beta'],d['log_t'],rng,method=d['method'])
            v=np.exp(lv+d['rescaling_log_factor']);s+=float(v.sum());ss+=float(v@v)
        checks['replays'].append(dict(file=str(p.relative_to(ROOT)),seed=d['seed'],sum_difference=s-d['scaled_sum'],sum_squares_difference=ss-d['scaled_sum_squares'],bitwise_equal=s==d['scaled_sum'] and ss==d['scaled_sum_squares']))
    for beta in [1,2]:
        for L in [640.,1280.]:
            seed=29100000+beta*1000+int(L);w=(2,2,2,2,2)
            a,b,_,_=sample_log_diagonals(50000,w,beta,L,np.random.default_rng(seed),method='shifted_window_mixture')
            f64=conditional_log_kernel(a,b,L,w,beta)
            x=a.astype(np.longdouble);y=b.astype(np.longdouble)
            suffix=np.concatenate([np.cumsum(x[:,::-1],axis=1)[:,::-1],np.zeros((len(x),1),dtype=np.longdouble)],axis=1)
            prefix=np.concatenate([np.zeros((len(x),1),dtype=np.longdouble),np.cumsum(y,axis=1)],axis=1)
            logv=logsumexp(suffix+prefix,axis=1);c=np.longdouble(L)+np.log(np.longdouble(2)/beta)
            extended=(-beta*w[1]/2*np.logaddexp(0,c+x.sum(1))-beta*(w[-1]-1)/2*np.logaddexp(0,c+y.sum(1))-beta/2*np.logaddexp(0,c+logv))
            delta=np.asarray(f64,dtype=np.longdouble)-extended
            checks['extended_precision'].append(dict(beta=beta,L=L,n=len(x),seed=seed,max_absolute_log_kernel_error=float(abs(delta).max()),median_absolute_log_kernel_error=float(np.median(abs(delta))),longdouble_mantissa_bits=np.finfo(np.longdouble).nmant))
    checks['environment']={'python':sys.version,'platform':platform.platform(),'rng':'NumPy default_rng / PCG64',
        'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','mpmath','sympy','matplotlib']},
        'thread_control':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1'}
    checks['qualification']='Replays are bitwise on recorded environment only. Extended precision tests the kernel, not every proposal-density operation; no rigorous floating-point error bound.'
    (ROOT/'results/reproducibility_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
