"""Independent exact Bartlett conditional estimators for bottleneck-two products.

No asymptotic coefficient, Conjecture E value, or fitted normalizing constant is
used in these estimators. The Schubert window affects only proposal efficiency;
an explicitly weighted full-support mixture protects unbiasedness.

All public `log_t` values use Z(t)=E exp(-t ||P||_F**2).
Real entries are N(0,1); complex entries are (N(0,1)+iN(0,1))/sqrt(2).
"""
from __future__ import annotations
import math
import numpy as np
from scipy.special import gammaln, logsumexp, expit, logit


def _loggamma_sample(shape, size, rng):
    """Gamma(shape,1) in log coordinates, with no underflow truncation."""
    shape=np.broadcast_to(np.asarray(shape,float),size)
    low=shape<1
    ans=np.log(rng.gamma(shape+low))
    u=rng.random(size)
    # Generator emits [0,1); an exact zero is rerolled, not replaced by a cutoff.
    while np.any(u==0):
        mask=u==0;u[mask]=rng.random(np.count_nonzero(mask))
    ans+=np.where(low,np.log(u)/shape,0)
    return ans


def _log_gamma_coordinate_density(logx, shapes, scale):
    z=logx-math.log(scale)
    with np.errstate(over='ignore'):
        return np.sum(shapes*z-np.exp(z)-gammaln(shapes),axis=1)


def _positive_logistic_sample(n,center,width,rng):
    f0=expit(-center/width)
    u=f0+(1-f0)*rng.random(n)
    return center+width*logit(np.minimum(u,np.nextafter(1.,0.)))


def _positive_logistic_logpdf(s,center,width):
    z=(s-center)/width
    return -math.log(width)-z-2*np.logaddexp(0.,-z)+np.logaddexp(0.,-center/width)


def _ordered_simplex_pairs(n,m,rng):
    """Two uniform simplex points, conditioned on prefix(b)>=prefix(a).

    The event has exact probability 1/m (Catalan/ballot interleavings of two
    independent sets of m-1 uniform order statistics). For m=1 it is vacuous.
    """
    aa=[];bb=[];need=n;attempts=0
    while need:
        batch=max(64,int(need*m*1.15))
        a=rng.dirichlet(np.ones(m),batch)
        b=rng.dirichlet(np.ones(m),batch)
        good=np.all(np.cumsum(b,axis=1)[:,:-1]>=np.cumsum(a,axis=1)[:,:-1],axis=1)
        ia=np.flatnonzero(good)[:need]
        aa.append(a[ia]);bb.append(b[ia]);need-=len(ia);attempts+=batch
    return np.concatenate(aa),np.concatenate(bb),attempts


def _face_logdensity(loga,logb,scale,center,width):
    ya=math.log(scale)-loga;yb=math.log(scale)-logb
    sa=ya.sum(1);sb=yb.sum(1);m=ya.shape[1]
    valid=np.all(ya>=0,axis=1)&np.all(yb>=0,axis=1)&(sa>0)&(sb>0)
    ans=np.full(len(ya),-np.inf)
    idx=np.flatnonzero(valid)
    if not len(idx):return ans
    a=ya[idx]/sa[idx,None];b=yb[idx]/sb[idx,None]
    ordered=np.all(np.cumsum(b,axis=1)[:,:-1]>=np.cumsum(a,axis=1)[:,:-1],axis=1)
    idx=idx[ordered]
    ans[idx]=(math.log(m)+2*gammaln(m)
              +_positive_logistic_logpdf(sa[idx],center,width)
              +_positive_logistic_logpdf(sb[idx],center,width)
              -(m-1)*(np.log(sa[idx])+np.log(sb[idx])))
    return ans


def _shifted_logdensity(loga,logb,scale,center,width,shift):
    ya=math.log(scale)-loga+shift;yb=math.log(scale)-logb+shift
    sa=ya.sum(1);sb=yb.sum(1);m=ya.shape[1]
    valid=np.all(ya>=0,axis=1)&np.all(yb>=0,axis=1)&(sa>0)&(sb>0)
    ans=np.full(len(ya),-np.inf);idx=np.flatnonzero(valid)
    ans[idx]=(2*gammaln(m)+_positive_logistic_logpdf(sa[idx],center+m*shift,width)
              +_positive_logistic_logpdf(sb[idx],center+m*shift,width)
              -(m-1)*(np.log(sa[idx])+np.log(sb[idx])))
    return ans


def _ridge_logdensity(loga,logb,scale,center,width,shift):
    ya=math.log(scale)-loga[:,0];yb=math.log(scale)-logb[:,0]
    valid=(ya>=-shift)&(ya<=center+shift)&(yb>=-shift)
    ans=np.full(len(ya),-np.inf)
    ans[valid]=-math.log(center+2*shift)+_positive_logistic_logpdf(yb[valid]+shift,center+shift,width)
    return ans


def sample_log_diagonals(n, widths, beta, log_t_design, rng, *,
                         method="schubert_mixture",face_probability=None,
                         logistic_width=3.0,tilt_strength=1.0,
                         ordinary_probability=0.05,shifted_probability=0.3,
                         log_coordinate_shift=4.0,ridge_probability=0.9,
                         fixed_allocation=False):
    """Return loga,logb,logimportance,metadata for an exact positive estimator.

    widths must begin with 2 and all later widths must be >=2. a_j is the first
    Bartlett diagonal square for j=2..N; b_j is the second for j=1..N-1.
    Methods: ordinary, gamma_tilt, schubert_mixture. Each proposal has a known
    normalized density. `ordinary` is analytic conditioning without IS.
    """
    widths=tuple(map(int,widths));beta=int(beta);N=len(widths)-1;m=N-1
    if N<2 or widths[0]!=2 or min(widths)<2 or beta not in (1,2):
        raise ValueError("Require N>=2, widths[0]=2, all widths>=2, beta=1 or2")
    if method not in ("ordinary","gamma_tilt","schubert_mixture","shifted_window_mixture","ridge_mixture"):
        raise ValueError(method)
    if method=='ridge_mixture' and m!=1:raise ValueError('Ridge proposal is for two factors only')
    scale=2/beta
    aa=beta*np.array(widths[2:])/2
    ab=beta*(np.array(widths[1:-1])-1)/2
    center=max(float(log_t_design)+N*math.log(scale),0.1)
    ka=np.minimum(aa,tilt_strength/max(center,1.))
    kb=np.minimum(ab,tilt_strength/max(center,1.))
    if method=="ordinary":
        a=_loggamma_sample(aa,(n,m),rng)+math.log(scale)
        b=_loggamma_sample(ab,(n,m),rng)+math.log(scale)
        return a,b,np.zeros(n),{'method':method,'face_count':0,'draws':n}
    a=_loggamma_sample(ka,(n,m),rng)+math.log(scale)
    b=_loggamma_sample(kb,(n,m),rng)+math.log(scale)
    if ordinary_probability<0 or ordinary_probability>=1:raise ValueError('Bad ordinary mixture probability')
    if face_probability is None:face_probability=.6 if method=='shifted_window_mixture' else .9
    pf=face_probability if method in ('schubert_mixture','shifted_window_mixture') else 0.
    ps=shifted_probability if method=='shifted_window_mixture' else 0.
    pr=ridge_probability if method=='ridge_mixture' else 0.
    if min(pf,ps,pr)<0 or pf+ps+pr+ordinary_probability>=1:raise ValueError('Mixture must retain positive tilted support')
    allocation_counts=None
    if fixed_allocation:
        probabilities=np.array([pf,ps,pr,ordinary_probability,1-pf-ps-pr-ordinary_probability])
        raw=n*probabilities;counts=np.floor(raw).astype(int)
        for k in np.argsort(-(raw-counts))[:n-counts.sum()]:counts[k]+=1
        if np.any((probabilities>0)&(counts<2)):raise ValueError('Each active stratum needs at least two samples')
        pf,ps,pr,ordinary_probability,_=counts/n
        probs=counts/n;edges=np.cumsum(np.r_[0.,probs])
        assignment=np.repeat((edges[:-1]+edges[1:])/2,counts)
        allocation_counts=dict(zip(['ordered','shifted','ridge','ordinary','tilted'],map(int,counts)))
    else:
        assignment=rng.random(n)
    face=assignment<pf
    shifted=(assignment>=pf)&(assignment<pf+ps)
    ridge=(assignment>=pf+ps)&(assignment<pf+ps+pr)
    ordinary=(assignment>=pf+ps+pr)&(assignment<pf+ps+pr+ordinary_probability)
    no=int(ordinary.sum())
    if no:
        a[ordinary]=_loggamma_sample(aa,(no,m),rng)+math.log(scale)
        b[ordinary]=_loggamma_sample(ab,(no,m),rng)+math.log(scale)
    attempts=0
    if face.any():
        idx=np.flatnonzero(face);size=len(idx)
        sa=_positive_logistic_sample(size,center,logistic_width,rng)
        sb=_positive_logistic_sample(size,center,logistic_width,rng)
        ua,ub,attempts=_ordered_simplex_pairs(size,m,rng)
        a[idx]=math.log(scale)-sa[:,None]*ua
        b[idx]=math.log(scale)-sb[:,None]*ub
    if shifted.any():
        idx=np.flatnonzero(shifted);size=len(idx)
        sa=_positive_logistic_sample(size,center+m*log_coordinate_shift,logistic_width,rng)
        sb=_positive_logistic_sample(size,center+m*log_coordinate_shift,logistic_width,rng)
        ua=rng.dirichlet(np.ones(m),size);ub=rng.dirichlet(np.ones(m),size)
        a[idx]=math.log(scale)+log_coordinate_shift-sa[:,None]*ua
        b[idx]=math.log(scale)+log_coordinate_shift-sb[:,None]*ub
    if ridge.any():
        idx=np.flatnonzero(ridge);size=len(idx)
        ya=rng.uniform(-log_coordinate_shift,center+log_coordinate_shift,size)
        yb=_positive_logistic_sample(size,center+log_coordinate_shift,logistic_width,rng)-log_coordinate_shift
        a[idx,0]=math.log(scale)-ya;b[idx,0]=math.log(scale)-yb
    logprior=(_log_gamma_coordinate_density(a,aa,scale)+
              _log_gamma_coordinate_density(b,ab,scale))
    logq=(_log_gamma_coordinate_density(a,ka,scale)+
          _log_gamma_coordinate_density(b,kb,scale))
    logq=math.log(1-pf-ps-pr-ordinary_probability)+logq
    if ordinary_probability:
        logq=np.logaddexp(logq,math.log(ordinary_probability)+logprior)
    if pf:
        logface=_face_logdensity(a,b,scale,center,logistic_width)
        logq=np.logaddexp(logq,math.log(pf)+logface)
    if ps:
        logshift=_shifted_logdensity(a,b,scale,center,logistic_width,log_coordinate_shift)
        logq=np.logaddexp(logq,math.log(ps)+logshift)
    if pr:
        logridge=_ridge_logdensity(a,b,scale,center,logistic_width,log_coordinate_shift)
        logq=np.logaddexp(logq,math.log(pr)+logridge)
    return a,b,logprior-logq,{'method':method,'face_count':int(face.sum()),
            'draws':n,'ordinary_count':no,'ordinary_probability':ordinary_probability,
            'shifted_count':int(shifted.sum()),'shifted_probability':ps,'log_coordinate_shift':log_coordinate_shift,
            'ridge_count':int(ridge.sum()),'ridge_probability':pr,
            'fixed_allocation':fixed_allocation,'allocation_counts':allocation_counts,
            'window_proposal_attempts':attempts,'simplex_dimension_each':m-1,
            'window_acceptance_probability':1/m,'face_probability':pf,
            'design_log_t':float(log_t_design),'total_log_scale_center':center,
            'logistic_width':logistic_width,'tilted_shapes_a':ka.tolist(),'tilted_shapes_b':kb.tolist()}


def conditional_log_kernel(loga,logb,log_t,widths,beta):
    """Exact expectation over every offdiagonal and two endpoint diagonals."""
    beta=int(beta);m=loga.shape[1]
    if logb.shape!=loga.shape or len(widths)!=m+2:raise ValueError('shape mismatch')
    suffix=np.concatenate([np.cumsum(loga[:,::-1],axis=1)[:,::-1],np.zeros((len(loga),1))],axis=1)
    prefix=np.concatenate([np.zeros((len(logb),1)),np.cumsum(logb,axis=1)],axis=1)
    logv=logsumexp(suffix+prefix,axis=1)
    c=float(log_t)+math.log(2/beta)
    return (-beta*widths[1]/2*np.logaddexp(0,c+loga.sum(1))
            -beta*(widths[-1]-1)/2*np.logaddexp(0,c+logb.sum(1))
            -beta/2*np.logaddexp(0,c+logv))


def log_contributions(n,widths,beta,log_t,rng,**proposal):
    """Convenience single-t interface returning per-sample log contributions."""
    a,b,lw,meta=sample_log_diagonals(n,widths,beta,log_t,rng,**proposal)
    return conditional_log_kernel(a,b,log_t,widths,beta)+lw,lw,meta


def stratified_log_contributions(n,widths,beta,log_t,rng,**proposal):
    """Fixed component counts, balance weights; use stratified variance below.

    Existing iid APIs are unchanged by default. Each active mixture component
    receives a deterministic sample count, and actual counts/n replace nominal
    weights in the balance-mixture denominator.
    """
    proposal=dict(proposal);proposal['fixed_allocation']=True
    a,b,lw,meta=sample_log_diagonals(n,widths,beta,log_t,rng,**proposal)
    if meta.get('allocation_counts') is None:
        labels=np.zeros(n,dtype=int)
    else:
        labels=np.repeat(np.arange(5),list(meta['allocation_counts'].values()))
    return conditional_log_kernel(a,b,log_t,widths,beta)+lw,lw,labels,meta


def summarize_stratified(logvalues,labels):
    """Sum alpha_k² s_k²/n_k; never use pooled iid variance for this design."""
    logvalues=np.asarray(logvalues);labels=np.asarray(labels)
    n=len(logvalues);offset=float(logvalues.max());v=np.exp(logvalues-offset)
    mean=float(v.mean());variance=0.;strata=[]
    for label in np.unique(labels):
        x=v[labels==label];nk=len(x);alpha=nk/n
        if nk<2:raise ValueError('Need >=2 per active stratum')
        sk2=float(x.var(ddof=1));variance+=alpha*alpha*sk2/nk
        strata.append({'label':int(label),'n':nk,'allocation':alpha,
                       'relative_mean_contribution':float(alpha*x.mean()/mean),
                       'scaled_sample_variance':sk2})
    logmean=offset+math.log(mean);relative_se=math.sqrt(variance)/mean
    return {'n':n,'log_estimate':logmean,'estimate':math.exp(logmean),
            'relative_se':relative_se,'standard_error':math.exp(logmean)*relative_se,
            'contribution_ess':float(v.sum()**2/(v@v)),
            'maximum_fraction':float(v.max()/v.sum()),'strata':strata,
            'variance_formula':'sum_k (n_k/n)^2 samplevar_k/n_k',
            'pooled_iid_relative_se_for_comparison':float(v.std(ddof=1)/(mean*math.sqrt(n)))}


def summarize_log_contributions(logvalues):
    """Unbiased mean with ordinary iid sample variance; no self-normalization."""
    logvalues=np.asarray(logvalues);n=len(logvalues);offset=float(np.max(logvalues))
    values=np.exp(logvalues-offset);mean=float(values.mean())
    sd=float(values.std(ddof=1));logmean=offset+math.log(mean)
    relative_se=sd/(mean*math.sqrt(n))
    return {'n':n,'log_estimate':logmean,'estimate':math.exp(logmean),
            'relative_se':relative_se,'standard_error':math.exp(logmean)*relative_se,
            'contribution_ess':float(values.sum()**2/(values@values)),
            'maximum_fraction':float(values.max()/values.sum()),
            'sample_skewness':float(np.mean((values-mean)**3)/(sd**3)) if sd else 0}


def gaussian_products(n,widths,beta,rng):
    """Independent direct matrix sampling, with exact entry normalization."""
    p=np.broadcast_to(np.eye(widths[0]),(n,widths[0],widths[0])).copy()
    for rows,cols in zip(widths[1:],widths[:-1]):
        a=rng.standard_normal((n,rows,cols))
        if beta==2:a=(a+1j*rng.standard_normal((n,rows,cols)))/math.sqrt(2)
        p=a@p
    return p


def final_matrix_log_kernel(product_before_last,log_t,final_rows,beta):
    """Exact final-Gaussian-matrix integration for Z(t)=E exp(-t||P||²)."""
    sv=np.linalg.svd(product_before_last,compute_uv=False)
    with np.errstate(divide='ignore'):
        logs=2*np.log(sv)+float(log_t)+math.log(2/beta)
    return -beta*final_rows/2*np.sum(np.logaddexp(0.,logs),axis=1)
