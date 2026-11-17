"""Actual Frobenius CDF estimation by exact conditional exponential tilting.

Target F(exp(-L))=P(||P||_F²<=exp(-L)); no asymptotic coefficient is used.
The remaining diagonal importance proposals come from vr_estimators.py.
This is randomized tilted-Gamma conditioning, not fully integrated CDF
quadrature: one auxiliary Gamma triple is sampled per importance draw.
"""
import math
import numpy as np
from scipy.special import expit,logsumexp
from vr_estimators import sample_log_diagonals,conditional_log_kernel


def tilted_cdf_log_contributions(n,widths,beta,log_inverse_epsilon,rng,*,tilt_factor=1.0,**proposal):
    """Return log CDF contributions, importance weights and diagnostics.

    Conditional W is sum of three independent Gammas. Under tilt t=alpha/eps,
    U=tW has the same shapes and scales logistic(log(t theta_j)). The exact
    contribution is G_t * exp(U) * 1(U<=alpha), times p/q. alpha=1 is default.
    The entire procedure accepts L even when eps and F underflow in float64.
    """
    if tilt_factor<=0:raise ValueError('Positive tilt_factor required')
    if proposal.get('fixed_allocation',False):
        raise ValueError('CDF summary uses iid variance; fixed allocation requires separate stratum-aware analysis')
    logt=float(log_inverse_epsilon)+math.log(tilt_factor)
    a,b,lw,meta=sample_log_diagonals(n,widths,beta,logt,rng,**proposal)
    suffix=np.concatenate([np.cumsum(a[:,::-1],axis=1)[:,::-1],np.zeros((n,1))],axis=1)
    prefix=np.concatenate([np.zeros((n,1)),np.cumsum(b,axis=1)],axis=1)
    logv=logsumexp(suffix+prefix,axis=1)
    log_coefficients=np.column_stack([a.sum(1),b.sum(1),logv])+math.log(2/beta)
    shapes=np.array([beta*widths[1]/2,beta*(widths[-1]-1)/2,beta/2])
    tilted_u_scales=expit(logt+log_coefficients)
    u=np.sum(rng.gamma(shapes,size=(n,3))*tilted_u_scales,axis=1)
    accepted=u<=tilt_factor
    result=conditional_log_kernel(a,b,logt,widths,beta)+lw+u
    result[~accepted]=-np.inf
    meta={**meta,'target':'Frobenius_CDF','log_inverse_epsilon':float(log_inverse_epsilon),
          'tilt_factor':float(tilt_factor),'auxiliary_gamma_shapes':shapes.tolist(),
          'auxiliary_draws':3*n,'accepted_indicator_count':int(accepted.sum()),
          'indicator_acceptance_fraction':float(accepted.mean()),
          'conditional_multiplier_upper_bound':float(math.exp(tilt_factor)),
          'analytic_inputs':'Exact Bartlett law, conditional Gaussian law, Gamma exponential tilt only.'}
    return result,lw,meta


def summarize_cdf_logs(logvalues,log_scale=0.0):
    """iid sample moments of the unnormalized actual-CDF estimator."""
    logvalues=np.asarray(logvalues,float);finite=np.isfinite(logvalues);n=len(logvalues)
    if not finite.any():return {'n':n,'positive_count':0,'estimate':0.,'standard_error':None,'status':'NO_POSITIVE_CONTRIBUTIONS'}
    offset=float(logvalues[finite].max())
    x=np.exp(logvalues-offset);mean=float(x.mean());sd=float(x.std(ddof=1))
    logmean=offset+math.log(mean);rse=sd/(mean*math.sqrt(n));scaled_logmean=logmean+log_scale
    estimate=math.exp(scaled_logmean);se=estimate*rse
    return {'n':n,'positive_count':int(finite.sum()),'log_cdf_estimate':logmean,
            'log_scale':float(log_scale),'estimate':estimate,'standard_error':se,
            'relative_se':rse,'ci95':[max(0.,estimate-1.959963984540054*se),estimate+1.959963984540054*se],
            'contribution_ess':float(x.sum()**2/(x@x)),'maximum_fraction':float(x.max()/x.sum()),
            'status':'FINITE_IID_MONTE_CARLO_ESTIMATE'}
