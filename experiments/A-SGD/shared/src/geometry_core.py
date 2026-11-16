"""Frozen shared-angular-field geometry probe. No optimizer or selected transition A."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import pathlib,json,time,resource,signal,hashlib,platform,csv
import numpy as np
from scipy.special import j0,iv

ROOT=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/config.json').read_text())
resource.setrlimit(resource.RLIMIT_CPU,(C['cpu_limit_seconds'],C['cpu_limit_seconds']))
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall cap')))
signal.alarm(C['wall_limit_seconds'])
np.seterr(all='raise')

def cn(rng,shape): return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
def mm(a,b): return np.einsum('ij,jk->ik',a,b)
def mv(a,b): return np.einsum('ij,bj->bi',a,b)
def pb(h,s): return np.einsum('bmn,bm->bn',h.conj(),s)
def gs(g,a): return np.einsum('bn,bi->bni',g,a)
def pack(x): return {'real':x.real.tolist(),'imag':x.imag.tolist()}
def energy(x): return np.sum(abs(x)**2,axis=tuple(range(1,x.ndim)))
def qreal(x,bound,signed=True):
    lo=-bound if signed else 0.; levels=2**C['control_bits_per_real']-1
    clipped=int(np.count_nonzero((x<lo)|(x>bound)))
    if signed:
        half=2**(C['control_bits_per_real']-1)-1
        return np.rint(np.clip(x,-bound,bound)/bound*half)/half*bound,clipped
    return np.rint((np.clip(x,lo,bound)-lo)/(bound-lo)*levels)/levels*(bound-lo)+lo,clipped
def qcomplex(x,bound):
    r,a=qreal(x.real,bound);i,b=qreal(x.imag,bound);return r+1j*i,a+b

def scalar_post(alpha,estimated=False):
    mag=abs(alpha)
    if mag<C['scalar_zero_tolerance']: return 0j,{'unavailable_zero':True,'floor':False}
    floor=C['estimated_scalar_floor'] if estimated else 0.
    adjusted=alpha/mag*max(mag,floor)
    return 1/adjusted.conjugate(),{'unavailable_zero':False,'floor':bool(mag<floor)}

def correction(a,estimated=False):
    u,d,vh=np.linalg.svd(a)
    cutoff=C['estimated_pseudoinverse_absolute_cutoff'] if estimated else C['exact_pseudoinverse_relative_cutoff']*d[0]
    keep=d>cutoff; inverse=np.zeros_like(d);inverse[keep]=1/d[keep]
    out=np.einsum('ik,k,kj->ij',u,inverse,vh)
    clips=0
    if estimated: out,clips=qcomplex(out,C['correction_quant_bound'])
    return out,{'singular_values':d.tolist(),'rank':int(keep.sum()),'cutoff':float(cutoff),'quant_clips':clips}

def characteristic(dx,dy,spectrum):
    base=j0(2*np.pi*np.sqrt(dx*dx+dy*dy))
    if spectrum=='isotropic': return complex(base)
    k=C['asymmetric_kappa'];mu=np.deg2rad(C['asymmetric_center_degrees'])
    z=np.sqrt((k*np.cos(mu)-2j*np.pi*dx)**2+(k*np.sin(mu)-2j*np.pi*dy)**2+0j)
    clustered=iv(0,z)/iv(0,k)
    w=C['asymmetric_cluster_weight'];return w*clustered+(1-w)*base

def geometry(spectrum,direction,displacement):
    theta=2*np.pi*np.arange(C['angular_grid'])/C['angular_grid']
    mu=np.deg2rad(C['asymmetric_center_degrees'])
    if spectrum=='isotropic': weight=np.ones_like(theta)
    else:
        k=C['asymmetric_kappa'];eta=C['asymmetric_cluster_weight']
        weight=eta*np.exp(k*np.cos(theta-mu))/iv(0,k)+(1-eta)
    weight=weight/weight.sum()
    pos=np.arange(C['nr'])*C['spacing_lambda']
    delta=np.array([displacement,0.]) if direction=='parallel' else np.array([0.,displacement])
    phase=pos[:,None]*np.cos(theta)[None,:]
    s0=np.exp(-2j*np.pi*phase)*np.sqrt(weight)[None,:]
    st=s0*np.exp(2j*np.pi*(delta[0]*np.cos(theta)+delta[1]*np.sin(theta)))[None,:]
    r0=mm(s0,s0.conj().T);rt=mm(st,s0.conj().T)
    analytic0=np.array([[characteristic(pos[i]-pos[j],0.,spectrum) for j in range(3)] for i in range(3)])
    analytict=np.array([[characteristic(pos[i]-pos[j]-delta[0],-delta[1],spectrum) for j in range(3)] for i in range(3)])
    analytic_error=max(np.linalg.norm(r0-analytic0),np.linalg.norm(rt-analytict))/np.linalg.norm(r0)
    assert analytic_error<1e-10
    r0inv=np.linalg.pinv(r0,rcond=C['exact_pseudoinverse_relative_cutoff'])
    a=mm(rt,r0inv);q=r0-mm(mm(a,r0),a.conj().T);q=(q+q.conj().T)/2
    joint=np.block([[r0,rt.conj().T],[rt,r0]])
    q_eig=np.linalg.eigvalsh(q);joint_eig=np.linalg.eigvalsh(joint)
    assert q_eig.min()>-1e-10 and joint_eig.min()>-1e-10
    assert np.max(abs(np.diag(r0)-1))<1e-12
    assert np.linalg.norm(mm(st,st.conj().T)-r0)<1e-12
    if displacement==0: assert np.linalg.norm(a-np.eye(3))<1e-10
    c,ci=correction(a)
    alpha=np.trace(rt)/np.trace(r0)
    nonnormal=np.linalg.norm(mm(a,a.conj().T)-mm(a.conj().T,a))/max(np.linalg.norm(a)**2,1e-30)
    scalar_miss=np.linalg.norm(rt-alpha*r0)/np.linalg.norm(rt)
    pred_mse=float(np.trace(q).real)
    scalar_mse=float(np.trace(r0-mm(alpha*np.eye(3),rt.conj().T)-mm(rt,np.conj(alpha)*np.eye(3))+abs(alpha)**2*r0).real)
    info={'case':f'{spectrum}/{direction}/{displacement:.1f}','spectrum':spectrum,'direction':direction,'displacement_lambda':displacement,
          'R0':pack(r0),'Rtau':pack(rt),'A':pack(a),'Q':pack(q),'R0_eigenvalues':np.linalg.eigvalsh(r0).tolist(),
          'Q_eigenvalues':q_eig.tolist(),'joint_eigenvalues':joint_eig.tolist(),'R0_condition':float(np.linalg.cond(r0)),
          'A_condition':float(np.linalg.cond(a)),'A_nonnormality':float(nonnormal),'scalar_covariance_relative_miss':float(scalar_miss),
          'alpha':{'real':float(alpha.real),'imag':float(alpha.imag),'magnitude':float(abs(alpha))},
          'matrix_channel_prediction_mse_per_column':pred_mse,'scalar_channel_prediction_mse_per_column':scalar_mse,
          'analytic_grid_relative_error':float(analytic_error),'exact_correction':ci,
          'restoration_operator_residual':float(np.linalg.norm(mm(a.conj().T,c)-np.eye(3)))}
    return {'s0':s0,'st':st,'r0':r0,'rt':rt,'a':a,'q':q,'c':c,'alpha':alpha,'info':info}

def generate(g,steering): return np.einsum('rk,bnk->brn',steering,g,optimize=False)

def fit(fhat,bhat):
    f,clips=qcomplex(fhat,C['channel_quant_bound']);n=len(f)
    r=np.einsum('bmn,bln->ml',f,f.conj())/(n*C['nt'])-C['pilot_noise_std']**2*np.eye(3)
    cross=np.einsum('bmn,bln->ml',bhat,f.conj())/(n*C['nt'])
    reg=(r+r.conj().T)/2+C['calibration_ridge']*np.eye(3)
    d,v=np.linalg.eigh(reg);dc=np.maximum(d,C['calibration_covariance_floor'])
    inv=np.einsum('ik,k,jk->ij',v,1/dc,v.conj())
    ahat=mm(cross,inv)
    alpha=np.trace(cross)/np.sum(dc)
    corr,ci=correction(ahat,True)
    return ahat,alpha,corr,{'channel_quant_clips':clips,'covariance_floor_count':int(np.count_nonzero(d<C['calibration_covariance_floor'])),'correction':ci}

def task(f,a,labels,W,noise):
    x=mv(W,a);y=np.einsum('bmn,bn->bm',f,x)+noise
    u=np.tanh(abs(y)**2);w=np.array(C['receiver_weights']);z=C['receiver_bias']+np.einsum('bi,i->b',u,w)
    p=1/(1+np.exp(-z));loss=np.logaddexp(0,z)-labels*z
    seed=2*(p-labels)[:,None]*w[None,:]*(1-u*u)*y
    return x,seed,loss

def covariance_check(samples,target):
    difference=samples-target
    coords=np.concatenate([difference.real.reshape(len(samples),-1),difference.imag.reshape(len(samples),-1)],axis=1)
    mean=coords.mean(axis=0);se=coords.std(axis=0,ddof=1)/np.sqrt(len(coords))
    z=np.zeros_like(mean);usable=se>1e-12;z[usable]=abs(mean[usable])/se[usable]
    assert np.max(abs(mean[~usable]),initial=0)<1e-10
    return {'max_abs_mean_error':float(np.max(abs(mean))),'max_z':float(np.max(z)),
            'frobenius_error':float(np.linalg.norm(difference.mean(axis=0)))}

def model_checks(f,b,g):
    pred=np.einsum('ij,bjn->bin',g['a'],f);e=b-pred
    def outer(x,y,conjugate=True): return np.einsum('bmn,bln->bml',x,y.conj() if conjugate else y)/C['nt']
    out={'forward_cov':covariance_check(outer(f,f),g['r0']),
         'return_cov':covariance_check(outer(b,b),g['r0']),
         'cross_lag':covariance_check(outer(b,f),g['rt']),
         'residual_cov':covariance_check(outer(e,e),g['q']),
         'residual_forward':covariance_check(outer(e,f),np.zeros((3,3))),
         'forward_pseudocov':covariance_check(outer(f,f,False),np.zeros((3,3))),
         'residual_pseudocov':covariance_check(outer(e,e,False),np.zeros((3,3)))}
    return out

def arm(g,f,b,inputs,seed,trueg,noise,corr,post,method,diagnostic):
    n=len(inputs);v=mv(corr,seed);norm=np.linalg.norm(v,axis=1)
    bound=2*np.sqrt(3)*max(abs(np.array(C['receiver_weights'])))*np.linalg.norm(corr,2)
    if bound==0: gain=np.zeros(n);gc=0;tx=np.zeros_like(v)
    else:
        assert np.max(norm)<=bound+1e-10
        gain,gc=qreal(norm,bound,False)
        tx=v/np.maximum(norm,1e-30)[:,None]
    t=post*tx*gain[:,None]
    actual=gs(post*(pb(b,tx)+noise)*gain[:,None],inputs)
    mean=gs(pb(f,mv(g['a'].conj().T,t)),inputs)
    target=trueg.mean(axis=0);pred=mean.mean(axis=0)
    den=float(np.linalg.norm(target)**2);normden=max(den,1e-30)
    cos=float(np.vdot(target,pred).real/max(np.linalg.norm(target)*np.linalg.norm(pred),1e-30))
    bias_energy=energy(mean-trueg)
    innovation=np.einsum('bi,ij,bj->b',t.conj(),g['q'],t).real
    assert innovation.min()>-1e-9
    innovation=np.maximum(innovation,0)*C['nt']*np.sum(inputs*inputs,axis=1)
    receiver_noise=abs(post)**2*gain**2*C['reverse_noise_std']**2*C['nt']*np.sum(inputs*inputs,axis=1)
    ref_individual=float(energy(trueg).mean())
    innov_mean=float(innovation.mean());noise_mean=float(receiver_noise.mean())
    err=actual-mean
    coords=np.concatenate([err.real.reshape(n,-1),err.imag.reshape(n,-1)],axis=1)
    sem=coords.std(axis=0,ddof=1)/np.sqrt(n);empmean=coords.mean(axis=0)
    z=np.zeros_like(sem);nz=sem>1e-12;z[nz]=abs(empmean[nz])/sem[nz]
    unavailable=bool(post==0 or np.linalg.norm(corr)==0)
    return {'method':method,'diagnostic_population_statistics':diagnostic,'population_conditional_cosine':cos,
            'population_conditional_relative_bias':float(np.linalg.norm(pred-target)/np.sqrt(normden)),
            'population_actual_relative_error':float(np.linalg.norm(actual.mean(axis=0)-target)/np.sqrt(normden)),
            'conditional_individual_relative_rms_bias':float(np.sqrt(bias_energy.mean()/max(ref_individual,1e-30))),
            'individual_true_gradient_energy':ref_individual,'mean_true_gradient_energy':den,
            'individual_analytic_mse':float(bias_energy.mean()+innov_mean+noise_mean),
            'individual_empirical_mse':float(energy(actual-trueg).mean()),
            'innovation_variance':innov_mean,'return_receiver_noise_variance':noise_mean,
            'mean_gradient_variance_relative':float((innov_mean+noise_mean)/n/normden),
            'conditional_signal_energy':float(energy(mean).mean()),
            'residual_empirical_variance':float(energy(err).mean()),'residual_mean_max_z':float(z.max()),
            'restored_norm_mean_square':float(np.mean(abs(post)**2*gain**2)),
            'transmitted_seed_energy':float(np.mean(np.sum(abs(tx)**2,axis=1))),
            'max_transmitted_seed_energy':float(np.max(np.sum(abs(tx)**2,axis=1))),
            'norm_quant_clips':gc,'max_unscaled_seed_norm':float(norm.max()),'norm_public_bound':float(bound),
            'unavailable_zero_correction':unavailable}

def verify():
    assert scalar_post(0j,True)[0]==0 and scalar_post(0j,False)[0]==0
    p,flags=scalar_post(1e-5j,True);assert np.isfinite(p) and abs(p)==10 and flags['floor']
    singular=np.diag([1.,.5,0.]).astype(complex);cc,ci=correction(singular)
    assert ci['rank']==2 and np.linalg.norm(mm(singular.conj().T,cc)-np.diag([1,1,0]))<1e-12
    rng=np.random.default_rng(491000);f=cn(rng,(31,3,2));a=rng.choice([-1.,1.],(31,3));labels=(a[:,0]*a[:,1]>0)
    W=cn(rng,(2,3));W=W/np.linalg.norm(W)*.4;noise=.1*cn(rng,(31,3))
    _,s,_=task(f,a,labels,W,noise);gradient=gs(pb(f,s),a).mean(axis=0);errors=[]
    for ix in np.ndindex(W.shape):
        for part in [1,1j]:
            d=np.zeros_like(W);d[ix]=part;step=1e-6
            fd=(task(f,a,labels,W+step*d,noise)[2].mean()-task(f,a,labels,W-step*d,noise)[2].mean())/(2*step)
            errors.append(abs(fd-np.vdot(gradient,d).real))
    assert max(errors)<1e-7
    return {'scalar_exact_zero_guard':True,'scalar_small_complex_floor_guard':True,'singular_support_projection_guard':True,'finite_difference_max_abs':float(max(errors))}

def summarize(rows,geometries):
    allout=[]
    for g in geometries:
        case=g['info']['case'];rr=[r for r in rows if r['case']==case];out={'case':case,'geometry':g['info'],'arms':{}}
        for method in ['raw','scalar_exact','matrix_exact','scalar_est','matrix_est']:
            sub=[r['arms'][method] for r in rr]
            keys=['population_conditional_cosine','population_conditional_relative_bias','population_actual_relative_error',
                  'conditional_individual_relative_rms_bias','innovation_variance','return_receiver_noise_variance',
                  'mean_gradient_variance_relative','individual_analytic_mse','individual_empirical_mse','restored_norm_mean_square']
            out['arms'][method]={k:{'mean':float(np.mean([x[k] for x in sub])),'median':float(np.median([x[k] for x in sub])),
                                      'min':float(min(x[k] for x in sub)),'max':float(max(x[k] for x in sub))} for k in keys}
            out['arms'][method]['ascent_count']=sum(x['population_conditional_cosine']<0 for x in sub)
        out['mechanism_gates']={}
        for control in ['raw','scalar_exact']:
            base=np.array([r['arms'][control]['population_conditional_relative_bias'] for r in rr]);exact=np.array([r['arms']['matrix_exact']['population_conditional_relative_bias'] for r in rr]);paid=np.array([r['arms']['matrix_est']['population_conditional_relative_bias'] for r in rr])
            positive=bool(np.sum(base>C['mechanism_relative_bias_threshold'])>=12 and exact.mean()<.001 and g['info']['exact_correction']['rank']==3)
            out['mechanism_gates'][control]={'mechanism_positive':positive,'paid_positive':bool(positive and np.median(paid)<C['paid_bias_reduction_fraction']*np.median(base)),
                'control_seeds_above_bias_threshold':int(np.sum(base>C['mechanism_relative_bias_threshold'])),
                'median_paid_bias':float(np.median(paid)),'median_control_bias':float(np.median(base))}
        out['calibration']={'mean_relative_A_error':float(np.mean([r['calibration']['relative_A_error'] for r in rr])),
            'estimated_rank_counts':{str(rank):sum(r['calibration']['correction']['rank']==rank for r in rr) for rank in range(4)},
            'total_quant_clips':sum(r['calibration']['channel_quant_clips']+r['calibration']['correction']['quant_clips'] for r in rr)}
        out['max_model_check_z']=max(x['max_z'] for r in rr for x in r['model_checks'].values())
        out['max_gradient_residual_z']=max(x['residual_mean_max_z'] for r in rr for x in r['arms'].values())
        allout.append(out)
    return allout

def main():
    started=time.time();checks=verify();rows=[]
    geometries=[geometry(s,d,t) for s in C['spectra'] for d in C['directions'] for t in C['displacements_lambda']]
    frozen={r:hashlib.sha256((ROOT/r).read_bytes()).hexdigest() for r in ['configs/config.json','configs/SPEC.md','src/run_probe.py']}
    (ROOT/'results/freeze_manifest.json').write_text(json.dumps({'hashes':frozen,'task':C['task_name'],'pid':os.getpid()},indent=2))
    for seednum in C['seeds']:
        rng=np.random.default_rng(seednum);calrng=np.random.default_rng(seednum+10000);initrng=np.random.default_rng(seednum+20000)
        angular=cn(rng,(C['examples'],C['nt'],C['angular_grid']));calangular=cn(calrng,(C['calibration_pairs'],C['nt'],C['angular_grid']))
        inputs=rng.choice([-1.,1.],(C['examples'],3));labels=(inputs[:,0]*inputs[:,1]>0)
        nf=C['forward_noise_std']*cn(rng,(C['examples'],3));nb=C['reverse_noise_std']*cn(rng,(C['examples'],2))
        npf=C['pilot_noise_std']*cn(calrng,(C['calibration_pairs'],3,2));npb=C['pilot_noise_std']*cn(calrng,(C['calibration_pairs'],3,2))
        W=cn(initrng,(2,3));W=W/np.linalg.norm(W)*C['initial_encoder_norm']
        cached={}
        for g in geometries:
            spectrum=g['info']['spectrum']
            if spectrum not in cached: cached[spectrum]=(generate(angular,g['s0']),generate(calangular,g['s0']))
            f,cf=cached[spectrum];b=generate(angular,g['st']);cb=generate(calangular,g['st'])
            ahat,alpha_hat,c_est,calinfo=fit(cf+npf,cb+npb)
            calinfo['relative_A_error']=float(np.linalg.norm(ahat-g['a'])/max(np.linalg.norm(g['a']),1e-30))
            calinfo['estimated_A']=pack(ahat)
            x,s,loss=task(f,inputs,labels,W,nf);trueg=gs(pb(f,s),inputs)
            assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-12
            pe,se=scalar_post(g['alpha']);ph,sh=scalar_post(alpha_hat,True)
            choices={'raw':(np.eye(3),1.,False),'scalar_exact':(np.eye(3),pe,True),'matrix_exact':(g['c'],1.,True),
                     'scalar_est':(np.eye(3),ph,False),'matrix_est':(c_est,1.,False)}
            arms={m:arm(g,f,b,inputs,s,trueg,nb,co,po,m,diag) for m,(co,po,diag) in choices.items()}
            projector=mm(g['a'].conj().T,g['c'])
            support_residual=float(np.mean(energy(mv(np.eye(3)-projector,s)))/max(np.mean(energy(s)),1e-30))
            rows.append({'seed':seednum,'case':g['info']['case'],'loss':float(loss.mean()),'W':pack(W),
                'mean_forward_energy':float(np.mean(np.sum(abs(x)**2,axis=1))),'max_forward_energy':float(np.max(np.sum(abs(x)**2,axis=1))),
                'calibration':calinfo,'scalar_exact_status':se,'scalar_est_status':sh,'seed_unrestorable_fraction':support_residual,
                'model_checks':model_checks(f,b,g),'arms':arms})
        (ROOT/'results/partial_runs.json').write_text(json.dumps(rows))
        print(json.dumps({'completed_seed':seednum,'rows':len(rows),'wall_seconds':time.time()-started}),flush=True)
    summary=summarize(rows,geometries)
    usage=resource.getrusage(resource.RUSAGE_SELF)
    result={'metadata':{'host':platform.node(),'pid':os.getpid(),'task':C['task_name'],'hashes':frozen,'checks':checks,
        'wall_seconds':time.time()-started,'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'rows':len(rows),'config':C},'cases':summary}
    (ROOT/'results/all_runs.json').write_text(json.dumps(rows,indent=2));(ROOT/'results/summary.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'results/summary.csv').open('w') as file:
        columns=['case','method','mean_cosine','mean_bias','median_bias','mean_average_variance_relative','individual_MSE','A_condition','R0_condition']
        wr=csv.DictWriter(file,fieldnames=columns);wr.writeheader()
        for case in summary:
            for m,r in case['arms'].items():
                wr.writerow({'case':case['case'],'method':m,'mean_cosine':r['population_conditional_cosine']['mean'],
                    'mean_bias':r['population_conditional_relative_bias']['mean'],'median_bias':r['population_conditional_relative_bias']['median'],
                    'mean_average_variance_relative':r['mean_gradient_variance_relative']['mean'],'individual_MSE':r['individual_analytic_mse']['mean'],
                    'A_condition':case['geometry']['A_condition'],'R0_condition':case['geometry']['R0_condition']})
    print(json.dumps({'status':'complete','rows':len(rows),'wall_seconds':time.time()-started,'user_cpu':usage.ru_utime,
        'checks':checks,'max_covariance_z':max(x['max_model_check_z'] for x in summary),'max_gradient_z':max(x['max_gradient_residual_z'] for x in summary)}),flush=True)

if __name__=='__main__': main()
