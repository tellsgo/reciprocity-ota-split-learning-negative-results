"""One frozen small-task gate; temporal statistics measured, oracle separately labeled."""
import hashlib, json, pathlib, platform, time, csv
import numpy as np
from scipy.optimize import minimize

ROOT=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/config.json').read_text())
np.seterr(all='raise')

def cn(rng,shape): return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
def mm(a,b): return np.einsum('ij,jk->ik',a,b)
def mv(a,b): return np.einsum('ij,bj->bi',a,b)
def pb(h,s): return np.einsum('bmn,bm->bn',h.conj(),s)
def pgrad(gx,a): return np.einsum('bn,bi->ni',gx,a)/len(a)

def qreal(v,bound,signed=True):
    levels=2**C['control_bits_per_real']-1
    low=-bound if signed else 0.
    clips=int(np.count_nonzero((v<low)|(v>bound)))
    q=np.rint((np.clip(v,low,bound)-low)/(bound-low)*levels)/levels*(bound-low)+low
    return q,clips

def qcomplex(v,bound):
    re,cr=qreal(v.real,bound); im,ci=qreal(v.imag,bound)
    return re+1j*im,cr+ci

def task(y,lab,w,b):
    u=np.tanh(abs(y)**2)
    z=b+np.einsum('bi,i->b',u,w)
    prob=1/(1+np.exp(-z))
    err=prob-lab
    loss=np.logaddexp(0,z)-lab*z
    seed=2*err[:,None]*w[None,:]*(1-u*u)*y
    gw=np.einsum('b,bi->i',err,u)/len(y)+C['receiver_l2']*w
    gb=err.mean()
    return loss,seed,gw,gb,prob

class World:
    """Only forward(), backward(), pilot_forward() expose training observations."""
    def forward(self,x,ids): return np.einsum('bmn,bn->bm',self._f[ids],x)+self._nf[ids]
    def backward(self,s,ids): return pb(self._b[ids],s)+self._nb[ids]
    def pilot_forward(self,ids): return self._f[ids]+self._np[ids]
    def service(self,W,w,b,sigma=0.):
        x=mv(W,self.test_inputs)
        if sigma: x=x+sigma*self.test_epsilon
        norms=np.linalg.norm(x,axis=1)
        clipped=int(np.count_nonzero(norms>1))
        x=x/np.maximum(1,norms)[:,None]
        y=np.einsum('bmn,bn->bm',self._test_f,x)+self._test_n
        loss,_,_,_,prob=task(y,self.test_labels,w,b)
        return {'ce':float(loss.mean()),'accuracy':float(np.mean((prob>=.5)==self.test_labels)),
                'mean_tx_energy':float(np.mean(np.sum(abs(x)**2,axis=1))),
                'max_tx_energy':float(np.max(np.sum(abs(x)**2,axis=1))),
                'perturbed_clipping_count':clipped}

def initial(seed):
    rng=stream_rng(seed,'initialization')
    W=cn(rng,(2,3)); W=W/np.linalg.norm(W)*C['initial_encoder_norm']
    return W,np.full(C['nr'],.8),-.4

class SGD:
    def __init__(self,shape,dtype): pass
    def step(self,x,g,lr): return x-lr*g

class Adam:
    def __init__(self,shape,dtype): self.m=np.zeros(shape,dtype=dtype); self.v=np.zeros(shape); self.t=0
    def step(self,x,g,lr):
        self.t+=1
        self.m=C['adam_beta1']*self.m+(1-C['adam_beta1'])*g
        self.v=C['adam_beta2']*self.v+(1-C['adam_beta2'])*abs(g)**2
        return x-lr*(self.m/(1-C['adam_beta1']**self.t))/(np.sqrt(self.v/(1-C['adam_beta2']**self.t))+C['adam_eps'])

def verify():
    rng=stream_rng(505018889,'diagnostics')
    a=rng.choice([-1.,1.],size=(29,3)); lab=(a[:,0]*a[:,1]>0).astype(float)
    f=cn(rng,(29,C['nr'],2)); noise=.1*cn(rng,(29,C['nr'])); W,w,b=initial(505018890)
    def obj(W,w,b):
        y=np.einsum('bmn,bn->bm',f,mv(W,a))+noise
        return task(y,lab,w,b)[0].mean()+C['receiver_l2']*np.sum(w*w)/2
    y=np.einsum('bmn,bn->bm',f,mv(W,a))+noise
    _,s,gw,gb,_=task(y,lab,w,b); gW=pgrad(pb(f,s),a)
    errors=[]; eps=1e-6
    for ix in np.ndindex(W.shape):
        for part in [1.,1j]:
            d=np.zeros_like(W); d[ix]=part
            errors.append(abs((obj(W+eps*d,w,b)-obj(W-eps*d,w,b))/(2*eps)-np.vdot(gW,d).real))
    for j in range(C['nr']):
        d=np.zeros(C['nr']); d[j]=eps
        errors.append(abs((obj(W,w+d,b)-obj(W,w-d,b))/(2*eps)-gw[j]))
    errors.append(abs((obj(W,w,b+eps)-obj(W,w,b-eps))/(2*eps)-gb))
    assert max(errors)<1e-7
    # Complex score factor/conjugation check under a known linear loss.
    epsil=.2*cn(rng,(200000,2)); q=np.array([.7+.2j,-.3+.4j])
    loss=np.einsum('j,bj->b',q.conj(),epsil).real
    score=2*loss[:,None]*epsil/(.2**2)
    dif=score-q
    vals=np.concatenate([dif.real,dif.imag],axis=1)
    zs=abs(vals.mean(axis=0))/(vals.std(axis=0,ddof=1)/np.sqrt(len(vals)))
    assert zs.max()<5
    return {'max_fd_abs':float(max(errors)),'score_linear_moment_max_z':float(zs.max())}

def train(world,seed,case,method,true_a):
    W,w,b=initial(seed); initialW=W.copy()
    initial_service=world.service(W,w,b)
    calpairs=0; sharingbits=0; corr=np.eye(C['nr'],dtype=complex); alpha=1.
    calinfo={}; calibration_error=None; oracle=False
    if method=='scalar_est':
        calpairs=C['scalar_calibration_pairs']; ahat,corr,alpha,calinfo=fit_temporal(world,calpairs,True)
        calibration_error=float(np.linalg.norm(ahat-true_a)/np.linalg.norm(true_a))
    if method in ['matrix_est','shift_zero','shift_phase']:
        calpairs=0 if method=='shift_zero' else C['matrix_calibration_pairs']; ahat,corr,alpha,calinfo=fit_temporal(world,calpairs)
        sharingbits=0 if method=='shift_zero' else (16 if method=='shift_phase' else 2*C['nr']**2*C['control_bits_per_real'])
        calibration_error=float(np.linalg.norm(ahat-true_a)/np.linalg.norm(true_a))
    if method=='matrix_oracle':
        # Sole algorithm arm that may consume true temporal A. Never an instant F oracle.
        corr=np.linalg.pinv(true_a.conj().T,rcond=C['exact_pseudoinverse_relative_cutoff']); oracle=True
        calibration_error=0.
    physical=method in ['raw','scalar_est','matrix_est','matrix_oracle','shift_zero','shift_phase']
    csi=method.startswith('csi')
    scoremethod=method.startswith('score_')
    if physical:
        n=C['fixed_training_examples']
        if oracle: n=(C['rf_observation_quota']-5*C['matrix_calibration_pairs'])//2
    elif csi: n=C['rf_observation_quota']//3 if method=='csi_rf_budget' else 4000
    else: n=C['rf_observation_quota']
    gainsbits=16*n if physical and method not in ['shift_zero','shift_phase'] else 0
    lossbits=16*n if scoremethod else 0
    gradientbits=64*n if csi else 0
    costs={'forward_data_observations':n,'reverse_seed_observations':n if physical else 0,
           'current_csi_pilot_observations':2*n if csi else 0,
           'temporal_calibration_pilot_observations':5*calpairs,
           'calibration_channel_sharing_bits':(2*C['nr']*C['nt']*C['control_bits_per_real'])*calpairs,'correction_matrix_sharing_bits':sharingbits,
           'seed_norm_bits':gainsbits,'loss_feedback_bits':lossbits,'digital_gradient_bits':gradientbits,
           'training_labels':n,'temporal_pairs':calpairs,'oracle_statistics':oracle}
    costs['rf_observations']=sum(costs[k] for k in ['forward_data_observations','reverse_seed_observations','current_csi_pilot_observations','temporal_calibration_pilot_observations'])
    costs['control_bits']=(2*C['nr']*C['nt']*C['control_bits_per_real'])*calpairs+sharingbits+gainsbits+lossbits+gradientbits
    for rate in C['control_bits_per_rf_slot_scenarios']:
        costs[f'serial_slots_if_{rate}_bits_per_rf_duration']=costs['rf_observations']+costs['control_bits']/rate
    gradient_trace=[]; trace=[]; clips=0; gainclips=0; scoreclip=0; mean_energy=0.; reverse_energy=0.
    rx_fit=None
    input_seed_energy=0.; retained_seed_energy=0.; restored_noise_variance=0.; max_reverse_energy=0.
    if method=='rx_only':
        ids=np.arange(n); x=mv(W,world.inputs[:n]); y=world.forward(x,ids)
        features=np.tanh(abs(y)**2); labels=world.labels[:n]
        def fun(v):
            zz=v[C['nr']]+np.einsum('bi,i->b',features,v[:C['nr']])
            er=1/(1+np.exp(-zz))-labels
            val=np.mean(np.logaddexp(0,zz)-labels*zz)+C['receiver_l2']*np.sum(v[:C['nr']]**2)/2
            g=np.r_[np.einsum('b,bi->i',er,features)/n+C['receiver_l2']*v[:C['nr']],er.mean()]
            return val,g
        opt=minimize(fun,np.r_[w,b],jac=True,method='L-BFGS-B',bounds=[(-4,4)]*(C['nr']+1),
                     options={'maxiter':300,'gtol':1e-9,'ftol':1e-12})
        w=opt.x[:C['nr']]; b=float(opt.x[C['nr']]); mean_energy=float(np.mean(np.sum(abs(x)**2,axis=1)))
        rx_fit={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'jac_norm':float(np.linalg.norm(opt.jac))}
        assert opt.success
    else:
        Opt=SGD if C['optimizer']=='SGD' else Adam
        aw=Opt(W.shape,complex); ar=Opt((C['nr']+1,),float)
        baseline=np.full(8,np.log(2.))
        prng=stream_rng(seed,'score')
        # The receiver-coordinate bound gives a deterministic public norm bound.
        seed_bound=2*np.sqrt(C['nr'])*C['receiver_coordinate_bound']*(1. if method=='shift_zero' else np.linalg.norm(corr,2))
        for start in range(0,n,C['batch_size']):
            ids=np.arange(start,min(start+C['batch_size'],n)); inp=world.inputs[ids]; lab=world.labels[ids]
            x=mv(W,inp); latent_eps=None
            if scoremethod:
                sig=float(method.split('_')[1]); latent_eps=sig*cn(prng,x.shape)
                x=x+latent_eps
                norms=np.linalg.norm(x,axis=1)
                scoreclip+=int(np.count_nonzero(norms>1)); x=x/np.maximum(1,norms)[:,None]
            assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-12
            mean_energy+=float(np.sum(abs(x)**2))
            y=world.forward(x,ids); loss,s,gw,gb,_=task(y,lab,w,b)
            if physical:
                seeds=s if method=='shift_zero' else mv(corr,s)
                input_seed_energy+=float(np.sum(abs(s)**2))
                retained_seed_energy+=float(np.sum(abs(seeds)**2))
                norms=np.linalg.norm(seeds,axis=1)
                assert norms.max()<=seed_bound+1e-10
                if method in ['shift_zero','shift_phase']:
                    tx=seeds/seed_bound
                    gain=np.full(len(ids),seed_bound)
                else:
                    tx=seeds/np.maximum(norms,1e-15)[:,None]
                    gain,gc=qreal(norms,seed_bound,False); gainclips+=gc
                assert np.max(np.sum(abs(tx)**2,axis=1))<=1+1e-12
                reverse_energy+=float(np.sum(abs(tx)**2))
                max_reverse_energy=max(max_reverse_energy,float(np.max(np.sum(abs(tx)**2,axis=1))))
                restored_noise_variance+=float(np.sum(gain**2))*C['reverse_noise_std']**2/abs(alpha)**2
                gx=np.zeros((len(ids),2),dtype=complex) if alpha==0 else world.backward(tx,ids)*gain[:,None]/np.conj(alpha)
            elif csi:
                # Perfect instantaneous CSI oracle, as explicitly requested in G31.
                gx,cc=qcomplex(pb(world._f[ids],s),C['digital_gradient_real_quant_bound']); clips+=cc
            else:
                # Score is of latent Gaussian, NOT clipped displacement.
                feedback,cc=qreal(loss,C['loss_quant_bound'],False); clips+=cc
                pattern=np.sum((inp>0)*np.array([1,2,4]),axis=1).astype(int)
                previous=baseline[pattern].copy()
                gx=2*(feedback-previous)[:,None]*latent_eps/(sig*sig)
                for j in range(8):
                    selected=pattern==j
                    if selected.any(): baseline[j]=C['score_baseline_decay']*baseline[j]+(1-C['score_baseline_decay'])*feedback[selected].mean()
            gW=pgrad(gx,inp)
            true_gW=pgrad(pb(world._f[ids],s),inp)
            dot=float(np.vdot(true_gW,gW).real); den=float(np.linalg.norm(true_gW)*np.linalg.norm(gW))
            gradient_trace.append(dict(step=start//C['batch_size'],dot=dot,cosine=dot/max(den,1e-30),true_real=true_gW.real.tolist(),true_imag=true_gW.imag.tolist(),estimated_real=gW.real.tolist(),estimated_imag=gW.imag.tolist()))
            oldW=W.copy()
            W=aw.step(W,gW,C['encoder_adam_lr'])
            unprojected_step=float(np.linalg.norm(W-oldW))
            projected=bool(np.linalg.norm(W)>C['encoder_frobenius_bound'])
            W=W*min(1.,C['encoder_frobenius_bound']/max(np.linalg.norm(W),1e-15))
            params=ar.step(np.r_[w,b],np.r_[gw,gb],C['receiver_adam_lr'])
            params=np.clip(params,-C['receiver_coordinate_bound'],C['receiver_coordinate_bound'])
            w=params[:C['nr']]; b=float(params[C['nr']])
            assert np.isfinite(W).all() and np.isfinite(w).all() and np.isfinite(b)
            if start==0 or (start//C['batch_size']+1)%20==0 or ids[-1]==n-1:
                trace.append({'seen_samples':int(ids[-1]+1),'training_ce':float(loss.mean()),'encoder_norm':float(np.linalg.norm(W)),'unprojected_step':unprojected_step,'encoder_projected':projected})
        mean_energy/=n; reverse_energy=reverse_energy/n if physical else 0.
    out={'seed':seed,'case':case,'method':method,'initial_service':initial_service,
         'service':world.service(W,w,b),'costs':costs,'calibration_relative_error':calibration_error,
         'calibration':calinfo,'gradient_or_loss_quant_clips':clips,'gain_quant_clips':gainclips,
         'score_train_clip_fraction':scoreclip/n,'mean_train_forward_energy':mean_energy,
         'mean_reverse_energy':reverse_energy,'max_reverse_energy':max_reverse_energy,
         'seed_input_energy':input_seed_energy,'seed_retained_energy':retained_seed_energy,
         'discarded_seed_energy_fraction':None if not physical else max(0.,1-retained_seed_energy/input_seed_energy),
         'restored_noise_variance_per_complex_coordinate':restored_noise_variance/n,'receiver_fit':rx_fit,'trace':trace,'gradient_trace':gradient_trace,
         'initial_W_real':initialW.real.tolist(),'initial_W_imag':initialW.imag.tolist(),
         'final_W_real':W.real.tolist(),'final_W_imag':W.imag.tolist(),'final_receiver_w':w.tolist(),'final_receiver_b':b}
    if scoremethod: out['smoothed_test_service']=world.service(W,w,b,float(method.split('_')[1]))
    return out
