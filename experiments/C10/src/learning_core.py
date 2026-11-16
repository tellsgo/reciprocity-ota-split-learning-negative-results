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

def make_a(seed,case):
    if case=='scalar': return C['scalar_rho']*np.eye(3,dtype=complex)
    rng=np.random.default_rng(C['matrix_seed_offset']+seed)
    u,_=np.linalg.qr(cn(rng,(3,3)))
    if case=='mild': return np.einsum('ik,k,jk->ij',u,np.array(C['mild_singular_values']),u.conj())
    v,_=np.linalg.qr(cn(rng,(3,3)))
    return np.einsum('ik,k,jk->ij',u,np.array(C['stress_singular_values']),v.conj())

def joint_return(a,f,z):
    q=np.eye(3)-np.einsum('ik,jk->ij',a,a.conj())
    d,v=np.linalg.eigh(q)
    assert d.min()>0
    qs=np.einsum('ik,k,jk->ij',v,np.sqrt(d),v.conj())
    return np.einsum('ij,bjn->bin',a,f)+np.einsum('ij,bjn->bin',qs,z)

class World:
    """Only forward(), backward(), pilot_forward() expose training observations."""
    def __init__(self,seed,a):
        rng=np.random.default_rng(C['train_seed_offset']+seed)
        n=C['rf_observation_quota']
        self.inputs=rng.choice([-1.,1.],size=(n,3))
        self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
        self._f=cn(rng,(n,3,2)); self._b=joint_return(a,self._f,cn(rng,(n,3,2)))
        self._nf=C['forward_noise_std']*cn(rng,(n,3))
        self._nb=C['reverse_noise_std']*cn(rng,(n,2))
        self._np=C['pilot_noise_std']*cn(rng,(n,3,2))
        calrng=np.random.default_rng(C['calibration_seed_offset']+seed)
        k=C['matrix_calibration_pairs']
        f=cn(calrng,(k,3,2)); b=joint_return(a,f,cn(calrng,(k,3,2)))
        self.cal_fhat=f+C['pilot_noise_std']*cn(calrng,f.shape)
        self.cal_bhat=b+C['pilot_noise_std']*cn(calrng,b.shape)
        trng=np.random.default_rng(C['test_seed_offset']+seed)
        self.test_inputs=trng.choice([-1.,1.],size=(C['test_samples'],3))
        self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
        self._test_f=cn(trng,(C['test_samples'],3,2))
        self._test_n=C['forward_noise_std']*cn(trng,(C['test_samples'],3))
        self.test_epsilon=cn(trng,(C['test_samples'],2))
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
    rng=np.random.default_rng(C['initialization_seed_offset']+seed)
    W=cn(rng,(2,3)); W=W/np.linalg.norm(W)*C['initial_encoder_norm']
    return W,np.array([.8,.8,.8]),-.4

def fit_temporal(world,k,scalar=False):
    # Fhat must cross from Rx to Tx. Bhat already resides at Tx.
    f,clip=qcomplex(world.cal_fhat[:k],C['channel_real_quant_bound'])
    b=world.cal_bhat[:k]
    xx=np.einsum('bmn,bln->ml',f,f.conj())/(k*2)-C['pilot_noise_std']**2*np.eye(3)
    cross=np.einsum('bmn,bln->ml',b,f.conj())/(k*2)
    reg=xx+C['eiv_ridge_per_pair']*np.eye(3)
    if scalar:
        alpha=np.trace(cross)/np.trace(reg)
        mag=abs(alpha)
        correction_alpha=alpha/max(mag,1e-12)*max(mag,C['scalar_magnitude_floor'])
        return alpha*np.eye(3),np.eye(3),correction_alpha,{'channel_clips':clip,'matrix_clips':0,'floor_count':int(mag<C['scalar_magnitude_floor'])}
    ahat=mm(cross,np.linalg.inv(reg))
    u,s,vh=np.linalg.svd(ahat)
    clipped=np.clip(s,C['correction_singular_floor'],C['correction_singular_cap'])
    corr=np.einsum('ik,k,kj->ij',u,1/clipped,vh)
    corr,mc=qcomplex(corr,C['correction_real_quant_bound'])
    return ahat,corr,1.,{'channel_clips':clip,'matrix_clips':mc,'floor_count':int(np.count_nonzero(s<C['correction_singular_floor'])),'singular_values':s.tolist()}

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
    rng=np.random.default_rng(132001)
    a=rng.choice([-1.,1.],size=(29,3)); lab=(a[:,0]*a[:,1]>0).astype(float)
    f=cn(rng,(29,3,2)); noise=.1*cn(rng,(29,3)); W,w,b=initial(142001)
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
    for j in range(3):
        d=np.zeros(3); d[j]=eps
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
    calpairs=0; sharingbits=0; corr=np.eye(3,dtype=complex); alpha=1.
    calinfo={}; calibration_error=None; oracle=False
    if method=='scalar_est':
        calpairs=C['scalar_calibration_pairs']; ahat,corr,alpha,calinfo=fit_temporal(world,calpairs,True)
        calibration_error=float(np.linalg.norm(ahat-true_a)/np.linalg.norm(true_a))
    if method in ['matrix_est','shift_zero']:
        calpairs=0 if method=='shift_zero' else C['matrix_calibration_pairs']; ahat,corr,alpha,calinfo=fit_temporal(world,calpairs)
        sharingbits=0 if method=='shift_zero' else 18*C['control_bits_per_real']
        calibration_error=float(np.linalg.norm(ahat-true_a)/np.linalg.norm(true_a))
    if method=='matrix_oracle':
        # Sole algorithm arm that may consume true temporal A. Never an instant F oracle.
        corr=np.linalg.pinv(true_a.conj().T,rcond=C['exact_pseudoinverse_relative_cutoff']); oracle=True
        calibration_error=0.
    physical=method in ['raw','scalar_est','matrix_est','matrix_oracle','shift_zero']
    csi=method.startswith('csi')
    scoremethod=method.startswith('score_')
    if physical:
        n=C['fixed_training_examples']
        if oracle: n=(C['rf_observation_quota']-5*C['matrix_calibration_pairs'])//2
    elif csi: n=C['rf_observation_quota']//3 if method=='csi_rf_budget' else 4000
    else: n=C['rf_observation_quota']
    gainsbits=16*n if physical and method!='shift_zero' else 0
    lossbits=16*n if scoremethod else 0
    gradientbits=64*n if csi else 0
    costs={'forward_data_observations':n,'reverse_seed_observations':n if physical else 0,
           'current_csi_pilot_observations':2*n if csi else 0,
           'temporal_calibration_pilot_observations':5*calpairs,
           'calibration_channel_sharing_bits':192*calpairs,'correction_matrix_sharing_bits':sharingbits,
           'seed_norm_bits':gainsbits,'loss_feedback_bits':lossbits,'digital_gradient_bits':gradientbits,
           'training_labels':n,'temporal_pairs':calpairs,'oracle_statistics':oracle}
    costs['rf_observations']=sum(costs[k] for k in ['forward_data_observations','reverse_seed_observations','current_csi_pilot_observations','temporal_calibration_pilot_observations'])
    costs['control_bits']=192*calpairs+sharingbits+gainsbits+lossbits+gradientbits
    for rate in C['control_bits_per_rf_slot_scenarios']:
        costs[f'serial_slots_if_{rate}_bits_per_rf_duration']=costs['rf_observations']+costs['control_bits']/rate
    trace=[]; clips=0; gainclips=0; scoreclip=0; mean_energy=0.; reverse_energy=0.
    rx_fit=None
    if method=='rx_only':
        ids=np.arange(n); x=mv(W,world.inputs[:n]); y=world.forward(x,ids)
        features=np.tanh(abs(y)**2); labels=world.labels[:n]
        def fun(v):
            zz=v[3]+np.einsum('bi,i->b',features,v[:3])
            er=1/(1+np.exp(-zz))-labels
            val=np.mean(np.logaddexp(0,zz)-labels*zz)+C['receiver_l2']*np.sum(v[:3]**2)/2
            g=np.r_[np.einsum('b,bi->i',er,features)/n+C['receiver_l2']*v[:3],er.mean()]
            return val,g
        opt=minimize(fun,np.r_[w,b],jac=True,method='L-BFGS-B',bounds=[(-4,4)]*4,
                     options={'maxiter':300,'gtol':1e-9,'ftol':1e-12})
        w=opt.x[:3]; b=float(opt.x[3]); mean_energy=float(np.mean(np.sum(abs(x)**2,axis=1)))
        rx_fit={'success':bool(opt.success),'message':str(opt.message),'iterations':int(opt.nit),'jac_norm':float(np.linalg.norm(opt.jac))}
        assert opt.success
    else:
        Opt=SGD if C['optimizer']=='SGD' else Adam
        aw=Opt(W.shape,complex); ar=Opt((4,),float)
        baseline=np.full(8,np.log(2.))
        prng=np.random.default_rng(151000+seed)
        # The receiver-coordinate bound gives a deterministic public norm bound.
        seed_bound=2*np.sqrt(3)*C['receiver_coordinate_bound']*np.linalg.norm(corr,2)
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
                seeds=mv(corr,s)
                norms=np.linalg.norm(seeds,axis=1)
                assert norms.max()<=seed_bound+1e-10
                if method=='shift_zero':
                    tx=seeds/seed_bound
                    gain=np.full(len(ids),seed_bound)
                else:
                    tx=seeds/np.maximum(norms,1e-15)[:,None]
                    gain,gc=qreal(norms,seed_bound,False); gainclips+=gc
                assert np.max(np.sum(abs(tx)**2,axis=1))<=1+1e-12
                reverse_energy+=float(np.sum(abs(tx)**2))
                gx=np.zeros((len(ids),2),dtype=complex) if alpha==0 else world.backward(tx,ids)*gain[:,None]/np.conj(alpha)
            elif csi:
                fhat=world.pilot_forward(ids)
                gx,cc=qcomplex(pb(fhat,s),C['digital_gradient_real_quant_bound']); clips+=cc
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
            oldW=W.copy()
            W=aw.step(W,gW,C['encoder_adam_lr'])
            unprojected_step=float(np.linalg.norm(W-oldW))
            projected=bool(np.linalg.norm(W)>C['encoder_frobenius_bound'])
            W=W*min(1.,C['encoder_frobenius_bound']/max(np.linalg.norm(W),1e-15))
            params=ar.step(np.r_[w,b],np.r_[gw,gb],C['receiver_adam_lr'])
            params=np.clip(params,-C['receiver_coordinate_bound'],C['receiver_coordinate_bound'])
            w=params[:3]; b=float(params[3])
            assert np.isfinite(W).all() and np.isfinite(w).all() and np.isfinite(b)
            if start==0 or (start//C['batch_size']+1)%20==0 or ids[-1]==n-1:
                trace.append({'seen_samples':int(ids[-1]+1),'training_ce':float(loss.mean()),'encoder_norm':float(np.linalg.norm(W)),'unprojected_step':unprojected_step,'encoder_projected':projected})
        mean_energy/=n; reverse_energy=reverse_energy/n if physical else 0.
    out={'seed':seed,'case':case,'method':method,'initial_service':initial_service,
         'service':world.service(W,w,b),'costs':costs,'calibration_relative_error':calibration_error,
         'calibration':calinfo,'gradient_or_loss_quant_clips':clips,'gain_quant_clips':gainclips,
         'score_train_clip_fraction':scoreclip/n,'mean_train_forward_energy':mean_energy,
         'mean_reverse_energy':reverse_energy,'receiver_fit':rx_fit,'trace':trace,
         'initial_W_real':initialW.real.tolist(),'initial_W_imag':initialW.imag.tolist(),
         'final_W_real':W.real.tolist(),'final_W_imag':W.imag.tolist(),'final_receiver_w':w.tolist(),'final_receiver_b':b}
    if scoremethod: out['smoothed_test_service']=world.service(W,w,b,float(method.split('_')[1]))
    return out

def analyze(rows):
    rng=np.random.default_rng(C['bootstrap_seed']); n=len(C['environment_seeds'])
    indices=rng.integers(0,n,size=(C['bootstrap_replicates'],n))
    summaries={}; comparisons={}
    for case in C['cases']:
        sub=[r for r in rows if r['case']==case]
        summaries[case]={}
        by={m:sorted([r for r in sub if r['method']==m],key=lambda x:x['seed']) for m in C['methods']}
        for method,rr in by.items():
            summaries[case][method]={'mean_ce':float(np.mean([r['service']['ce'] for r in rr])),
               'mean_accuracy':float(np.mean([r['service']['accuracy'] for r in rr])),
               'mean_calibration_relative_error':None if rr[0]['calibration_relative_error'] is None else float(np.mean([r['calibration_relative_error'] for r in rr])),
               'rf_observations':rr[0]['costs']['rf_observations'],'control_bits':rr[0]['costs']['control_bits'],
               'mean_score_clipping':float(np.mean([r['score_train_clip_fraction'] for r in rr]))}
        comparisons[case]={}
        mat=np.array([r['service']['ce'] for r in by['matrix_est']])
        for other in C['methods']:
            if other=='matrix_est': continue
            d=np.array([r['service']['ce'] for r in by[other]])-mat
            ci=np.percentile(d[indices].mean(axis=1),[2.5,97.5])
            comparisons[case][f'matrix_gain_over_{other}']={'mean_ce_gain':float(d.mean()),'ci95':ci.tolist(),'wins':int(np.count_nonzero(d>0)),'all_paired_gains':d.tolist()}
    primary=comparisons[C['primary_case']]['matrix_gain_over_raw']
    passed=primary['mean_ce_gain']>=C['primary_ce_gain_threshold'] and primary['ci95'][0]>0 and primary['wins']>=C['primary_min_winning_environments']
    return {'summary':summaries,'comparisons':comparisons,'primary_repair_gate_passed':bool(passed),'primary':primary}

def main():
    start=time.time(); checks=verify(); rows=[]
    frozen={rel:hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() for rel in ['configs/config.json','configs/SPEC.md','src/train_probe.py']}
    for seed in C['environment_seeds']:
        for case in C['cases']:
            a=make_a(seed,case); world=World(seed,a)
            for method in C['methods']:
                rows.append(train(world,seed,case,method,a))
                assert time.time()-start<C['wall_budget_seconds'], 'Frozen time budget exceeded'
        print(json.dumps({'completed_seed':seed,'rows':len(rows),'elapsed':time.time()-start}),flush=True)
    report=analyze(rows)
    report['metadata']={'host':platform.node(),'python':platform.python_version(),'numpy':np.__version__,
        'task_name':'pbp-stat-learn-v1-20261014','elapsed_seconds':time.time()-start,'checks':checks,'hashes':frozen,
        'rows':len(rows),'config':C}
    (ROOT/'results/all_runs.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'results/summary.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'results/summary.csv').open('w') as fp:
        fields=['case','method','mean_ce','mean_accuracy','mean_calibration_relative_error','rf_observations','control_bits','mean_score_clipping']
        wr=csv.DictWriter(fp,fieldnames=fields); wr.writeheader()
        for case,arms in report['summary'].items():
            for method,r in arms.items(): wr.writerow({'case':case,'method':method,**r})
    print(json.dumps({'status':'complete','rows':len(rows),'elapsed_seconds':time.time()-start,'primary_repair_gate_passed':report['primary_repair_gate_passed'],'primary':report['primary'],'checks':checks}))

if __name__=='__main__': main()
