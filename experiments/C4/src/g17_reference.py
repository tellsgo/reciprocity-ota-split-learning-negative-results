"""Frozen, dimension-faithful two single-antenna-user backprop experiment."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, pathlib, json, time, hashlib, traceback, resource, signal, csv
import numpy as np
import learning_core as K

R=pathlib.Path(__file__).resolve().parents[1]
SEEDS=list(range(261700000,261700008))
METHODS=['naive_polar','ZF_polar','MMSE_polar','single_user','CSI']
N=3840; NT=8192; PILOTS=64
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False))
def rng(seed,stream): return np.random.default_rng(np.random.SeedSequence([seed,stream]))
def norm(x): return float(np.linalg.norm(x))
def steering(theta): return np.exp(-2j*np.pi*.5*np.arange(4)*np.cos(np.deg2rad(theta)))
def geometry():
    a=np.array([steering(30),steering(35)])
    phase=np.exp(2j*np.pi*.383*np.cos(np.deg2rad([30,35])))
    return a,phase
def initial(seed,stream):
    W=K.cn(rng(seed,stream),(4,3));return W/norm(W)*.4,np.array([.8]),-.4
def channels(gen,n):
    a,phase=geometry();z=K.cn(gen,(n,2));h0=z[:,:,None]*a[None,:,:]
    return h0,h0*phase[None,:,None]
def forward(h,W,inp,noise):
    x=K.mv(W,inp)
    assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-12
    return (np.einsum('bi,bi->b',h.conj(),x)+noise)[:,None]
def make_world(seed):
    gen=rng(seed,1);inp=gen.choice([-1.,1.],(N,3));h0,h1=channels(gen,N)
    out={'inputs':inp,'labels':(inp[:,0]*inp[:,1]>0).astype(float),
         'h0':h0,'h1':h1,'nf':.1*K.cn(gen,(N,)), 'nb':.03*K.cn(gen,(N,4))}
    gen=rng(seed,2);c0,c1=channels(gen,PILOTS)
    out.update(cal0=c0+.1*K.cn(gen,c0.shape),cal1=c1+.1*K.cn(gen,c1.shape),cal_true0=c0,cal_true1=c1)
    gen=rng(seed,3);inp=gen.choice([-1.,1.],(NT,3));t0,_=channels(gen,NT)
    out.update(test_inputs=inp,test_labels=(inp[:,0]*inp[:,1]>0).astype(float),
               test_h=t0[:,0],test_n=.1*K.cn(gen,(NT,)))
    gen=rng(seed,6);inp=gen.choice([-1.,1.],(N,3));noise=.1*K.cn(gen,(N,))
    W,w,b=initial(seed,5)
    y=forward(h0[:,1],W,inp,noise)
    loss,s,_,_,_=K.task(y,(inp[:,0]*inp[:,1]>0).astype(float),w,b)
    out.update(user2_inputs=inp,user2_noise=noise,user2_W=W,user2_seed=s[:,0],user2_loss=loss,
               user2_tx=s[:,0]/np.maximum(abs(s[:,0]),1e-15))
    return out
def estimate(world,method):
    if method=='MMSE_polar': return estimate_mmse(world)
    I=np.eye(4,dtype=complex)
    r2=np.einsum('bi,bj->ij',world['cal1'][:,1],world['cal1'][:,1].conj())/PILOTS
    eig,vec=np.linalg.eigh(r2);hhat=vec[:,-1]
    P=I-np.outer(hhat,hhat.conj())/np.vdot(hhat,hhat)
    if method=='single_user': f0=world['cal0'][:,0].copy();f1=world['cal1'][:,0].copy();noise=.01;proj=I
    else:
        f0=world['cal0'][:,0]+np.sqrt(10)*world['cal0'][:,1]
        f1=world['cal1'][:,0]+np.sqrt(10)*world['cal1'][:,1]
        noise=.11;proj=P if method=='ZF_polar' else I
        f0=K.mv(proj,f0);f1=K.mv(proj,f1)
    f0,fc=K.qcomplex(f0,4.)
    raw=np.einsum('bi,bj->ij',f0,f0.conj())/PILOTS-noise*proj
    cross=np.einsum('bi,bj->ij',f1,f0.conj())/PILOTS
    reg=(raw+raw.conj().T)/2+.02*I
    ev,vr=np.linalg.eigh(reg);evreg=np.maximum(ev,.02);reg=(vr*evreg)@vr.conj().T
    A=cross@np.linalg.inv(reg);u,sv,vh=np.linalg.svd(A);Q=vh.conj().T@u.conj().T
    Qraw=Q.copy();Q,qc=K.qcomplex(Q,32.);u,_,vh=np.linalg.svd(Q);Q=u@vh
    T=proj@Q@proj
    a,phase=geometry();target=T@(phase[0]*a[0]);inter=T@(phase[1]*a[1])
    diag={'pilot_clips':fc,'correction_clips':qc,'A_singular_values':sv.tolist(),
          'covariance_floor_count':int((ev<.02).sum()),'Creg_condition':float(np.linalg.cond(reg)),
          'P_hermitian_error':norm(P-P.conj().T),'P_idempotence_error':norm(P@P-P),
          'P_rank':int(np.linalg.matrix_rank(P,tol=1e-10)), 'estimated_null_residual':norm(P@hhat),
          'true_interference_energy_fraction':norm(P@a[1])**2/norm(a[1])**2,
          'target_projected_energy_fraction':norm(P@a[0])**2/norm(a[0])**2,
          'Q_unitarity_error':norm(Q.conj().T@Q-I),'Q_quantization_error':norm(Q-Qraw),
          'T_spectral_norm':float(np.linalg.norm(T,2)),
          'target_effective_alignment_real':float(np.vdot(a[0],target).real/4),
          'target_effective_relative_error':norm(target-a[0])/2,
          'output_interference_gain':norm(inter)/2}
    mats={'P':P,'hhat2':hhat,'Q':Q,'T':T,'A':A,'Creg':reg,'C10':cross,'F0_quantized':f0,'F1':f1}
    return T,diag,mats
def spatial_estimate(samples):
    cov=samples.T@samples.conj()/PILOTS
    ev,v=np.linalg.eigh(cov);u=v[:,-1]
    u=u*np.exp(-1j*np.angle(u[0]))
    return u*np.sqrt(max(float(ev[-1])-.01,.02))
def estimate_mmse(world):
    # Fixed pilot-only spatial signatures; instantaneous scalar fading is NOT supplied.
    h10=spatial_estimate(world['cal0'][:,0])
    h11=spatial_estimate(world['cal1'][:,0])
    h21=np.sqrt(10)*spatial_estimate(world['cal1'][:,1])
    reg=np.outer(h21,h21.conj())+.1*np.eye(4)
    Wmmse=np.linalg.solve(reg,h11)
    f0=K.mv(Wmmse.conj()[None,:],world['cal0'][:,0]+np.sqrt(10)*world['cal0'][:,1])
    f1=K.mv(Wmmse.conj()[None,:],world['cal1'][:,0]+np.sqrt(10)*world['cal1'][:,1])
    f0,fc=K.qcomplex(f0,4.)
    raw=(f0.T@f0.conj()/PILOTS).item()-.11*np.vdot(Wmmse,Wmmse).real
    creg=max(raw.real+.02,.02)
    cross=(f1.T@f0.conj()/PILOTS).item()
    A=np.array([[cross/creg]])
    u,sv,vh=np.linalg.svd(A);Q=vh.conj().T@u.conj().T
    Qraw=Q.copy();Q,qc=K.qcomplex(Q,32.);u,_,vh=np.linalg.svd(Q);Q=u@vh
    # Scalar spatial combining requires lifting back into the old target channel.
    T=np.outer(h10,Wmmse.conj())*Q.item()
    a,ph=geometry();target=T@(ph[0]*a[0]);inter=T@(ph[1]*a[1])
    diag={'pilot_clips':fc,'correction_clips':qc,'A_singular_values':sv.tolist(),
          'covariance_floor_count':int(raw.real+.02<.02),'Creg_condition':1.,
          'Q_unitarity_error':norm(Q.conj().T@Q-np.eye(1)), 'Q_quantization_error':norm(Q-Qraw),
          'W_norm':norm(Wmmse),'regularized_system_condition':float(np.linalg.cond(reg)),
          'linear_solve_relative_residual':norm(reg@Wmmse-h11)/norm(h11),
          'T_spectral_norm':float(np.linalg.norm(T,2)),
          'noise_energy_ratio':float(np.trace(T@T.conj().T).real/4),
          'target_effective_alignment_real':float(np.vdot(a[0],target).real/4),
          'target_effective_relative_error':norm(target-a[0])/2,'output_interference_gain':norm(inter)/2,
          'gamma':.1,'reverse_noise_variance':.03**2}
    mats={'W_MMSE':Wmmse,'Hhat10':h10,'Hhat11':h11,'Hhat21':h21,'MMSE_system':reg,
          'Q':Q,'T':T,'A':A,'Creg':np.array([[creg]]),'C10':np.array([[cross]]),'F0_quantized':f0,'F1':f1}
    return T,diag,mats

def service(world,W,w,b):
    y=forward(world['test_h'],W,world['test_inputs'],world['test_n'])
    loss,_,_,_,prob=K.task(y,world['test_labels'],w,b)
    return {'ce':float(loss.mean()),'accuracy':float(np.mean((prob>=.5)==world['test_labels']))}
def train(world,seed,method):
    W,w,b=initial(seed,4);initial_service=service(world,W,w,b)
    n=2666 if method=='CSI' else N;T=None;cal={};mats={}
    if method!='CSI':T,cal,mats=estimate(world,method)
    aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float);tr=[];gainclips=0
    errors=[];cosines=[];maxpow=0.;totalraw=0.;totaleffective=0.
    for start in range(0,n,32):
        ids=np.arange(start,min(start+32,n));inp=world['inputs'][ids];lab=world['labels'][ids]
        y=forward(world['h0'][ids,0],W,inp,world['nf'][ids])
        loss,s,gw,gb,_=K.task(y,lab,w,b);s=s[:,0]
        truegx=world['h0'][ids,0]*s[:,None]
        if method=='CSI':gx=truegx
        else:
            mag=abs(s);tx=s/np.maximum(mag,1e-15)
            gain,gc=K.qreal(mag,8.,False);gainclips+=gc
            maxpow=max(maxpow,float(np.max(abs(tx)**2)))
            assert maxpow<=1+1e-12 and np.max(mag)<=8+1e-12
            inter=np.zeros((len(ids),4),complex) if method=='single_user' else np.sqrt(10)*world['h1'][ids,1]*world['user2_tx'][ids,None]
            received=world['h1'][ids,0]*tx[:,None]+inter+world['nb'][ids]
            gx=K.mv(T,received)*gain[:,None]
            totalraw+=float(np.sum(abs(inter)**2));totaleffective+=float(np.sum(abs(K.mv(T,inter))**2))
        gW=K.pgrad(gx,inp);truth=K.pgrad(truegx,inp)
        errors.append(norm(gW-truth));den=norm(gW)*norm(truth)
        cosines.append(float(np.vdot(truth,gW).real/max(den,1e-30)))
        oldW=W.copy();W=aw.step(W,gW,.025);unproj=norm(W-oldW)
        W*=min(1.,(1/np.sqrt(3))/max(norm(W),1e-15))
        par=ar.step(np.r_[w,b],np.r_[gw,gb],.06);par=np.clip(par,-4,4);w=par[:1];b=float(par[1])
        assert np.isfinite(W).all() and np.isfinite(par).all()
        tr.append({'step':len(tr)+1,'samples':int(ids[-1]+1),'training_ce':float(loss.mean()),
                   'encoder_norm':norm(W),'gradient_norm':norm(gW),'true_gradient_norm':norm(truth),
                   'gradient_error':errors[-1],'gradient_cosine':cosines[-1],
                   'unprojected_step':unproj,'receiver_w':float(w[0]),'receiver_b':b})
    cal_slots=128 if method=='single_user' else 256
    if method=='CSI':cost={'target_reserved_rf':7998,'actual_system_rf':7998,'reserved_system_rf':7998,'seed_norm_bits':0,'ideal_csi_privilege':True}
    else:cost={'target_reserved_rf':8000,'actual_system_rf':2*N+cal_slots+(0 if method=='single_user' else N),
              'reserved_system_rf':8000+(0 if method=='single_user' else N),'seed_norm_bits':16*N,'ideal_csi_privilege':False,
              'actual_calibration_slots':cal_slots,'reserved_calibration_slots':320}
    cost.update(training_examples=n,updates=len(tr),test_forward_evaluations=NT)
    row={'seed':seed,'method':method,'initial_service':initial_service,'service':service(world,W,w,b),
         'calibration':cal,'costs':cost,'trace':tr,'gain_quant_clips':gainclips,'max_reverse_tx_power':maxpow,
         'mean_gradient_error':float(np.mean(errors)),'mean_gradient_cosine':float(np.mean(cosines)),
         'raw_interference_energy':totalraw,'corrected_interference_energy':totaleffective}
    np.savez_compressed(R/'results'/f'model_{seed}_{method}.npz',W=W,w=w,b=b,**mats)
    return row

def preflight():
    gen=rng(SEEDS[0],7);inp=gen.choice([-1.,1.],(23,3));lab=(inp[:,0]*inp[:,1]>0).astype(float)
    h=K.cn(gen,(23,4));noise=.1*K.cn(gen,(23,));W,w,b=initial(SEEDS[0],4)
    def obj(W,w,b):return K.task(forward(h,W,inp,noise),lab,w,b)[0].mean()+.001*np.sum(w*w)/2
    _,s,gw,gb,_=K.task(forward(h,W,inp,noise),lab,w,b);gW=K.pgrad(h*s,inp)
    err=[];eps=1e-6
    for ix in np.ndindex(W.shape):
        for part in [1.,1j]:
            d=np.zeros_like(W);d[ix]=part
            err.append(abs((obj(W+eps*d,w,b)-obj(W-eps*d,w,b))/(2*eps)-np.vdot(gW,d).real))
    err.append(abs((obj(W,w+eps,b)-obj(W,w-eps,b))/(2*eps)-gw[0]))
    err.append(abs((obj(W,w,b+eps)-obj(W,w,b-eps))/(2*eps)-gb));assert max(err)<1e-7
    a,ph=geometry();P=np.eye(4)-np.outer(a[1],a[1].conj())/4
    overlap=abs(np.vdot(a[1],a[0])/4)**2
    assert norm(P@a[1])<1e-12 and abs(norm(P@a[0])**2/4-(1-overlap))<1e-12
    ideal=np.conj(ph[0])*np.eye(4);single=norm(ideal@(ph[0]*a[0])-a[0]);assert single<1e-12
    # Analytic rank-one full-unitary polar extension has the correct adjoint direction.
    A=ph[0]*np.outer(a[0],a[0].conj())/4;u,_,vh=np.linalg.svd(A);Q=vh.conj().T@u.conj().T
    errpolar=norm(Q@(ph[0]*a[0])-a[0]);assert errpolar<1e-12
    world=make_world(SEEDS[0]);other=make_world(SEEDS[0]);assert all(np.array_equal(world[k],other[k]) for k in world)
    ds={}
    for method in METHODS[:-1]:
        _,d,_=estimate(world,method);ds[method]=d
        if method=='MMSE_polar':
            assert d['linear_solve_relative_residual']<1e-12 and d['Q_unitarity_error']<1e-12
            continue
        assert d['P_rank']==3 and d['P_idempotence_error']<1e-12 and d['Q_unitarity_error']<1e-12
        assert d['estimated_null_residual']<1e-12 and d['T_spectral_norm']<=1+1e-12
    assert np.max(abs(world['user2_tx'])**2)<=1+1e-12
    return {'finite_difference_max_abs_error':max(err),'ideal_single_return_error':single,
            'rank_one_polar_direction_error':errpolar,'spatial_overlap_squared':float(overlap),'spatial_overlap_modulus':float(np.sqrt(overlap)),
            'true_projection_singular_values':np.linalg.svd(P,compute_uv=False).tolist(),
            'true_projection_noise_energy_ratio':float(np.trace(P@P.conj().T).real/4),
            'target_gain_inverse_noise_power_factor_if_added':1/(1-float(overlap)),
            'ideal_target_retained_energy_fraction':1-float(overlap),'paired_world_exactly_reproducible':True,
            'diagnostics_seed0':ds}
def stats(vals):
    a=np.asarray(vals);assert a.shape==(8,)
    idx=np.random.default_rng(261799999).integers(0,8,(20000,8))
    return {'mean':float(a.mean()),'ci95':np.quantile(a[idx].mean(1),[.025,.975]).tolist(),
            'per_seed':a.tolist(),'sd':float(a.std(ddof=1))}
def analyze(rows):
    lookup={(r['seed'],r['method']):r for r in rows}
    summaries={m:{k:stats([lookup[s,m]['service'][k] for s in SEEDS]) for k in ['accuracy','ce']} for m in METHODS}
    ac=stats([lookup[s,'MMSE_polar']['service']['accuracy']-lookup[s,'ZF_polar']['service']['accuracy'] for s in SEEDS])
    ce=stats([lookup[s,'ZF_polar']['service']['ce']-lookup[s,'MMSE_polar']['service']['ce'] for s in SEEDS])
    loss=stats([lookup[s,'single_user']['service']['accuracy']-lookup[s,'ZF_polar']['service']['accuracy'] for s in SEEDS])
    na=summaries['naive_polar']['accuracy']['mean'];za=summaries['ZF_polar']['accuracy']['mean'];ma=summaries['MMSE_polar']['accuracy']['mean']
    claims={}
    for name,support,falsifier,value in [
        ('G17.1',za<=.65 and loss['mean']>=.25,za>=.80,{'zf_accuracy':za,'single_minus_zf_accuracy':loss['mean']}),
        ('G17.2',ma>=.88 and ac['mean']>=.20 and ce['mean']>=.30,ma<.80 or ac['mean']<.10,{'mmse_accuracy':ma,'accuracy_gain':ac['mean'],'ce_gain':ce['mean']}),
        ('G17.3',na<.70,na>=.80,{'naive_accuracy':na})]:
        claims[name]={'verdict':'archived' if falsifier else ('archived' if support else 'archived'),
                      'support_threshold_met':bool(support),'falsification_triggered':bool(falsifier),**value}
        if falsifier:dump(R/'results'/f'STOP_{name}.json',{'claim':name,'time':time.time(),'reason':claims[name],'action':'closed; no additional configurations or seeds'})
    dump(R/'results/summary.json',summaries)
    dump(R/'results/contrasts.json',{'accuracy_MMSE_minus_ZF':ac,'ce_ZF_minus_MMSE':ce,'accuracy_single_minus_ZF':loss})
    dump(R/'results/claims.json',claims)
    raw=[{'seed':r['seed'],'method':r['method'],**r['service'],
          'mean_gradient_cosine':r['mean_gradient_cosine'],'pilot_clips':r['calibration'].get('pilot_clips',0),
          'system_rf':r['costs']['actual_system_rf']} for r in rows]
    with (R/'results/raw_metrics.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(raw[0]));wr.writeheader();wr.writerows(raw)
    return claims
def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    for f,h in json.loads((R/'freeze_manifest.json').read_text())['hashes'].items():
        assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
    dump(R/'results/preflight.json',preflight());print('PREFLIGHT PASS',flush=True)
    rows=[]
    for seed in SEEDS:
        world=make_world(seed);np.savez_compressed(R/'results'/f'world_{seed}.npz',**world)
        for method in METHODS:
            row=train(world,seed,method);rows.append(row)
            with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            dump(R/'results/partial_runs.json',rows)
            print(seed,method,row['service'],flush=True)
    dump(R/'results/all_runs.json',rows);claims=analyze(rows)
    dump(R/'results/runtime.json',{'pid':os.getpid(),'wall_seconds':time.time()-start,'completed_rows':len(rows),'finished_epoch':time.time()})
    print('DONE',claims,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'results/failure.txt').write_text(traceback.format_exc());raise
