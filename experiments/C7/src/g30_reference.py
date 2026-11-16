"""G30: one frozen topology, no adaptive hyperparameters or extra seeds."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys,pathlib,json,time,hashlib,traceback,resource,signal,csv
sys.dont_write_bytecode=True
import numpy as np
import learning_core as K
R=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((R/'configs/effective.json').read_text());SEEDS=C['seeds'];N=3840;NT=8192;P=64
SINGLES=[f'single_AP_{i+1}' for i in range(4)]
DIST='distributed_consensus_polar';CENT='centralized_polar';CSI='CSI_centralized';SINGLE='single_AP_isolated'
METHODS=[SINGLE,DIST,CENT,CSI]
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))
def norm(x):return float(np.linalg.norm(x))
def geometry():
    t=np.deg2rad(C['angles']);a=np.exp(-2j*np.pi*.5*np.cos(t)[:,None]*np.arange(2))
    ph=np.exp(2j*np.pi*.383*np.cos(t));return a,ph

def channels(gen,n):
    a,ph=geometry();h=K.cn(gen,(n,4))[:,:,None]*a[None,:,:]
    return h.reshape(n,8),(h*ph[None,:,None]).reshape(n,8)

def initial(seed,ap=None):
    W=K.cn(rng(seed,4),(8,3))
    if ap is not None:W=W[2*ap:2*ap+2].copy()
    return W/norm(W)*.4,np.array([.8]),-.4

def make_world(seed):
    gen=rng(seed,1);inp=gen.choice([-1.,1.],(N,3));h0,h1=channels(gen,N)
    out=dict(inputs=inp,labels=(inp[:,0]*inp[:,1]>0).astype(float),h0=h0,h1=h1,nf=.1*K.cn(gen,(N,)),nb=.03*K.cn(gen,(N,8)))
    gen=rng(seed,2);c0,c1=channels(gen,P)
    out.update(cal0=c0+.1*K.cn(gen,c0.shape),cal1=c1+.1*K.cn(gen,c1.shape),cal_true0=c0,cal_true1=c1)
    gen=rng(seed,3);inp=gen.choice([-1.,1.],(NT,3));h,_=channels(gen,NT)
    out.update(test_inputs=inp,test_labels=(inp[:,0]*inp[:,1]>0).astype(float),test_h=h,test_n=.1*K.cn(gen,(NT,)))
    return out

def forward(h,W,inp,noise):
    x=K.mv(W,inp);assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-12
    return (np.einsum('bi,bi->b',h.conj(),x)+noise)[:,None]

def polar(f0,f1):
    d=f0.shape[1];I=np.eye(d);f0,fc=K.qcomplex(f0,4.)
    c00=np.einsum("bi,bj->ij",f0,f0.conj())/P-.01*I;c00=(c00+c00.conj().T)/2
    c10=np.einsum("bi,bj->ij",f1,f0.conj())/P;ev,v=np.linalg.eigh(c00+.02*I);reg=K.mm(v*np.maximum(ev,.02),v.conj().T)
    A=np.linalg.solve(reg.T,c10.T).T;u,sv,vh=np.linalg.svd(A);M=K.mm(vh.conj().T,u.conj().T)
    M0=M.copy();M,mc=K.qcomplex(M,32.);u,_,vh=np.linalg.svd(M);M=K.mm(u,vh)
    diag=dict(pilot_clips=fc,correction_clips=mc,unitarity_error=norm(K.mm(M.conj().T,M)-I),polar_quantization_error=norm(M-M0),covariance_floor_count=int(sum(ev<.02)),A_singular_values=sv.tolist(),Creg_condition=float(np.linalg.cond(reg)),signal_trace=float(np.trace(c00).real))
    return M,diag,dict(M=M,A=A,C00=c00,C10=c10,Creg=reg,F0_quantized=f0,F1=f1)

def estimate(world,method):
    if method.startswith('single_AP_') and method!=SINGLE:
        ap=int(method[-1])-1;sl=slice(2*ap,2*ap+2)
        return polar(world['cal0'][:,sl],world['cal1'][:,sl])
    if method==CENT:return polar(world['cal0'],world['cal1'])
    assert method==DIST
    ds=[];mats={};ms=[]
    for ap in range(4):
        sl=slice(2*ap,2*ap+2);m,d,ma=polar(world['cal0'][:,sl],world['cal1'][:,sl]);ms.append(m);ds.append(d)
        mats.update({f'ap{ap+1}_{k}':v for k,v in ma.items()})
    power=np.array([d['signal_trace'] for d in ds]);assert min(power)>0
    weights=power/power.sum();T=np.zeros((8,8),complex)
    for ap in range(4):T[2*ap:2*ap+2,2*ap:2*ap+2]=weights[ap]*ms[ap]
    mats.update(T=T,weights=weights)
    return T,dict(weights=weights.tolist(),locals=ds,pilot_clips=sum(d['pilot_clips'] for d in ds),correction_clips=sum(d['correction_clips'] for d in ds),unitarity_error=max(d['unitarity_error'] for d in ds)),mats

def service(world,W,w,b,ap=None):
    h=world['test_h'] if ap is None else world['test_h'][:,2*ap:2*ap+2]
    y=forward(h,W,world['test_inputs'],world['test_n']);loss,_,_,_,prob=K.task(y,world['test_labels'],w,b)
    x=K.mv(W,world['test_inputs']);energy=np.sum(abs(x)**2,axis=1)
    return dict(ce=float(loss.mean()),accuracy=float(np.mean((prob>=.5)==world['test_labels'])),max_tx_energy=float(energy.max()),mean_tx_energy=float(energy.mean()))

def train(world,seed,method):
    ap=int(method[-1])-1 if method in SINGLES else None;sl=slice(None) if ap is None else slice(2*ap,2*ap+2)
    W,w,b=initial(seed,ap);W0=W.copy();initial_service=service(world,W,w,b,ap)
    n=2666 if method==CSI else N;cal={};mats={};T=None
    if method!=CSI:T,cal,mats=estimate(world,method)
    aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float);trace=[];clips=0;maxpow=0.
    for start in range(0,n,32):
        ids=np.arange(start,min(start+32,n));inp=world['inputs'][ids];lab=world['labels'][ids]
        h0=world['h0'][ids,sl];h1=world['h1'][ids,sl]
        y=forward(h0,W,inp,world['nf'][ids]);loss,s,gw,gb,_=K.task(y,lab,w,b);s=s[:,0];truthgx=h0*s[:,None]
        if method==CSI:gx=truthgx
        else:
            mag=abs(s);tx=s/np.maximum(mag,1e-15);gain,gc=K.qreal(mag,8.,False);clips+=gc
            maxpow=max(maxpow,float(np.max(abs(tx)**2)));assert maxpow<=1+1e-12
            Y=h1*tx[:,None]+world['nb'][ids,sl];gx=K.mv(T,Y)*gain[:,None]
        gW=K.pgrad(gx,inp);truth=K.pgrad(truthgx,inp)
        cosine=float(np.vdot(truth,gW).real/max(norm(truth)*norm(gW),1e-30));err=norm(gW-truth)
        old=W.copy();W=aw.step(W,gW,.025);rawstep=norm(W-old);W*=min(1.,(1/np.sqrt(3))/max(norm(W),1e-15))
        par=np.clip(ar.step(np.r_[w,b],np.r_[gw,gb],.06),-4,4);w=par[:1];b=float(par[1]);assert np.isfinite(W).all() and np.isfinite(par).all()
        trace.append(dict(step=len(trace)+1,samples=int(ids[-1]+1),training_ce=float(loss.mean()),encoder_norm=norm(W),gradient_norm=norm(gW),true_gradient_norm=norm(truth),gradient_cosine=cosine,gradient_error=err,unprojected_step=rawstep,receiver_w=float(w[0]),receiver_b=b))
    costs=dict(training_examples=n,updates=len(trace),test_forward_evaluations=NT,actual_rf_slots=7998 if method==CSI else 7808,reserved_rf_slots=7998 if method==CSI else 8000,antenna_observations_per_return=2 if ap is not None else 8,seed_norm_bits=0 if method==CSI else 16*n,ideal_csi_privilege=method==CSI)
    if method==DIST:costs.update(reverse_payload_bits_per_example=256,static_strength_bits=64)
    if method==CENT:costs.update(reverse_payload_bits_per_example=256,calibration_payload_bits=2*P*8*32)
    row=dict(seed=seed,method=method,ap=ap,initial_service=initial_service,service=service(world,W,w,b,ap),calibration=cal,costs=costs,trace=trace,gain_quant_clips=clips,max_reverse_tx_power=maxpow,mean_gradient_cosine=float(np.mean([t['gradient_cosine'] for t in trace])))
    np.savez_compressed(R/'results'/f'model_{seed}_{method}.npz',W=W,w=w,b=b,W0=W0,w0=np.array([.8]),b0=-.4,**mats)
    return row

def stats(v):
    a=np.asarray(v);assert a.shape==(8,) and np.isfinite(a).all()
    ix=np.random.default_rng(C['bootstrap_seed']).integers(0,8,(20000,8))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[ix].mean(1),[.025,.975]).tolist(),min=float(a.min()),max=float(a.max()),per_seed=a.tolist())

def chosen(rows):
    out=[]
    for seed in SEEDS:
        candidates=[r for r in rows if r['seed']==seed and r['method'] in SINGLES]
        assert len(candidates)==4
        best=sorted(candidates,key=lambda r:(-r['service']['accuracy'],r['service']['ce'],r['ap']))[0]
        out.append(dict(seed=seed,method=SINGLE,selected_method=best['method'],selected_ap=best['ap'],service=best['service'],selection='test-oracle; derived not trained',oracle_selection_training_rf_slots=4*7808,oracle_selection_test_examples=4*NT))
    return out

def adjudicate(rows,line):
    lookup={(r['seed'],r['method']):r for r in rows+chosen(rows)}
    def values(m,k):return np.array([lookup[s,m]['service'][k] for s in SEEDS])
    sa=stats(values(SINGLE,'accuracy'))
    if line=='G30.1':
        support=sa['mean']<=.75;falsifier=sa['mean']>=.82;metrics={'single_accuracy':sa}
    elif line=='G30.2':
        da=stats(values(DIST,'accuracy'));ag=stats(values(DIST,'accuracy')-values(SINGLE,'accuracy'));cg=stats(values(SINGLE,'ce')-values(DIST,'ce'))
        support=da['mean']>=.92 and ag['mean']>=.15 and cg['mean']>=.20;falsifier=da['mean']<.85 or ag['mean']<.08
        metrics=dict(distributed_accuracy=da,accuracy_gain=ag,ce_gain=cg)
    else:
        gap=stats(values(DIST,'ce')-values(CENT,'ce'));support=gap['mean']<=.05;falsifier=gap['mean']>.10
        metrics=dict(ce_gap=gap,absolute_mean_ce_gap=abs(gap['mean']),mean_absolute_per_seed_ce_gap=float(np.mean(abs(values(DIST,'ce')-values(CENT,'ce')))))
    result=dict(claim=line,verdict='archived' if falsifier else ('archived' if support else 'archived'),support_threshold_met=bool(support),falsification_triggered=bool(falsifier),**metrics)
    dump(R/'results'/f'{line}.json',result)
    event=dict(time=time.time(),completed_training_rows=len(rows),claim=line,triggered=bool(falsifier),result=result,action='closed; no further experiments for this claim' if falsifier else 'terminal batch evaluated; no expansion')
    with (R/'results/decisions.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    if falsifier:dump(R/'results'/f'STOP_{line}.json',event)
    print('ADJUDICATED',line,result,flush=True)
    return result

def preflight():
    world=make_world(SEEDS[0]);other=make_world(SEEDS[0]);assert all(np.array_equal(world[k],other[k]) for k in world)
    inp=world['inputs'][:23];lab=world['labels'][:23];h=world['h0'][:23];noise=world['nf'][:23];W,w,b=initial(SEEDS[0])
    def obj(W,w,b):return K.task(forward(h,W,inp,noise),lab,w,b)[0].mean()+.001*sum(w*w)/2
    _,s,gw,gb,_=K.task(forward(h,W,inp,noise),lab,w,b);grad=K.pgrad(h*s,inp);errors=[];ep=1e-6
    for ix in np.ndindex(W.shape):
        for part in [1.,1j]:
            d=np.zeros_like(W);d[ix]=part;errors.append(abs((obj(W+ep*d,w,b)-obj(W-ep*d,w,b))/(2*ep)-np.vdot(grad,d).real))
    errors.extend([abs((obj(W,w+ep,b)-obj(W,w-ep,b))/(2*ep)-gw[0]),abs((obj(W,w,b+ep)-obj(W,w,b-ep))/(2*ep)-gb)])
    assert max(errors)<1e-7
    a,ph=geometry();exact_error=[]
    for ap in range(4):
        A=ph[ap]*np.outer(a[ap],a[ap].conj())/2;u,_,vh=np.linalg.svd(A);M=K.mm(vh.conj().T,u.conj().T)
        exact_error.append(norm(M@(ph[ap]*a[ap])-a[ap]))
    assert max(exact_error)<1e-12
    T,ds,mats=estimate(world,DIST);assert abs(sum(ds['weights'])-1)<1e-12
    y=world['h1'][:23]+world['nb'][:23];expanded=np.zeros((23,8),complex)
    for ap in range(4):
        sl=slice(2*ap,2*ap+2);M=mats[f'ap{ap+1}_M'];expanded[:,sl]=ds['weights'][ap]*K.mv(M,y[:,sl])
        mm,dd,_=estimate(world,SINGLES[ap]);assert np.array_equal(mm,M)
    blockerr=norm(expanded-K.mv(T,y));assert blockerr<1e-12
    altered={k:v.copy() for k,v in world.items()};altered['cal0'][:,2:]*=3;altered['cal1'][:,2:]*=-2
    assert np.array_equal(estimate(altered,SINGLES[0])[0],estimate(world,SINGLES[0])[0])
    assert ds['unitarity_error']<1e-12 and estimate(world,CENT)[1]['unitarity_error']<1e-12
    return dict(finite_difference_max_abs_error=max(errors),exact_polar_return_errors=exact_error,zero_embedded_sum_block_diagonal_error=blockerr,local_independence_from_other_ap_data=True,shared_world_reproducible=True,phase_radians=np.angle(ph).tolist(),temporal_correlation_magnitudes=abs(ph).tolist(),distributed_seed0=ds)

def analyze(rows):
    derived=chosen(rows);dump(R/'results/selected_single.json',derived);allrows=rows+derived
    summary={m:{k:stats([r['service'][k] for r in allrows if r['method']==m]) for k in ['accuracy','ce']} for m in SINGLES+METHODS}
    dump(R/'results/summary.json',summary)
    raw=[dict(seed=r['seed'],method=r['method'],accuracy=r['service']['accuracy'],ce=r['service']['ce'],selected_ap=r.get('selected_ap',''),derived=r['method']==SINGLE) for r in allrows]
    with (R/'results/raw_metrics.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(raw[0]));wr.writeheader();wr.writerows(raw)
    dump(R/'results/claims.json',{line:json.loads((R/'results'/f'{line}.json').read_text()) for line in ['G30.1','G30.2','G30.3']})

def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    for p,h in json.loads((R/'freeze_manifest.json').read_text())['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
    dump(R/'results/preflight.json',preflight());print('PREFLIGHT PASS',flush=True)
    for seed in SEEDS:np.savez_compressed(R/'results'/f'world_{seed}.npz',**make_world(seed))
    rows=[]
    for methods,line in [(SINGLES,'G30.1'),([DIST],'G30.2'),([CENT],'G30.3'),([CSI],None)]:
        for seed in SEEDS:
            with np.load(R/'results'/f'world_{seed}.npz') as z:world={k:z[k] for k in z.files}
            for method in methods:
                row=train(world,seed,method);rows.append(row)
                with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                dump(R/'results/partial_runs.json',rows);print(seed,method,row['service'],flush=True)
        if line:adjudicate(rows,line)
    dump(R/'results/all_runs.json',rows);analyze(rows)
    dump(R/'results/runtime.json',dict(pid=os.getpid(),completed_training_rows=len(rows),wall_seconds=time.time()-start,finished_epoch=time.time()))
    print('DONE',len(rows),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'results/failure.txt').write_text(traceback.format_exc());raise
