"""One preregistered G33 experiment; no tuning or alternative jitter grids."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,pathlib,time,hashlib,csv,traceback,resource,signal
sys.dont_write_bytecode=True
import numpy as np
import g30_reference as G
K=G.K;R=G.R;SEEDS=G.SEEDS;N=G.N
CENT='centralized_polar_jitter';DIST='distributed_consensus_polar';CSI='CSI_jitter';SINGLE='single_AP_jitter'
CI='centralized_polar_ideal';DI='distributed_consensus_polar_ideal'
SINGLES=[f'single_AP_jitter_{i+1}' for i in range(4)]
METHODS=[CENT,CI,DIST,DI]+SINGLES+[CSI]
dump=G.dump;norm=G.norm

def quant(v,bound=4.,positive=False):
    lo=0. if positive else -bound
    q=lo+(bound-lo)/15*np.rint(15*(np.clip(v,lo,bound)-lo)/(bound-lo))
    return q,dict(scalars=v.size,clips=int(np.count_nonzero((v<lo)|(v>bound))),squared_error=float(np.sum((q-v)**2)),max_abs_error=float(np.max(abs(q-v))))

def qc(v):
    a,da=quant(v.real);b,db=quant(v.imag)
    return a+1j*b,dict(scalars=da['scalars']+db['scalars'],clips=da['clips']+db['clips'],squared_error=da['squared_error']+db['squared_error'],max_abs_error=max(da['max_abs_error'],db['max_abs_error']))

def make_world(seed):
    z=G.make_world(seed)
    z['phi_train']=G.rng(seed,11).uniform(-np.pi/3,np.pi/3,(N,4))
    z['phi_cal']=G.rng(seed,12).uniform(-np.pi/3,np.pi/3,(G.P,4))
    z['cal1_jitter']=z['cal1']*np.repeat(np.exp(1j*z['phi_cal']),2,axis=1)
    return z

def estimate(world,method):
    z=dict(world);z['cal1']=world['cal1_jitter']
    if method==CENT:
        f0,d0=qc(z['cal0']);f1,d1=qc(z['cal1']);T,d,m=G.polar(f0,f1)
        d['fronthaul_pilots']=[d0,d1]
        return T,d,m
    aps=[int(method[-1])-1] if method in SINGLES else list(range(4))
    ms=[];ds=[];mats={}
    for ap in aps:
        sl=slice(2*ap,2*ap+2);m,d,ma=G.polar(z['cal0'][:,sl],z['cal1'][:,sl]);ms.append(m);ds.append(d)
        mats.update({f'ap{ap+1}_{k}':v for k,v in ma.items()})
    power=np.array([d['signal_trace'] for d in ds]);qp,qd=quant(power,8.,True)
    assert min(power)>0 and min(qp)>0,'Nonpositive local/quantized power: no rescue floor'
    weights=qp/qp.sum();sigma=np.sqrt(power/2);sigma_cpu=np.sqrt(qp/2)
    mats.update(local_M=np.array(ms),signal_power=power,quantized_power=qp,weights=weights,sigma=sigma,sigma_cpu=sigma_cpu)
    return np.array(ms),dict(locals=ds,weights=weights.tolist(),sigma=sigma.tolist(),sigma_cpu=sigma_cpu.tolist(),strength_quantization=qd),mats

def train(world,seed,method):
    if method in [CI,DI]:
        orig=G.CENT if method==CI else G.DIST
        # Isolate inherited filenames: G.DIST equals the impaired method name.
        output_root=R/'tmp/baseline_models';(output_root/'results').mkdir(parents=True,exist_ok=True)
        previous_root=G.R
        try:
            G.R=output_root
            row=G.train(world,seed,orig)
            (output_root/'results'/f'model_{seed}_{orig}.npz').rename(R/'results'/f'model_{seed}_{method}.npz')
        finally:G.R=previous_root
        row['method']=method;return row
    ap=int(method[-1])-1 if method in SINGLES else None
    sl=slice(None) if ap is None else slice(2*ap,2*ap+2)
    W,w,b=G.initial(seed,ap);W0=W.copy();initial=G.service(world,W,w,b,ap)
    n=2666 if method==CSI else N;cal={};mats={};T=None
    if method!=CSI:T,cal,mats=estimate(world,method)
    aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float);trace=[];gainclips=0;maxpow=0.
    for start in range(0,n,32):
        ids=np.arange(start,min(start+32,n));inp=world['inputs'][ids];lab=world['labels'][ids]
        h0=world['h0'][ids,sl];h1=world['h1'][ids,sl]
        y=G.forward(h0,W,inp,world['nf'][ids]);loss,s,gw,gb,_=K.task(y,lab,w,b);s=s[:,0];truthgx=h0*s[:,None];qd={}
        if method==CSI:gx=truthgx
        else:
            mag=abs(s);tx=s/np.maximum(mag,1e-15);gain,gc=K.qreal(mag,8.,False);gainclips+=gc
            maxpow=max(maxpow,float(np.max(abs(tx)**2)));assert maxpow<=1+1e-12
            phase=np.repeat(np.exp(1j*world['phi_train'][ids]),2,axis=1)[:,sl]
            Y=phase*(h1*tx[:,None]+world['nb'][ids,sl])
            if method==CENT:
                yq,qd=qc(Y);gx=K.mv(T,yq)*gain[:,None]
            else:
                out=[];qds=[]
                for j in range(len(T)):
                    local=K.mv(T[j],Y[:,2*j:2*j+2])/mats['sigma'][j]
                    lq,dq=qc(local);qds.append(dq)
                    out.append(mats['weights'][j]*mats['sigma_cpu'][j]*lq)
                gx=np.concatenate(out,axis=1)*gain[:,None]
                qd=dict(scalars=sum(d['scalars'] for d in qds),clips=sum(d['clips'] for d in qds),squared_error=sum(d['squared_error'] for d in qds),max_abs_error=max(d['max_abs_error'] for d in qds))
        gW=K.pgrad(gx,inp);truth=K.pgrad(truthgx,inp)
        cosine=float(np.vdot(truth,gW).real/max(norm(truth)*norm(gW),1e-30))
        old=W.copy();W=aw.step(W,gW,.025);rawstep=norm(W-old);W*=min(1.,(1/np.sqrt(3))/max(norm(W),1e-15))
        par=np.clip(ar.step(np.r_[w,b],np.r_[gw,gb],.06),-4,4);w=par[:1];b=float(par[1]);assert np.isfinite(W).all() and np.isfinite(par).all()
        trace.append(dict(step=len(trace)+1,samples=int(ids[-1]+1),training_ce=float(loss.mean()),encoder_norm=norm(W),gradient_norm=norm(gW),true_gradient_norm=norm(truth),gradient_cosine=cosine,gradient_error=norm(gW-truth),unprojected_step=rawstep,receiver_w=float(w[0]),receiver_b=b,fronthaul=qd))
    costs=dict(training_examples=n,updates=len(trace),test_forward_evaluations=8192,actual_rf_slots=7998 if method==CSI else 7808,reserved_rf_slots=7998 if method==CSI else 8000,antenna_observations_per_return=2 if ap is not None else 8,seed_norm_bits=0 if method==CSI else 16*n,ideal_csi_privilege=method==CSI,reverse_payload_bits_per_example=0 if method==CSI else (16 if ap is not None else 64),static_strength_bits=4 if ap is not None else (16 if method==DIST else 0),calibration_payload_bits=8192 if method==CENT else 0)
    row=dict(seed=seed,method=method,ap=ap,initial_service=initial,service=G.service(world,W,w,b,ap),calibration=cal,costs=costs,trace=trace,gain_quant_clips=gainclips,max_reverse_tx_power=maxpow,mean_gradient_cosine=float(np.mean([t['gradient_cosine'] for t in trace])))
    np.savez_compressed(R/'results'/f'model_{seed}_{method}.npz',W=W,w=w,b=b,W0=W0,w0=np.array([.8]),b0=-.4,**mats)
    return row

def chosen(rows):
    result=[]
    for seed in SEEDS:
        rs=[r for r in rows if r['seed']==seed and r['method'] in SINGLES];assert len(rs)==4
        best=sorted(rs,key=lambda r:(-r['service']['accuracy'],r['service']['ce'],r['ap']))[0]
        result.append(dict(seed=seed,method=SINGLE,selected_method=best['method'],selected_ap=best['ap'],service=best['service'],selection='test oracle; derived not trained',oracle_selection_training_rf_slots=31232,oracle_selection_test_examples=32768))
    return result

def adjudicate(rows,line):
    lookup={(r['seed'],r['method']):r for r in rows}
    def v(m,k):return np.array([lookup[s,m]['service'][k] for s in SEEDS])
    if line=='G33.1':
        a=G.stats(v(CENT,'accuracy'));drop=G.stats(v(CI,'accuracy')-v(CENT,'accuracy'))
        support=a['mean']<=.75 and drop['mean']>=.15;fail=a['mean']>=.82;metrics=dict(central_accuracy=a,accuracy_drop=drop)
    elif line=='G33.2':
        a=G.stats(v(DIST,'accuracy'));ag=G.stats(v(DIST,'accuracy')-v(CENT,'accuracy'));cg=G.stats(v(CENT,'ce')-v(DIST,'ce'))
        support=a['mean']>=.88 and ag['mean']>=.10 and cg['mean']>=.15;fail=a['mean']<.80 or ag['mean']<.05
        metrics=dict(distributed_accuracy=a,accuracy_gain=ag,ce_gain=cg)
    else:
        delta=v(DIST,'ce')-v(DI,'ce');gap=G.stats(delta)
        support=gap['mean']<=.08;fail=gap['mean']>.10
        metrics=dict(ce_gap=gap,absolute_mean_gap=abs(gap['mean']),mean_absolute_per_seed_gap=float(abs(delta).mean()))
    result=dict(claim=line,verdict='archived' if fail else ('archived' if support else 'archived'),support_threshold_met=bool(support),falsification_triggered=bool(fail),**metrics)
    dump(R/'results'/f'{line}.json',result)
    event=dict(time=time.time(),completed_training_rows=len(rows),claim=line,triggered=bool(fail),action='STOP: claim closed, no further exploration' if fail else 'terminal batch closed, no expansion',result=result)
    with (R/'results/decisions.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
    if fail:dump(R/'results'/f'STOP_{line}.json',event)
    print('ADJUDICATED',line,json.dumps(result),flush=True)
    return result

def preflight():
    base=G.preflight();world=make_world(SEEDS[0]);other=make_world(SEEDS[0]);assert all(np.array_equal(world[k],other[k]) for k in world)
    for k in ['phi_train','phi_cal']:
        assert np.max(abs(world[k]))<=np.pi/3
        expanded=np.repeat(world[k],2,axis=1);assert np.array_equal(expanded[:,::2],expanded[:,1::2])
    levels=np.linspace(-4,4,16);q,_=quant(levels);assert np.max(abs(q-levels))<1e-14
    edge,qd=quant(np.array([-5.,-4.,0.,4.,5.]));assert qd['clips']==2 and edge[0]==-4 and edge[-1]==4
    ms,d,ma=estimate(world,DIST);assert abs(sum(d['weights'])-1)<1e-12
    altered={k:v.copy() for k,v in world.items()};altered['cal0'][:,2:]*=3;altered['cal1_jitter'][:,2:]*=-2
    assert np.array_equal(estimate(world,SINGLES[0])[0],estimate(altered,SINGLES[0])[0])
    local_errors=[norm(K.mm(m.conj().T,m)-np.eye(2)) for m in ms]
    m,_,_=estimate(world,CENT);assert norm(K.mm(m.conj().T,m)-np.eye(8))<1e-12 and max(local_errors)<1e-12
    Y=world['h1'][:23];truew=ma['signal_power']/ma['signal_power'].sum()
    plain=np.concatenate([truew[j]*K.mv(ms[j],Y[:,2*j:2*j+2]) for j in range(4)],axis=1)
    agc=np.concatenate([truew[j]*ma['sigma'][j]*(K.mv(ms[j],Y[:,2*j:2*j+2])/ma['sigma'][j]) for j in range(4)],axis=1)
    agcerr=norm(plain-agc);assert agcerr<1e-12
    # Static diagonal phase admits a global unitary inverse; a fixed inverse cannot undo new random phase.
    _,ph=G.geometry();delta=np.array([-.7,.4,.9,-.2]);D=np.diag(np.repeat(ph*np.exp(1j*delta),2))
    u,_,vh=np.linalg.svd(D);inv=K.mm(vh.conj().T,u.conj().T)
    exact=norm(K.mv(inv,K.mv(D,world['h0'][:23]))-world['h0'][:23]);assert exact<1e-12
    local0=G.estimate(world,G.DIST)[2]['ap1_M'];raw=Y[:,:2];random_phase=np.exp(1j*world['phi_train'][:23,0])[:,None]
    equiv=norm(K.mv(local0,raw*random_phase)-K.mv(local0,raw)*random_phase);assert equiv<1e-12
    residual=norm(K.mv(local0,raw*random_phase)-K.mv(local0,raw));assert residual>0
    return dict(inherited=base,quantizer_test_values=edge.tolist(),local_unitarity_errors=local_errors,agc_lossless_cancellation_error=agcerr,global_static_phase_exact_inverse_error=exact,local_phase_equivariance_error=equiv,local_random_phase_not_cancelled_norm=residual,analytic_mean_phase=float(np.sin(np.pi/3)/(np.pi/3)),empirical_mean_phase=[float(np.cos(world['phi_train']).mean()),float(np.sin(world['phi_train']).mean())],same_ap_phase_shared=True,world_reproducible=True,local_information_isolation=True)

def analyze(rows):
    derived=chosen(rows);dump(R/'results/selected_single.json',derived);rr=rows+derived
    summary={m:{k:G.stats([r['service'][k] for r in rr if r['method']==m]) for k in ['accuracy','ce']} for m in METHODS+[SINGLE]}
    dump(R/'results/summary.json',summary)
    raw=[dict(seed=r['seed'],method=r['method'],accuracy=r['service']['accuracy'],ce=r['service']['ce'],selected_ap=r.get('selected_ap',''),derived=r['method']==SINGLE) for r in rr]
    with (R/'results/raw_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(raw[0]));w.writeheader();w.writerows(raw)
    dump(R/'results/claims.json',{s:json.loads((R/'results'/f'{s}.json').read_text()) for s in ['G33.1','G33.2','G33.3']})

def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900));signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    for p,h in json.loads((R/'freeze_manifest.json').read_text())['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
    dump(R/'results/preflight.json',preflight());print('PREFLIGHT PASS',flush=True)
    for seed in SEEDS:np.savez_compressed(R/'results'/f'world_{seed}.npz',**make_world(seed))
    rows=[]
    for methods,lines in [([CENT,CI],['G33.1']),([DIST,DI],['G33.2','G33.3']),(SINGLES+[CSI],[])]:
        for seed in SEEDS:
            with np.load(R/'results'/f'world_{seed}.npz') as z:world={k:z[k] for k in z.files}
            for method in methods:
                row=train(world,seed,method);rows.append(row)
                with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                dump(R/'results/partial_runs.json',rows);print(seed,method,row['service'],flush=True)
        for line in lines:adjudicate(rows,line)
    dump(R/'results/all_runs.json',rows);analyze(rows)
    dump(R/'results/runtime.json',dict(pid=os.getpid(),completed_training_rows=len(rows),wall_seconds=time.time()-start,finished_epoch=time.time()))
    print('DONE',len(rows),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'results/failure.txt').write_text(traceback.format_exc());raise
