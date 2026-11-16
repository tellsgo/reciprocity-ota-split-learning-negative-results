"""G32 frozen two-SI-transmitter, four-receiver learning proxy."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, pathlib, json, time, hashlib, traceback, resource, signal, csv
import numpy as np
import learning_core as K

R=pathlib.Path(__file__).resolve().parents[1]
SEEDS=list(range(263200000,263200008))
N=3840; NT=8192; PILOTS=64
METHODS=['naive_polar','ZF_SIC_polar','MMSE_SIC_polar','half_duplex_polar','CSI']
GAMMA=.1
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False))

def rng(seed,stream): return np.random.default_rng(np.random.SeedSequence([seed,stream]))

def norm(x): return float(np.linalg.norm(x))

def steering(theta): return np.exp(-2j*np.pi*.5*np.arange(4)*np.cos(np.deg2rad(theta)))

def initial(seed,stream):
    W=K.cn(rng(seed,stream),(4,3));return W/norm(W)*.4,np.array([.8]),-.4

def forward(h,W,inp,noise):
    x=K.mv(W,inp)
    assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-12
    return (np.einsum('bi,bi->b',h.conj(),x)+noise)[:,None]

def service(world,W,w,b):
    y=forward(world['test_h'],W,world['test_inputs'],world['test_n'])
    loss,_,_,_,prob=K.task(y,world['test_labels'],w,b)
    return {'ce':float(loss.mean()),'accuracy':float(np.mean((prob>=.5)==world['test_labels']))}

def geometry():
    a=steering(30)
    x=(np.arange(4)-1.5)*.5
    angles=np.deg2rad([15.,-35.])
    sources=np.stack([np.sin(angles),np.cos(angles)],axis=1)
    distances=np.sqrt((x[:,None]-sources[None,:,0])**2+sources[None,:,1]**2)
    b=np.exp(-2j*np.pi*distances)/distances
    b=b/np.linalg.norm(b,axis=0,keepdims=True)*2
    phase=np.exp(2j*np.pi*.383*np.cos(np.deg2rad(30)))
    return a,b,phase,distances

def channels(gen,n):
    a,_,phase,_=geometry();h0=K.cn(gen,(n,1))*a[None,:]
    return h0,h0*phase

def qpsk(gen,n): return np.array([1.,1j,-1.,-1j])[gen.integers(0,4,n)]

def make_world(seed):
    gen=rng(seed,1);inp=gen.choice([-1.,1.],(N,3));h0,h1=channels(gen,N)
    out={'inputs':inp,'labels':(inp[:,0]*inp[:,1]>0).astype(float),
         'h0':h0,'h1':h1,'nf':.1*K.cn(gen,(N,)),'nb':.03*K.cn(gen,(N,4))}
    gen=rng(seed,2);c0,c1=channels(gen,PILOTS)
    out.update(cal0=c0+.1*K.cn(gen,c0.shape),cal1=c1+.1*K.cn(gen,c1.shape),cal_true0=c0,cal_true1=c1)
    q=qpsk(gen,PILOTS);qcal=np.stack([q,q*(-1.)**np.arange(PILOTS)],axis=1)
    out.update(qcal=qcal,si_n0=.1*K.cn(gen,(PILOTS,4)),si_n1=.1*K.cn(gen,(PILOTS,4)))
    gen=rng(seed,3);inp=gen.choice([-1.,1.],(NT,3));t0,_=channels(gen,NT)
    out.update(test_inputs=inp,test_labels=(inp[:,0]*inp[:,1]>0).astype(float),test_h=t0,test_n=.1*K.cn(gen,(NT,)))
    out['si_q']=qpsk(rng(seed,6),(N,2))
    return out

def estimate(world,method,db):
    I=np.eye(4,dtype=complex);a,B,phase,_=geometry();rho=10.**(db/10)
    H=np.sqrt(rho/2)*B
    si=K.mv(H,world['qcal']);v0=si+world['si_n0'];v1=si+world['si_n1']
    # Orthogonal simultaneous pilots, rows are two known Tx symbols.
    Hhat=K.mm(v1.T,world['qcal'].conj())/PILOTS
    Bhat=Hhat/np.sqrt(rho/2)
    P=I-K.mm(K.mm(Bhat,np.linalg.inv(K.mm(Bhat.conj().T,Bhat))),Bhat.conj().T)
    cov=K.mm(Hhat,Hhat.conj().T)
    Wmmse=np.linalg.inv(cov+GAMMA*I)
    S=GAMMA*Wmmse
    proj=P if method=='ZF_SIC_polar' else (S if method=='MMSE_SIC_polar' else I)
    if method=='half_duplex_polar': f0=world['cal0'];f1=world['cal1'];noise=.01
    else:f0=world['cal0']+v0;f1=world['cal1']+v1;noise=.02
    f0=K.mv(proj,f0);f1=K.mv(proj,f1)
    f0,fc=K.qcomplex(f0,4.)
    raw=K.mm(f0.T,f0.conj())/PILOTS-noise*K.mm(proj,proj.conj().T)
    cross=K.mm(f1.T,f0.conj())/PILOTS
    reg=(raw+raw.conj().T)/2+.02*I
    ev,vr=np.linalg.eigh(reg);reg=K.mm(vr*np.maximum(ev,.02),vr.conj().T)
    A=K.mm(cross,np.linalg.inv(reg));u,sv,vh=np.linalg.svd(A);Q=K.mm(vh.conj().T,u.conj().T)
    Qraw=Q.copy();Q,qc=K.qcomplex(Q,32.);u,_,vh=np.linalg.svd(Q);Q=K.mm(u,vh)
    T=K.mm(K.mm(proj,Q),proj)
    residual=norm(K.mm(proj,H))**2;noiseP=.03**2*norm(proj)**2
    diag={'pilot_clips':fc,'correction_clips':qc,'A_singular_values':sv.tolist(),
      'covariance_floor_count':int((ev<.02).sum()),'Creg_condition':float(np.linalg.cond(reg)),
      'P_hermitian_error':norm(P-P.conj().T),'P_idempotence_error':norm(K.mm(P,P)-P),
      'P_rank':int(np.linalg.matrix_rank(P,tol=1e-10)),'estimated_null_residual':norm(K.mm(P,Bhat)),
      'Bhat_relative_error':norm(Bhat-B)/norm(B),
      'filter_eigenvalues':np.linalg.eigvalsh(proj).tolist(),
      'target_projected_energy_fraction':norm(K.mv(proj,a[None,:]))**2/4,
      'projected_si_power':float(residual),'projected_noise_power':float(noiseP),
      'projected_si_noise_ratio':float(residual/noiseP),
      'corrected_si_power':norm(K.mm(T,H))**2,'corrected_noise_power':float(.03**2*norm(T)**2),
      'Q_unitarity_error':norm(K.mm(Q.conj().T,Q)-I),'Q_quantization_error':norm(Q-Qraw),
      'T_spectral_norm':float(np.linalg.norm(T,2)),
      'target_effective_alignment_real':float(np.vdot(a,K.mv(T,(phase*a)[None,:])[0]).real/4),
      'target_effective_relative_error':norm(K.mv(T,(phase*a)[None,:])[0]-a)/2,
      'soft_filter_ZF_distance':norm(S-P)}
    mats={'P':P,'Hhat_si':Hhat,'Bhat_si':Bhat,'S':S,'Wmmse':Wmmse,'filter':proj,'Q':Q,'T':T,'A':A,'Creg':reg,'C10':cross,'F0_quantized':f0,'F1':f1}
    return T,diag,mats

def train(world,seed,method,db):
    W,w,b=initial(seed,4);initial_service=service(world,W,w,b)
    n=2666 if method=='CSI' else N;T=None;cal={};mats={}
    if method!='CSI':T,cal,mats=estimate(world,method,db)
    aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float);tr=[];gainclips=0
    errors=[];cosines=[];maxpow=0.;totalraw=0.;totaleffective=0.;totaldesired=0.;zero_tx=0
    rho=10.**(db/10);_,si_b,_,_=geometry()
    for start in range(0,n,32):
        ids=np.arange(start,min(start+32,n));inp=world['inputs'][ids];lab=world['labels'][ids]
        y=forward(world['h0'][ids],W,inp,world['nf'][ids])
        loss,s,gw,gb,_=K.task(y,lab,w,b);s=s[:,0]
        truegx=world['h0'][ids]*s[:,None]
        if method=='CSI':gx=truegx
        else:
            mag=abs(s);tx=s/np.maximum(mag,1e-15)
            gain,gc=K.qreal(mag,8.,False);gainclips+=gc
            maxpow=max(maxpow,float(np.max(abs(tx)**2)))
            assert maxpow<=1+1e-12 and np.max(mag)<=8+1e-12
            inter=np.zeros((len(ids),4),complex) if method=='half_duplex_polar' else K.mv(np.sqrt(rho/2)*si_b,world['si_q'][ids])
            totaldesired+=float(np.sum(abs(world['h1'][ids]*tx[:,None])**2));zero_tx+=int(np.count_nonzero(mag<1e-15))
            received=world['h1'][ids]*tx[:,None]+inter+world['nb'][ids]
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
    if method=='CSI': cost={'target_reserved_rf':7998,'actual_system_rf':7998,'seed_norm_bits':0,'ideal_csi_privilege':True}
    else:
        cal_slots=128 if method=='half_duplex_polar' else 256
        cost={'target_reserved_rf':8000,'actual_system_rf':2*N+cal_slots,'seed_norm_bits':16*N,
              'calibration_channel_sharing_bits':64*4*2*16,'correction_sharing_bits':4*4*2*16,
              'actual_calibration_slots':cal_slots,'reserved_calibration_slots':320,
              'concurrent_si_transmissions':0 if method=='half_duplex_polar' else 2*N,'ideal_csi_privilege':False}
    cost.update(training_examples=n,updates=len(tr),test_forward_evaluations=NT)
    row={'seed':seed,'method':method,'isr_db':db if method in ['ZF_SIC_polar','MMSE_SIC_polar','naive_polar'] else None,'initial_service':initial_service,'service':service(world,W,w,b),
         'calibration':cal,'costs':cost,'trace':tr,'gain_quant_clips':gainclips,'max_reverse_tx_power':maxpow,
         'mean_gradient_error':float(np.mean(errors)),'mean_gradient_cosine':float(np.mean(cosines)),
         'desired_return_energy':totaldesired,'zero_gradient_transmissions':zero_tx,
         'empirical_isr':totalraw/max(totaldesired,1e-30),'raw_interference_energy':totalraw,'corrected_interference_energy':totaleffective}
    np.savez_compressed(R/'results'/f'model_{seed}_{method}_{db}.npz',W=W,w=w,b=b,**mats)
    return row

def stats(vals):
    a=np.asarray(vals);assert a.shape==(8,)
    idx=np.random.default_rng(263299999).integers(0,8,(20000,8))
    return {'mean':float(a.mean()),'ci95':np.quantile(a[idx].mean(1),[.025,.975]).tolist(),
            'per_seed':a.tolist(),'sd':float(a.std(ddof=1))}

def preflight():
    gen=rng(SEEDS[0],7);inp=gen.choice([-1.,1.],(23,3));lab=(inp[:,0]*inp[:,1]>0).astype(float)
    h=K.cn(gen,(23,4));noise=.1*K.cn(gen,(23,));W,w,b=initial(SEEDS[0],4)
    def obj(W,w,b): return K.task(forward(h,W,inp,noise),lab,w,b)[0].mean()+.001*np.sum(w*w)/2
    _,s,gw,gb,_=K.task(forward(h,W,inp,noise),lab,w,b);gW=K.pgrad(h*s,inp)
    err=[];eps=1e-6
    for ix in np.ndindex(W.shape):
        for part in [1.,1j]:
            d=np.zeros_like(W);d[ix]=part
            err.append(abs((obj(W+eps*d,w,b)-obj(W-eps*d,w,b))/(2*eps)-np.vdot(gW,d).real))
    err.append(abs((obj(W,w+eps,b)-obj(W,w-eps,b))/(2*eps)-gw[0]))
    err.append(abs((obj(W,w,b+eps)-obj(W,w,b-eps))/(2*eps)-gb));assert max(err)<1e-7
    a,B,phase,dist=geometry();I=np.eye(4)
    P=I-K.mm(K.mm(B,np.linalg.inv(K.mm(B.conj().T,B))),B.conj().T)
    assert np.linalg.matrix_rank(B)==2 and np.linalg.matrix_rank(P,tol=1e-10)==2
    assert norm(K.mm(P,B))<1e-12 and norm(K.mm(P,P)-P)<1e-12
    A=phase*np.outer(a,a.conj())/4;u,_,vh=np.linalg.svd(A);Q=K.mm(vh.conj().T,u.conj().T)
    polar_error=norm(K.mv(Q,(phase*a)[None,:])[0]-a);assert polar_error<1e-12
    world=make_world(SEEDS[0]);other=make_world(SEEDS[0]);assert all(np.array_equal(world[k],other[k]) for k in world)
    assert norm(K.mm(world['qcal'].conj().T,world['qcal'])/PILOTS-np.eye(2))<1e-12
    # Enumerate the full two-QPSK alphabet: analytic SI covariance identity.
    alphabet=np.array([[x,y] for x in [1,1j,-1,-1j] for y in [1,1j,-1,-1j]])
    ds={}
    for method in METHODS[:-1]:
        T,d,m=estimate(world,method,20);ds[method]=d
        assert d['P_rank']==2 and d['P_idempotence_error']<1e-12 and d['Q_unitarity_error']<1e-12
        assert d['estimated_null_residual']<1e-12 and d['T_spectral_norm']<=1+1e-10
        Y=K.mv(np.sqrt(50)*B,alphabet)
        explicit=float(np.mean(np.sum(abs(K.mv(T,Y))**2,axis=1)))
        assert abs(explicit-d['corrected_si_power'])<1e-10
        assert norm(K.mm(K.mm(m['Hhat_si'],m['Hhat_si'].conj().T)+GAMMA*I,m['S'])-GAMMA*I)<1e-10
    assert np.max(abs(world['si_q'])**2)==1
    return {'finite_difference_max_abs_error':max(err),'rank_one_polar_direction_error':polar_error,
      'si_distances_lambda':dist.tolist(),'si_singular_values':np.linalg.svd(B,compute_uv=False).tolist(),
      'ideal_target_retained_energy_fraction':norm(K.mv(P,a[None,:]))**2/4,
      'paired_world_exactly_reproducible':True,'single_path_channel_correlation_modulus':float(abs(phase)),
      'diagnostics_seed0_20db':ds}

def claim(name,support,falsifier,values):
    out={'verdict':'archived' if falsifier else ('archived' if support else 'archived'),
         'support_threshold_met':bool(support),'falsification_triggered':bool(falsifier),**values}
    if falsifier:
        dump(R/'results'/f'STOP_{name}.json',{'claim':name,'time':time.time(),'reason':out,'action':'closed; no new seeds or rescue configurations'})
    return out

def summarize(rows,db):
    lookup={(r['seed'],r['method']):r for r in rows if r['isr_db'] is None or r['isr_db']==db}
    summary={m:{k:stats([lookup[s,m]['service'][k] for s in SEEDS]) for k in ['accuracy','ce']} for m in METHODS}
    con={}
    for label,m1,m2,key in [('accuracy_MMSE_minus_ZF','MMSE_SIC_polar','ZF_SIC_polar','accuracy'),
      ('ce_ZF_minus_MMSE','ZF_SIC_polar','MMSE_SIC_polar','ce'),('ce_MMSE_minus_half','MMSE_SIC_polar','half_duplex_polar','ce')]:
        con[label]=stats([lookup[s,m1]['service'][key]-lookup[s,m2]['service'][key] for s in SEEDS])
    return {'summary':summary,'contrasts':con}

def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    for f,h in json.loads((R/'freeze_manifest.json').read_text())['hashes'].items():
        assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
    dump(R/'results/preflight.json',preflight());print('PREFLIGHT PASS',flush=True)
    rows=[];worlds={};claims={};analysis={};tested=[]
    def run(seed,method,db):
        row=train(worlds[seed],seed,method,db);rows.append(row)
        with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        dump(R/'results/partial_runs.json',rows)
        print(seed,method,db,row['service'],flush=True)
    for seed in SEEDS:
        worlds[seed]=make_world(seed);np.savez_compressed(R/'results'/f'world_{seed}.npz',**worlds[seed])
        for method in ['half_duplex_polar','CSI']:run(seed,method,0)
    for db in [20,10,0]:
        for seed in SEEDS:
            for method in ['naive_polar','ZF_SIC_polar','MMSE_SIC_polar']:run(seed,method,db)
        tested.append(db);report=summarize(rows,db);analysis[str(db)]=report
        sm=report['summary'];con=report['contrasts']
        if db==20:
            za=sm['ZF_SIC_polar']['accuracy']['mean'];ma=sm['MMSE_SIC_polar']['accuracy']['mean']
            gain=con['accuracy_MMSE_minus_ZF']['mean'];ce=con['ce_ZF_minus_MMSE']['mean']
            claims['G32.1']=claim('G32.1',za<=.80,za>=.88,{'ZF_accuracy':za})
            claims['G32.2']=claim('G32.2',ma>=.90 and gain>=.08 and ce>=.15,ma<.82 or gain<.04,
              {'MMSE_accuracy':ma,'accuracy_gain':gain,'ce_gain':ce})
        gap=con['ce_MMSE_minus_half']['mean']
        dump(R/'results/analysis.json',analysis);dump(R/'results/claims_partial.json',claims)
        print('BATCH_COMPLETE',db,'CE_GAP',gap,flush=True)
        if gap>.10:
            claims['G32.3']=claim('G32.3',False,True,{'tested_isr_db':tested,'falsifying_isr_db':db,'ce_gap':gap})
            break
    if 'G32.3' not in claims:
        gaps={db:analysis[str(db)]['contrasts']['ce_MMSE_minus_half']['mean'] for db in tested}
        claims['G32.3']=claim('G32.3',len(tested)==3 and max(gaps.values())<=.06,False,{'ce_gaps':gaps,'tested_isr_db':tested})
    dump(R/'results/claims.json',claims);dump(R/'results/all_runs.json',rows)
    raw=[{'seed':r['seed'],'method':r['method'],'isr_db':r['isr_db'],**r['service'],
      'mean_gradient_cosine':r['mean_gradient_cosine'],'pilot_clips':r['calibration'].get('pilot_clips',0),
      'empirical_isr':r['empirical_isr'],'si_noise_ratio':r['calibration'].get('projected_si_noise_ratio',None)} for r in rows]
    with (R/'results/raw_metrics.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(raw[0]));wr.writeheader();wr.writerows(raw)
    dump(R/'results/runtime.json',{'pid':os.getpid(),'wall_seconds':time.time()-start,'completed_rows':len(rows),
      'tested_isr_db':tested,'omitted_isr_db':[db for db in [20,10,0] if db not in tested],'finished_epoch':time.time()})
    print('DONE',claims,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'results/failure.txt').write_text(traceback.format_exc());raise
