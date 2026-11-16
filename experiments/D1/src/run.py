"""Frozen G15 physical OFDM experiment, independent taps, correlated tones."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,time,resource,signal,traceback,csv
import numpy as np
import learning_core as K
R=pathlib.Path(__file__).resolve().parents[1]
SEEDS=list(range(815150000,815150008))
METHODS=['S0','wideband_avg_polar','subcarrier_polar','CSI']
P=np.exp(-np.arange(4)/2);P/=P.sum()
N=8

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def csvwrite(p,rows):
    with p.open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def geometry():
    th=2*np.pi*np.arange(128)/128
    s0=np.exp(-2j*np.pi*np.arange(3)[:,None]*.5*np.cos(th))/np.sqrt(128)
    s1=s0*np.exp(2j*np.pi*.383*np.cos(th))
    r=s0@s0.conj().T;t=s1@s0.conj().T;a=np.linalg.solve(r.T,t.T).T
    q=r-a@r@a.conj().T;q=(q+q.conj().T)/2
    def root(m):
        v,u=np.linalg.eigh(m);assert v.min()>-1e-10
        return (u*np.sqrt(np.maximum(v,0)))@u.conj().T
    rt=root(r);joint=np.block([[rt,np.zeros((3,3))],[a@rt,root(q)]])
    err=np.linalg.norm(joint@joint.conj().T-np.block([[r,t.conj().T],[t,r]]))
    assert err<1e-10
    return r,t,a,joint,float(err)

def taps(rng,n,joint):
    return np.einsum('ij,bljn->blin',joint,K.cn(rng,(n,4,6,2)))*np.sqrt(P)[None,:,None,None]
def tones(h):return np.fft.fft(h,n=8,axis=1)
def covariances(f,b):
    fq,clip=K.qcomplex(f,4.)
    r=np.einsum('bkmn,bkln->kml',fq,fq.conj())/(2*len(f))-.01*np.eye(3)
    t=np.einsum('bkmn,bkln->kml',b,fq.conj())/(2*len(f))
    return r,t,clip

def polar(r,t):
    reg=(r+r.conj().T)/2+.02*np.eye(3)
    d,v=np.linalg.eigh(reg);dc=np.maximum(d,.02)
    reg=(v*dc)@v.conj().T
    a=np.linalg.solve(reg.T,t.T).T
    u,s,vh=np.linalg.svd(a);m=u@vh
    mq,clip=K.qcomplex(m,32.);uu,_,vv=np.linalg.svd(mq);m=uu@vv
    assert np.linalg.norm(m.conj().T@m-np.eye(3))<1e-12
    return m,a,reg,dict(matrix_clips=clip,floor_count=int((d<.02).sum()),condition=float(dc.max()/dc.min()),singular_values=s.tolist())

class ToneWorld(K.World):
    def __init__(self,data,k):
        for name in ['inputs','labels','test_inputs','test_labels','_f','_b','_nf','_nb','_test_f','_test_n','test_epsilon']:
            setattr(self,name,data[name][:,k])
    def pilot_forward(self,ids):return self._f[ids] # ideal CSI, deliberately no estimation noise

def make_data(seed,joint):
    rng=np.random.default_rng(seed+91000);n=3840
    inp=rng.choice([-1.,1.],(n,8,3));tr=taps(rng,n,joint);h=tones(tr)
    data=dict(inputs=inp,labels=(inp[:,:,0]*inp[:,:,1]>0).astype(float),_f=h[:,:,:3],_b=h[:,:,3:],
              _nf=.1*K.cn(rng,(n,8,3)),_nb=.03*K.cn(rng,(n,8,2)))
    crng=np.random.default_rng(seed+81000);cal=taps(crng,64,joint);ch=tones(cal)
    cf=ch[:,:,:3]+.1*K.cn(crng,(64,8,3,2));cb=ch[:,:,3:]+.1*K.cn(crng,(64,8,3,2))
    trng=np.random.default_rng(seed+101000);ti=trng.choice([-1.,1.],(8192,8,3));tt=taps(trng,8192,joint)[:,:,:3];tf=tones(tt)
    data.update(test_inputs=ti,test_labels=(ti[:,:,0]*ti[:,:,1]>0).astype(float),_test_f=tf,
                _test_n=.1*K.cn(trng,(8192,8,3)),test_epsilon=np.zeros((8192,8,2),complex))
    # Raw generative arrays suffice to reconstruct every training/test channel.
    np.savez_compressed(R/f'results/raw_{seed}.npz',train_taps=tr,test_taps=tt,cal_taps=cal,cal_fhat=cf,cal_bhat=cb,
        inputs=inp,test_inputs=ti,forward_noise=data['_nf'],reverse_noise=data['_nb'],test_noise=data['_test_n'])
    return data,cf,cb

def preflight(r,t,a,joint,err):
    out=K.verify();rng=np.random.default_rng(815159998)
    ht=taps(rng,8192,joint);hf=tones(ht)
    # CP serial linear convolution equals diagonal frequency-domain channel.
    x=K.cn(rng,(8,2));xt=np.fft.ifft(x,axis=0,norm='ortho');xcp=np.concatenate([xt[-4:],xt])
    yc=np.zeros((15,3),complex)
    for i in range(3):
        for j in range(2):yc[:,i]+=np.convolve(xcp[:,j],ht[0,:,i,j])
    yy=np.fft.fft(yc[4:12],axis=0,norm='ortho');expected=np.einsum('kij,kj->ki',hf[0,:,:3],x)
    cp_err=float(np.max(abs(yy-expected)));assert cp_err<1e-12
    parseval=float(abs(np.sum(abs(xt)**2)-np.sum(abs(x)**2)));assert parseval<1e-12
    rc=np.einsum('bkin,bkjn->kij',hf[:,:,:3],hf[:,:,:3].conj())/(2*len(ht))
    tc=np.einsum('bkin,bkjn->kij',hf[:,:,3:],hf[:,:,:3].conj())/(2*len(ht))
    rerr=float(np.max(abs(rc-r)));terr=float(np.max(abs(tc-t)));assert max(rerr,terr)<.07
    power=np.mean(abs(ht[:,:,:3])**2,axis=(0,2,3));assert np.max(abs(power-P))<.02
    fq=hf[:64,:,:3]+.1*K.cn(rng,(64,8,3,2));bq=hf[:64,:,3:]+.1*K.cn(rng,(64,8,3,2))
    rh,th,_=covariances(fq,bq)
    mg,*_=polar(rh.mean(0),th.mean(0))
    fflat=fq.transpose(0,1,2,3).reshape(-1,1,3,2);bflat=bq.reshape(-1,1,3,2)
    rp,tp,_=covariances(fflat,bflat);mm,*_=polar(rp[0],tp[0]);avg_err=float(np.linalg.norm(mg-mm));assert avg_err<1e-12
    samef=np.repeat(fq[:,:1],8,axis=1);sameb=np.repeat(bq[:,:1],8,axis=1)
    sr,st,_=covariances(samef,sameb);sm,*_=polar(sr[0],st[0]);gm,*_=polar(sr.mean(0),st.mean(0));assert np.linalg.norm(sm-gm)<1e-12
    u,sv,vh=np.linalg.svd(a);m=u@vh;z=a.conj().T@m
    assert np.linalg.norm(z-z.conj().T)<1e-12 and np.linalg.eigvalsh(z).min()>-1e-12
    # Analytic cross-tone covariance and direct tap phase construction.
    cross=np.einsum('bin,bjn->ij',hf[:,1,:3],hf[:,0,:3].conj())/(2*len(ht))
    theoretical=np.sum(P*np.exp(-2j*np.pi*np.arange(4)/8))*r
    xerr=float(np.max(abs(cross-theoretical)));assert xerr<.07
    nz=K.cn(rng,(100000,));nv=float(np.mean(abs(.03*nz)**2));assert abs(nv-.0009)<.00003
    out.update(joint_factor_error=err,cp_fft_error=cp_err,parseval_error=parseval,max_R_covariance_error=rerr,
        max_T_covariance_error=terr,tap_powers=power.tolist(),target_tap_powers=P.tolist(),
        average_covariance_order_error=avg_err,cross_tone_covariance_error=xerr,reverse_noise_variance=nv,
        physical_polar_sign_check=True,identical_tones_same_estimate=True,
        analytic_A_spread=0.,analytic_M_spread=0.,instantaneous_adjacent_tone_difference=float(np.mean(abs(hf[:,1]-hf[:,0])**2)))
    np.savez(R/'results/theory.npz',R=r,T=t,A=a,M=m,powers=P,joint=joint)
    return out

def stats(values):
    v=np.asarray(values,float);assert v.shape==(8,)
    ix=np.random.default_rng(815159999).integers(0,8,(20000,8))
    return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),per_seed=v.tolist())

def summarize(rows):
    by=[]
    for s in SEEDS:
        for m in METHODS:
            rr=[r for r in rows if r['seed']==s and r['method']==m];assert len(rr)==8
            by.append(dict(seed=s,method=m,ce=float(np.mean([r['service']['ce'] for r in rr])),accuracy=float(np.mean([r['service']['accuracy'] for r in rr]))))
    csvwrite(R/'results/by_seed.csv',by)
    arms={m:{key:stats([r[key] for r in by if r['method']==m]) for key in ['ce','accuracy']} for m in METHODS}
    def contrast(a,b,key):return stats(np.array(arms[a][key]['per_seed'])-arms[b][key]['per_seed'])
    con=dict(S0_minus_wide_CE=contrast('S0','wideband_avg_polar','ce'),wide_minus_sub_CE=contrast('wideband_avg_polar','subcarrier_polar','ce'),
             sub_minus_wide_accuracy=contrast('subcarrier_polar','wideband_avg_polar','accuracy'),sub_minus_CSI_CE=contrast('subcarrier_polar','CSI','ce'))
    wa=arms['wideband_avg_polar']['accuracy']['mean'];sa=arms['subcarrier_polar']['accuracy']['mean'];gap=con['sub_minus_CSI_CE']['mean']
    verdict={
        'G15.1':dict(falsified=wa>=.85,supported=wa<.70 and con['S0_minus_wide_CE']['mean']<.10 and con['S0_minus_wide_CE']['ci95'][0]<=0<=con['S0_minus_wide_CE']['ci95'][1]),
        'G15.2':dict(falsified=sa<.80 or con['sub_minus_wide_accuracy']['mean']<.10,supported=sa>=.92 and con['sub_minus_wide_accuracy']['mean']>=.20 and con['wide_minus_sub_CE']['mean']>=.30),
        'G15.3':dict(falsified=gap>.10,supported=gap<=.05)}
    dump(R/'results/summary.json',dict(arms=arms,contrasts=con,verdict=verdict,independent_units=8,tones_per_unit=8))
    dump(R/'results/STOP.json',{'terminal':True,'lines':{k:'closed_falsifier' if v['falsified'] else 'closed_prespecified_endpoint' for k,v in verdict.items()},'no_followup_grid':True})
    return verdict

def main():
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall budget 1800s')));signal.alarm(1800)
    r,t,a,joint,err=geometry();dump(R/'results/preflight.json',preflight(r,t,a,joint,err));print('PREFLIGHT PASSED',flush=True)
    rows=[];diagnostics=[]
    for seed in SEEDS:
        data,cf,cb=make_data(seed,joint);rh,th,clips=covariances(cf,cb)
        mg,ag,cg,ig=polar(rh.mean(0),th.mean(0));local=[polar(rh[k],th[k]) for k in range(8)]
        np.savez_compressed(R/f'results/calibration_{seed}.npz',C00=rh,C10=th,Mglobal=mg,Aglobal=ag,Creg_global=cg,
            Mlocal=np.stack([v[0] for v in local]),Alocal=np.stack([v[1] for v in local]),Creg_local=np.stack([v[2] for v in local]))
        diagnostics.append(dict(seed=seed,channel_clips=clips,global_info=ig,local_info=[v[3] for v in local],
            local_M_distance_to_global=[float(np.linalg.norm(v[0]-mg)) for v in local],
            global_A_relative_error=float(np.linalg.norm(ag-a)/np.linalg.norm(a)),
            local_A_relative_error=[float(np.linalg.norm(v[1]-a)/np.linalg.norm(a)) for v in local]))
        for k in range(8):
            world=ToneWorld(data,k)
            for method in METHODS:
                if method=='S0':internal='raw'
                elif method=='CSI':internal='csi_rf_budget'
                else:
                    internal='matrix_est';m,ah,reg,info=(mg,ag,cg,ig) if method=='wideband_avg_polar' else local[k]
                    K.fit_temporal=lambda *args,**kwargs:(ah,m,1.,dict(info,channel_clips=clips))
                row=K.train(world,seed,'isotropic/parallel/0.383/OFDM8',internal,a)
                row.update(method=method,subcarrier=k,ideal_CSI=(method=='CSI'))
                # Core costs are per tone. Global sharing is amortized over tones.
                if method=='wideband_avg_polar':
                    row['costs']['correction_matrix_sharing_bits']=36;row['costs']['control_bits']-=252
                    for rate in K.C['control_bits_per_rf_slot_scenarios']:
                        row['costs'][f'serial_slots_if_{rate}_bits_per_rf_duration']=row['costs']['rf_observations']+row['costs']['control_bits']/rate
                row['costs']['tone_RF_resource_units']=row['costs']['rf_observations']
                row['costs']['updates']=120 if method!='CSI' else 84
                assert row['gain_quant_clips']==0 and row['mean_reverse_energy']<=1+1e-12
                rows.append(row)
        dump(R/'results/partial_runs.json',rows);dump(R/'results/calibration_diagnostics.json',diagnostics)
        print(json.dumps(dict(seed_complete=seed,rows=len(rows),elapsed=time.time()-start)),flush=True)
    dump(R/'results/all_runs.json',rows)
    csvwrite(R/'results/by_tone.csv',[dict(seed=r['seed'],method=r['method'],subcarrier=r['subcarrier'],ce=r['service']['ce'],accuracy=r['service']['accuracy']) for r in rows])
    verdict=summarize(rows)
    use=resource.getrusage(resource.RUSAGE_SELF);dump(R/'results/runtime.json',dict(pid=os.getpid(),wall_seconds=time.time()-start,cpu_user=use.ru_utime,cpu_system=use.ru_stime,rows=len(rows)))
    print('DONE',json.dumps(verdict),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'results/failure.txt').write_text(traceback.format_exc());raise
