"""Frozen G23: physical frequency dependent array and displacement phases."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import sys
sys.dont_write_bytecode=True
import pathlib,json,hashlib,time,resource,signal,traceback,csv
import numpy as np
import learning_core as K
R=pathlib.Path(__file__).resolve().parents[1]; SEEDS=K.C['seeds']; METHODS=K.C['methods']; P=np.array([.5,.3,.2]); TH=np.deg2rad([30.,60.,90.]); NR=4

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def csvwrite(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def geom(eta):
 ratio=1+(np.arange(8)-3.5)*eta/8
 v=np.exp(1j*np.pi*ratio[:,None,None]*np.arange(4)[None,:,None]*np.cos(TH)[None,None,:])
 g=np.exp(2j*np.pi*.383*ratio[:,None]*np.cos(TH))
 phase=np.exp(-2j*np.pi*np.arange(8)[:,None]*np.arange(3)/8)
 factor=v*phase[:,None,:]*np.sqrt(P)[None,None,:]
 rr=np.einsum('kil,l,kjl->kij',v,P,v.conj());tt=np.einsum('kil,kl,l,kjl->kij',v,g,P,v.conj())
 aa=tt@np.linalg.pinv(rr,rcond=1e-10)
 return dict(ratio=ratio,V=v,gamma=g,factor=factor,R=rr,T=tt,A=aa)
def channel(h,geo):
 f=np.einsum('kil,bln->bkin',geo['factor'],h)
 b=np.einsum('kil,kl,bln->bkin',geo['factor'],geo['gamma'],h)
 return f,b

def covariances(f,b):
 fq,clip=K.qcomplex(f,4.)
 return (np.einsum('bkin,bkjn->kij',fq,fq.conj())/(2*len(f))-.01*np.eye(4),np.einsum('bkin,bkjn->kij',b,fq.conj())/(2*len(f)),clip)
def polar(r,t):
 reg=(r+r.conj().T)/2+.02*np.eye(4);d,v=np.linalg.eigh(reg);dc=np.maximum(d,.02);reg=(v*dc)@v.conj().T
 a=np.linalg.solve(reg.T,t.T).T;u,s,vh=np.linalg.svd(a);m=u@vh
 mq,clip=K.qcomplex(m,32.);uu,_,vv=np.linalg.svd(mq);m=uu@vv
 assert np.linalg.norm(m.conj().T@m-np.eye(4))<1e-12
 return m,a,reg,dict(matrix_clips=clip,floor_count=int((d<.02).sum()),condition=float(dc.max()/dc.min()),singular_values=s.tolist())
class ToneWorld(K.World):
 def __init__(self,data,k):
  for name,value in data.items():setattr(self,name,value[:,k])
 def pilot_forward(self,ids):return self._f[ids]
def make_data(seed,geo,folder):
 rng=np.random.default_rng(seed+91000);n=3840
 inp=rng.choice([-1.,1.],(n,8,3));h=K.cn(rng,(n,3,2));f,b=channel(h,geo)
 nf=.1*K.cn(rng,(n,8,4));nb=.03*K.cn(rng,(n,8,2))
 crng=np.random.default_rng(seed+81000);ch=K.cn(crng,(64,3,2));cf,cb=channel(ch,geo)
 cnf=.1*K.cn(crng,(64,8,4,2));cnb=.1*K.cn(crng,(64,8,4,2));cf=cf+cnf;cb=cb+cnb
 trng=np.random.default_rng(seed+101000);ti=trng.choice([-1.,1.],(8192,8,3));th=K.cn(trng,(8192,3,2));tf,_=channel(th,geo);tn=.1*K.cn(trng,(8192,8,4))
 np.savez_compressed(folder/f'raw_{seed}.npz',train_h=h,test_h=th,cal_h=ch,cal_fhat=cf,cal_bhat=cb,cal_forward_noise=cnf,cal_reverse_noise=cnb,inputs=inp,test_inputs=ti,forward_noise=nf,reverse_noise=nb,test_noise=tn)
 return dict(inputs=inp,labels=(inp[:,:,0]*inp[:,:,1]>0).astype(float),_f=f,_b=b,_nf=nf,_nb=nb,test_inputs=ti,test_labels=(ti[:,:,0]*ti[:,:,1]>0).astype(float),_test_f=tf,_test_n=tn,test_epsilon=np.zeros((8192,8,2),complex)),cf,cb

def preflight(eta,geo,folder):
 rng=np.random.default_rng(K.C['preflight_seed']);h=K.cn(rng,(8192,3,2));f,b=channel(h,geo)
 rr=np.einsum('bkin,bkjn->kij',f,f.conj())/(2*len(h));tt=np.einsum('bkin,bkjn->kij',b,f.conj())/(2*len(h))
 re=float(np.max(abs(rr-geo['R'])));te=float(np.max(abs(tt-geo['T'])));assert max(re,te)<.07
 ranks=[int(np.linalg.matrix_rank(r,tol=1e-10)) for r in geo['R']];assert ranks==[3]*8
 direct=geo['V']*geo['gamma'][:,None,:]@np.linalg.pinv(geo['V'],rcond=1e-10)
 aerr=float(np.max(abs(direct-geo['A'])));assert aerr<1e-11
 maperr=float(np.max(abs(b-np.einsum('kij,bkjn->bkin',geo['A'],f))));assert maperr<1e-11
 partial=[];full=[];sv=[]
 for a in geo['A']:
  u,s,vh=np.linalg.svd(a);m=u@vh;partial.append(u[:,:3]@vh[:3]);full.append(m);sv.append(s)
  z=a.conj().T@m;assert np.linalg.norm(z-z.conj().T)<1e-12 and np.linalg.eigvalsh(z).min()>-1e-12
 # Full eight-tap effective impulse: CP7 is an identity check, not a CP3 sufficiency claim.
 x=K.cn(rng,(8,2));xt=np.fft.ifft(x,axis=0,norm='ortho');tap=np.fft.ifft(f[0],axis=0);xcp=np.concatenate([xt[-7:],xt]);y=np.zeros((22,4),complex)
 for i in range(4):
  for j in range(2):y[:,i]+=np.convolve(xcp[:,j],tap[:,i,j])
 cp_err=float(np.max(abs(np.fft.fft(y[7:15],axis=0,norm='ortho')-np.einsum('kij,kj->ki',f[0],x))));assert cp_err<1e-12
 parseval=float(abs(np.sum(abs(xt)**2)-np.sum(abs(x)**2)));assert parseval<1e-12
 rh,th,_=covariances(f[:64],b[:64]);mg,*_=polar(rh.mean(0),th.mean(0))
 rp,tp,_=covariances(f[:64].reshape(-1,1,4,2),b[:64].reshape(-1,1,4,2));mp,*_=polar(rp[0],tp[0]);avg=float(np.linalg.norm(mg-mp));assert avg<1e-10
 zero=geom(0.);assert np.max(abs(zero['A']-zero['A'][0]))<1e-12
 part=np.array(partial);sv=np.array(sv)
 out=dict(eta=eta,ranks=ranks,max_R_error=re,max_T_error=te,exact_mapping_error=maperr,pinv_identity_error=aerr,cp7_effective_channel_error=cp_err,parseval_error=parseval,average_covariance_error=avg,eta0_identity_check=True,A_pairwise_diameter=float(max(np.linalg.norm(a-c) for a in geo['A'] for c in geo['A'])),partial_polar_pairwise_diameter=float(max(np.linalg.norm(a-c) for a in part for c in part)),signal_projector_pairwise_diameter=float(max(np.linalg.norm(a@np.linalg.pinv(a)-c@np.linalg.pinv(c)) for a in geo['V'] for c in geo['V'])),A_singular_values=sv.tolist(),ratio=geo['ratio'].tolist(),frequency_displacement_phase=True)
 np.savez(folder/'theory.npz',**geo,partial_polar=part,full_svd_polar=np.array(full))
 return out

def stats(values):
 v=np.asarray(values);assert v.shape==(8,);ix=np.random.default_rng(K.C['bootstrap_seed']).integers(0,8,(20000,8))
 return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),per_seed=v.tolist())
def summarize(rows,eta,folder):
 by=[]
 for s in SEEDS:
  for m in METHODS:
   rr=[r for r in rows if r['seed']==s and r['method']==m];assert len(rr)==8
   by.append(dict(seed=s,method=m,ce=float(np.mean([r['service']['ce'] for r in rr])),accuracy=float(np.mean([r['service']['accuracy'] for r in rr]))))
 csvwrite(folder/'by_seed.csv',by)
 arms={m:{key:stats([r[key] for r in by if r['method']==m]) for key in ['ce','accuracy']} for m in METHODS}
 def contrast(a,b,key):return stats(np.array(arms[a][key]['per_seed'])-arms[b][key]['per_seed'])
 con=dict(wide_minus_sub_CE=contrast('wideband_avg_polar','subcarrier_polar','ce'),sub_minus_wide_accuracy=contrast('subcarrier_polar','wideband_avg_polar','accuracy'),wide_minus_CSI_CE=contrast('wideband_avg_polar','CSI','ce'),sub_minus_CSI_CE=contrast('subcarrier_polar','CSI','ce'))
 wa=arms['wideband_avg_polar']['accuracy']['mean'];sa=arms['subcarrier_polar']['accuracy']['mean'];ag=con['sub_minus_wide_accuracy']['mean'];cg=con['wide_minus_sub_CE']['mean'];gap=con['wide_minus_CSI_CE']['mean']
 if eta==.20:
  f1=wa>=.88 and gap<.10;f2=sa<.80 or ag<.04
  verdict={'G23.1':dict(falsified=f1,supported=not f1 and (wa<.85 or gap>=.15)),'G23.2':dict(falsified=f2,supported=not f2 and sa>=.92 and ag>=.08 and cg>=.12)}
 else:verdict={'G23.3':dict(falsified=abs(ag)>.03,supported=abs(ag)<=.01 and abs(cg)<=.02)}
 out=dict(eta=eta,arms=arms,contrasts=con,verdict=verdict,independent_units=8,tones_per_unit=8)
 dump(folder/'summary.json',out);dump(folder/'STOP.json',dict(terminal=True,lines={k:'closed_falsifier' if v['falsified'] else 'closed_fixed_endpoint' for k,v in verdict.items()},no_followup_grid=True))
 return out

def main():
 start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall budget')));signal.alarm(1800)
 dump(R/'results/gradient_check.json',K.verify());allrows=[];summaries={}
 for eta in [.20,.02]:
  folder=R/f'results/eta_{eta:.2f}';folder.mkdir();geo=geom(eta);dump(folder/'preflight.json',preflight(eta,geo,folder));print('PREFLIGHT',eta,'PASSED',flush=True)
  rows=[];diagnostics=[]
  for seed in SEEDS:
   data,cf,cb=make_data(seed,geo,folder);rh,th,clips=covariances(cf,cb);globalfit=polar(rh.mean(0),th.mean(0));local=[polar(rh[k],th[k]) for k in range(8)];mg,ag,cg,ig=globalfit
   np.savez_compressed(folder/f'calibration_{seed}.npz',C00=rh,C10=th,Mglobal=mg,Aglobal=ag,Creg_global=cg,Mlocal=np.array([q[0] for q in local]),Alocal=np.array([q[1] for q in local]),Creg_local=np.array([q[2] for q in local]))
   diagnostics.append(dict(seed=seed,channel_clips=clips,global_info=ig,local_info=[v[3] for v in local],local_M_distance_to_global=[float(np.linalg.norm(v[0]-mg)) for v in local]))
   for k in range(8):
    world=ToneWorld(data,k)
    for method in METHODS:
     if method=='S0':internal='raw'
     elif method=='CSI':internal='csi_rf_budget'
     else:
      internal='matrix_est';m,ah,reg,info=globalfit if method=='wideband_avg_polar' else local[k]
      K.fit_temporal=lambda *args,**kwargs:(ah,m,1.,dict(info,channel_clips=clips))
     row=K.train(world,seed,f'eta{eta:.2f}',internal,geo['A'][k]);row.update(method=method,eta=eta,subcarrier=k,ideal_CSI=method=='CSI')
     if method=='wideband_avg_polar':
      row['costs']['correction_matrix_sharing_bits']=64;row['costs']['control_bits']-=448
      for rate in K.C['control_bits_per_rf_slot_scenarios']:row['costs'][f'serial_slots_if_{rate}_bits_per_rf_duration']=row['costs']['rf_observations']+row['costs']['control_bits']/rate
     row['costs']['updates']=120 if method!='CSI' else 84
     assert row['gain_quant_clips']==0 and row['mean_reverse_energy']<=1+1e-12
     rows.append(row)
   dump(folder/'partial_runs.json',rows);dump(folder/'calibration_diagnostics.json',diagnostics);print(json.dumps(dict(eta=eta,seed_complete=seed,rows=len(rows),elapsed=time.time()-start)),flush=True)
  dump(folder/'all_runs.json',rows);csvwrite(folder/'by_tone.csv',[dict(seed=r['seed'],method=r['method'],subcarrier=r['subcarrier'],ce=r['service']['ce'],accuracy=r['service']['accuracy']) for r in rows]);summaries[f'{eta:.2f}']=summarize(rows,eta,folder);allrows+=rows
  print('CLOSED',eta,json.dumps(summaries[f'{eta:.2f}']['verdict']),flush=True)
 dump(R/'results/summary.json',summaries);dump(R/'results/all_runs.json',allrows);u=resource.getrusage(resource.RUSAGE_SELF);dump(R/'results/runtime.json',dict(pid=os.getpid(),wall_seconds=time.time()-start,cpu_user=u.ru_utime,cpu_system=u.ru_stime,rows=len(allrows)));print('DONE',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:
  (R/'results/failure.txt').write_text(traceback.format_exc());raise
