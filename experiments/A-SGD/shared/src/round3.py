"""Frozen round-three harness; reads training source/config only inside this run."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,json,pathlib,time,hashlib,platform,traceback
import numpy as np
from scipy.special import jn_zeros
import experiment as X
K,L,G=X.K,X.L,X.G
R=pathlib.Path(__file__).resolve().parents[2]
SEEDS=list(range(9400000,9400008)); SGDSEEDS=list(range(9600000,9600008));TUNE=list(range(9800000,9800003))
GRID=[.01,.03,.1,.3,1.,3.,10.];NEAR=[.38,.4,.42];FAR=[.55,.7];D1=float(jn_zeros(1,1)[0]/(2*np.pi))
E4CASES=[('isotropic',d) for d in NEAR+FAR]
E5CASES=[(sp,d) for sp in ['broad','k50'] for d in [.1,.2,.3,.4,.5]]
CASES=E4CASES+[('isotropic',D1)]+E5CASES
DEPLOY=['phase_64','pinv_64','polar_64'];E4METHODS=['phase_64','scalar_oracle','pinv_64','pinv_oracle','polar_64','polar_oracle','CSI']
CUTOFF=.1; LARGE=2.

def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def boot(v):
 v=np.asarray(v,float);assert v.shape==(8,) and np.isfinite(v).all()
 ids=np.random.default_rng(9999999).integers(0,8,(20000,8))
 return {'mean':float(v.mean()),'ci95':np.quantile(v[ids].mean(1),[.025,.975]).tolist(),'per_seed':v.tolist(),'positive_seeds':int((v>0).sum()),'negative_seeds':int((v<0).sum())}
def correction(a,kind,quantized=False):
 # B=A^H=U S Vh, M_pinv=V S+ U^H; M_polar=V U^H.
 B=a.conj().T;u,s,vh=np.linalg.svd(B);v=vh.conj().T;keep=s>CUTOFF*s[0]
 if kind=='pinv':
  si=np.zeros_like(s);si[keep]=1/s[keep];m=(v*si)@u.conj().T
 else:m=v@u.conj().T
 exact=m.copy();clips=0;decode_err=0.
 if quantized:
  m,clips=G.qcomplex(m,32.)
  if kind=='polar':
   qu,_,qvh=np.linalg.svd(m);qm=qu@qvh;decode_err=float(np.linalg.norm(qm-m));m=qm
 return m,{'singular_values':s.tolist(),'rank':int(keep.sum()) if kind=='pinv' else 3,'cutoff':float(CUTOFF*s[0]) if kind=='pinv' else None,'matrix_clips':clips,'matrix_quantized':quantized,'quant_decode_error_fro':float(np.linalg.norm(m-exact)),'polar_decode_reprojection_fro':decode_err,'M_spectral_norm':float(np.linalg.norm(m,2)),'M_unitarity_error':float(np.linalg.norm(m.conj().T@m-np.eye(3))),'M':X.pack(m),'A':X.pack(a)}

def geo(sp,d):
 g=X.geo(sp,d);mp,di=correction(g['a'],'pinv');sing=np.linalg.svd(g['a'],compute_uv=False)
 g['info'].update(exact_pinv_M_norm=di['M_spectral_norm'],norm_group='large' if di['M_spectral_norm']>LARGE else 'small',exact_pinv_rank=di['rank'],exact_A_singular_values=sing.tolist(),untruncated_pinv_norm=float(1/sing[-1]))
 return g

def train(seed,g,m,w=None,optimizer='Adam',lr=None):
 w=L.World(seed,g,g['jr']) if w is None else w
 oracle=m.endswith('_oracle');matrix=m.startswith(('pinv','polar'))
 K.C.update(optimizer=optimizer,encoder_adam_lr=.025 if lr is None else lr,receiver_adam_lr=.06 if lr is None else lr,scalar_calibration_pairs=0 if oracle else 64,matrix_calibration_pairs=0 if oracle else 64)
 if m=='CSI':aa=np.eye(3);corr=np.eye(3);eff=1.;ci={};internal='csi_rf_budget'
 elif matrix:
  if oracle:aa=g['a'];base={}
  else:aa,_,_,base=L.fit_temporal(w,64,False)
  corr,ci=correction(aa,m.split('_')[0],not oracle)
  ci.update(channel_clips=base.get('channel_clips',0),covariance_floor_count=base.get('covariance_floor_count',0))
  eff=1.;internal='matrix_est'
 else:
  if oracle:alpha=g['alpha'];ci={}
  else:aa,_,_,ci=L.fit_temporal(w,64,True);alpha=complex(aa[0,0])
  aa=alpha*np.eye(3);corr=np.eye(3);eff=np.exp(1j*np.angle(alpha));internal='scalar_est'
  ci.update(estimated_rho=X.pack(alpha),phase_error_degrees=float(np.angle(alpha*np.conj(g['alpha']),deg=True)),floor_applied=False)
 def fit(*args,**kwargs):return aa,corr,eff,ci
 K.fit_temporal=fit
 row=K.train(w,seed,f"{g['info']['spectrum']}/{g['info']['displacement_lambda']}",internal,g['a'])
 row.update(method=m,geometry=g['info'],optimizer=optimizer,learning_rate=lr)
 co=row['costs'];co.update(oracle_statistics=oracle,calibration_used=not oracle and m!='CSI',matrix_payload_equivalent_bits=288 if matrix else 0,oracle_free_matrix=bool(matrix and oracle),return_wait_displacement_lambda=g['info']['displacement_lambda'],return_wait_seconds_example=g['info']['displacement_lambda']*(299792458/3.5e9)/15)
 if matrix and oracle:co['control_bits']-=co['correction_matrix_sharing_bits'];co['correction_matrix_sharing_bits']=0
 for rate in [1,16,64]:co[f'serial_slots_if_{rate}_bits_per_rf_duration']=co['rf_observations']+co['control_bits']/rate
 assert co['rf_observations']==(7998 if m=='CSI' else (7680 if oracle else 8000))
 assert co['control_bits']==(170624 if m=='CSI' else (61440 if oracle else 73728+(288 if matrix else 0)))
 return row

def contrast(rows,seeds,cases,a,b):
 idx={(r['seed'],r['geometry']['spectrum'],r['geometry']['displacement_lambda'],r['method']):r['service']['ce'] for r in rows}
 return boot([np.mean([idx[(s,sp,d,a)]-idx[(s,sp,d,b)] for sp,d in cases]) for s in seeds])
def m4(rows,seeds):
 cases=sorted(set((r['geometry']['spectrum'],r['geometry']['displacement_lambda']) for r in rows))
 cases=[c for c in cases if any(r['method']=='polar_64' and (r['geometry']['spectrum'],r['geometry']['displacement_lambda'])==c for r in rows)]
 groups={k:[c for c in cases if geo(*c)['info']['norm_group']==k] for k in ['large','small']}
 per={f'{sp}/{d}':contrast(rows,seeds,[(sp,d)],'pinv_64','polar_64') for sp,d in cases}
 grp={k:contrast(rows,seeds,cc,'pinv_64','polar_64') if cc else None for k,cc in groups.items()}
 allst=[v for v in grp.values() if v]+list(per.values())
 falsifiers=[k for k,v in dict(per,**grp).items() if v and v['mean']<-.02 and v['ci95'][1]<0]
 support=all(grp.values()) and all(v['ci95'][0]>-.02 for v in allst) and grp['large']['mean']>=.02 and grp['large']['ci95'][0]>0
 return {'groups':grp,'group_cases':groups,'per_geometry':per,'falsifiers':falsifiers,'verdict':'archived（archived）' if falsifiers else ('archived0.02archived（archived）' if support else 'archived'),'strict_nonworse_all_CI_nonnegative':all(v['ci95'][0]>=0 for v in allst)}

def e4(p):
 rows=[]
 for sp,d in E4CASES+[('isotropic',D1)]:
  g=geo(sp,d)
  for s in SEEDS:
   w=L.World(s,g,g['jr'])
   for m in (['phase_64'] if d==D1 else E4METHODS):rows.append(train(s,g,m,w))
  dump(p/'results/partial_runs.json',rows);print('E4',sp,d,len(rows),flush=True)
 dump(p/'results/all_runs.json',rows)
 idx={(r['seed'],r['geometry']['displacement_lambda'],r['method']):r['service']['ce'] for r in rows}
 gaps={m:np.array([np.mean([idx[s,d,m] for d in NEAR])-np.mean([idx[s,d,m] for d in FAR]) for s in SEEDS]) for m in ['scalar_oracle','pinv_oracle','polar_oracle']}
 out={'gaps':{m:boot(v) for m,v in gaps.items()},'spatial':{},'primary':'polar'}
 for kind in ['polar','pinv']:
  ga=boot(gaps[kind+'_oracle']);red=boot(gaps['scalar_oracle']-gaps[kind+'_oracle'])
  gain=contrast(rows,SEEDS,E4CASES[:3],'phase_64',kind+'_64')
  non=boot([np.mean([idx[s,d,kind+'_64'] for d in NEAR])-idx[s,D1,'phase_64'] for s in SEEDS])
  v1='archived（archived）' if ga['mean']>=.126 else ('archived（archived）' if ga['ci95'][1]<.126 and red['ci95'][0]>0 else 'archived')
  v2='archived（archived）' if gain['mean']<.02 or gain['ci95'][0]<=0<=gain['ci95'][1] else ('archived' if gain['mean']>=.02 and gain['ci95'][0]>0 else 'archived')
  v3='archived（archived0.02）' if non['ci95'][1]<=.02 else 'archived（archived）'
  out['spatial'][kind]={'M1_gap':ga,'paired_gap_reduction':red,'M1_verdict':v1,'M2_improvement':gain,'M2_verdict':v2,'M3_difference':non,'M3_verdict':v3}
 out['M4_Adam_partial']=m4(rows,SEEDS)
 dump(p/'results/primary.json',out)

def e6(p):
 tune=[]
 for sp,d in CASES:
  g=geo(sp,d)
  for s in TUNE:
   w=L.World(s,g,g['jr'])
   for m in DEPLOY:
    for lr in GRID:tune.append(train(s,g,m,w,'SGD',lr))
  dump(p/'results/tuning_partial.json',tune);print('E6 tune',sp,d,len(tune),flush=True)
 dump(p/'results/tuning_all_runs.json',tune)
 scores={m:{str(lr):float(np.mean([r['service']['ce'] for r in tune if r['method']==m and r['learning_rate']==lr])) for lr in GRID} for m in DEPLOY}
 selected={m:min(GRID,key=lambda lr:scores[m][str(lr)]) for m in DEPLOY}
 dump(p/'results/selected_learning_rates.json',{'selected':selected,'scores':scores,'frozen_before_evaluation_unix':time.time()});print('E6 selected',selected,flush=True)
 rows=[]
 for sp,d in CASES:
  g=geo(sp,d)
  for s in SGDSEEDS:
   w=L.World(s,g,g['jr'])
   for m in DEPLOY:rows.append(train(s,g,m,w,'SGD',selected[m]))
  dump(p/'results/partial_runs.json',rows);print('E6 eval',sp,d,len(rows),flush=True)
 dump(p/'results/all_runs.json',rows);out=m4(rows,SGDSEEDS);out['selected_learning_rates']=selected;dump(p/'results/primary.json',out)

def e5(p):
 rows=[]
 for sp,d in E5CASES+[('isotropic',D1)]:
  g=geo(sp,d)
  for s in SEEDS:
   w=L.World(s,g,g['jr'])
   for m in (['pinv_64','polar_64'] if sp=='isotropic' else DEPLOY+['CSI']):rows.append(train(s,g,m,w))
  dump(p/'results/partial_runs.json',rows);print('E5',sp,d,len(rows),flush=True)
 dump(p/'results/all_runs.json',rows)
 def stats(cc):return {a+'-minus-'+b:contrast(rows,SEEDS,cc,a,b) for a,b in [('phase_64','pinv_64'),('phase_64','polar_64'),('pinv_64','polar_64')]}
 out={'per_geometry':{f'{sp}/{d}':stats([(sp,d)]) for sp,d in E5CASES},'by_spectrum':{sp:stats([c for c in E5CASES if c[0]==sp]) for sp in ['broad','k50']},'all_far':stats(E5CASES)}
 e4rows=json.loads((R/'E4/results/all_runs.json').read_text());out['M4_Adam']=m4(e4rows+rows,SEEDS)
 dump(p/'results/primary.json',out)

def verify():
 out=K.verify();rng=np.random.default_rng(13310001);a=K.cn(rng,(3,3));u,s,vh=np.linalg.svd(a.conj().T);right=(vh.conj().T*s)@vh;left=(u*s)@u.conj().T
 m,di=correction(a,'polar');err=float(np.linalg.norm(a.conj().T@m-left));assert err<1e-12
 assert np.linalg.norm(a.conj().T@m-right)>1e-3
 mp,_=correction(a,'pinv');assert np.linalg.norm(mp-np.linalg.pinv(a.conj().T,rcond=.1))<1e-12
 gd=[]
 for sp,d in CASES:
  g=geo(sp,d);pol,diag=correction(g['a'],'polar',True)
  assert np.linalg.norm(pol.conj().T@pol-np.eye(3))<1e-12
  unq,_=correction(g['a'],'polar');b=g['a'].conj().T@unq
  assert np.linalg.norm(b-b.conj().T)<1e-10 and np.linalg.eigvalsh((b+b.conj().T)/2).min()>-1e-10
  gd.append(g['info'])
 # New streams separated from earlier known independent training seeds, including seed offsets.
 sets=[{s+o for s in ss for o in [81000,91000,101000,111000]} for ss in [SEEDS,SGDSEEDS,TUNE,list(range(6100000,6100008)),list(range(8200000,8200008)),list(range(8400000,8400008)),list(range(8600000,8600003))]]
 assert all(not sets[i]&sets[j] for i in range(len(sets)) for j in range(i))
 # The estimator consumed by the new arms is exactly the original returned Ahat.
 g=geo('isotropic',.4);w=L.World(13320001,g,g['jr']);ah,_,_,_=L.fit_temporal(w,64,False)
 f,_=G.qcomplex(w.cal_fhat,4);r=np.einsum('bmn,bln->ml',f,f.conj())/128-.01*np.eye(3);cross=np.einsum('bmn,bln->ml',w.cal_bhat,f.conj())/128
 z=(r+r.conj().T)/2+.02*np.eye(3);ev,vec=np.linalg.eigh(z);direct=cross@(vec*(1/np.maximum(ev,.02)))@vec.conj().T
 assert np.linalg.norm(ah-direct)<1e-12
 # Exact zero-tangent guard and normalizer bounds remain inherited.
 out.update(polar_left_factor_error=err,polar_right_factor_mismatch=float(np.linalg.norm(a.conj().T@m-right)),pinv_reference_match=True,polar_quantized_decode_unitary=True,Ahat_original_estimator_match=True,independent_streams=True,geometry=gd)
 return out

def freeze(v):
 files=[R/v/'SPEC.md']+list((R/v/'src').glob('*.py'))+list((R/'shared/src').glob('*.py'))+list((R/'shared/configs').glob('*.json'))+[R/'infra/gtask.py']
 dump(R/v/'freeze_manifest.json',{'time_unix':time.time(),'hashes':{str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}})
def main(v):
 p=R/v;t=time.time()
 for f,h in json.loads((p/'freeze_manifest.json').read_text())['hashes'].items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
 try:
  dump(p/'results/preflight.json',verify());{'E4':e4,'E6':e6,'E5':e5}[v](p)
 except BaseException:
  (p/'results/failure.txt').write_text(traceback.format_exc());raise
 dump(p/'results/runtime.json',{'host':platform.node(),'pid':os.getpid(),'wall_seconds':time.time()-t,'experiment':v})
 print('DONE',v,time.time()-t,flush=True)
if __name__=='__main__':main(sys.argv[1])
