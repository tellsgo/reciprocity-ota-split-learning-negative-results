"""Frozen independent harness. Source task/Jacobians reused; no input writes."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import pathlib,json,time,hashlib,sys,signal,platform,traceback
import numpy as np
from scipy.special import j0,jn_zeros
import run_learning as L
import learning_core as K
import geometry_core as G
R=pathlib.Path(__file__).resolve().parents[2]
SEEDS=list(range(6100000,6100008))
def pack(z):return {'real':np.asarray(z).real.tolist(),'imag':np.asarray(z).imag.tolist()}
def geo(spectrum,d,direction='parallel'):
 th=np.arange(128)*2*np.pi/128;mu=np.deg2rad(35)
 if spectrum=='isotropic':p=np.ones(128)
 elif spectrum=='broad':p=.7*np.exp(2*np.cos(th-mu))/np.i0(2)+.3
 else:
  k=float(spectrum[1:]);p=np.exp(k*(np.cos(th-mu)-1))
 p/=p.sum();u=np.cos(th) if direction=='parallel' else np.sin(th)
 s0=np.exp(-2j*np.pi*np.arange(3)[:,None]*.5*np.cos(th))*np.sqrt(p)
 st=s0*np.exp(2j*np.pi*d*u);r=s0@s0.conj().T;t=st@s0.conj().T
 a=np.linalg.solve(r.T,t.T).T;q=r-a@r@a.conj().T;q=(q+q.conj().T)/2
 def root(m):
  e,v=np.linalg.eigh(m);assert e.min()>-1e-9
  return (v*np.sqrt(np.maximum(e,0)))@v.conj().T
 rt=root(r);qr=root(q);jr=np.block([[rt,np.zeros((3,3))],[a@rt,qr]])
 err=np.linalg.norm(jr@jr.conj().T-np.block([[r,t.conj().T],[t,r]]));assert err<1e-9
 rho=np.trace(t)/np.trace(r);dop=2*np.pi*d*sum(p*u)
 info={'spectrum':spectrum,'direction':direction,'displacement_lambda':d,'rho':pack(rho),'magnitude':float(abs(rho)),'phase_degrees':float(np.angle(rho,deg=True)),'doppler_degrees':float(np.rad2deg(dop)),'joint_factor_error':float(err),'R0_condition':float(np.linalg.cond(r))}
 return {'r0':r,'rt':t,'a':a,'q':q,'alpha':rho,'info':info,'jr':jr,'root':rt,'qr':qr}
def bootstrap(vals):
 vals=np.array(vals,float);ids=np.random.default_rng(6199999).integers(0,len(vals),(10000,len(vals)))
 return {'mean':float(vals.mean()),'ci95':np.quantile(vals[ids].mean(1),[.025,.975]).tolist(),'positive_seeds':int(sum(vals>0)),'per_seed':vals.tolist()}
def phase_alpha(alpha,method,g):
 phi=np.angle(alpha)
 if method=='raw':return 1+0j
 if method=='doppler':return np.exp(1j*np.deg2rad(g['info']['doppler_degrees']))
 if method=='phase2':phi=np.round(phi/(np.pi/2))*(np.pi/2)
 if method=='phase1':phi=0 if np.cos(phi)>=0 else np.pi
 if method=='scalar':return alpha
 return np.exp(1j*phi) if abs(alpha)>0 else 0j
BASE_FIT=L.fit_temporal

def train_geo(s,g,method):
 w=L.World(s,g,g['jr']);a0,_,a,ci=BASE_FIT(w,64,True)
 def fit(world,k,scalar=False):return a0,np.eye(3),phase_alpha(a,method,g),ci
 K.fit_temporal=fit
 row=K.train(w,s,f"{g['info']['spectrum']}/{g['info']['displacement_lambda']}", 'csi_rf_budget' if method=='csi' else 'scalar_est',g['a'])
 row['method']=method;row['geometry']=g['info'];row['calibration_alpha']=pack(a0[0,0]);row['costs']['calibration_used']=method not in ['raw','doppler','csi'];row['costs']['privileged_mean_doppler']=method=='doppler'
 return row

def summary(rows):
 cases={}
 for row in rows:
  key=row['case'];cases.setdefault(key,{})
 for key in cases:
  rr=[x for x in rows if x['case']==key];methods=sorted(set(x['method'] for x in rr))
  cases[key]={'geometry':rr[0]['geometry'],'arms':{m:{'ce':float(np.mean([x['service']['ce'] for x in rr if x['method']==m])),'accuracy':float(np.mean([x['service']['accuracy'] for x in rr if x['method']==m]))} for m in methods},'contrasts':{}}
  for a,b in [('raw','phase'),('doppler','phase'),('phase','scalar'),('phase2','scalar'),('phase1','scalar'),('raw','csi')]:
   if a in methods and b in methods:
    cases[key]['contrasts'][a+'-minus-'+b]=bootstrap([next(x['service']['ce'] for x in rr if x['seed']==s and x['method']==a)-next(x['service']['ce'] for x in rr if x['seed']==s and x['method']==b) for s in SEEDS])
 return cases

def freeze(v):
 p=R/v;files=[p/'SPEC.md',pathlib.Path(__file__)]+list((R/'shared/src').glob('*.py'))+list((R/'shared/configs').glob('*.json'))+list((p/'src').glob('*.py'))
 manifest={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files};(p/'freeze_manifest.json').write_text(json.dumps({'time':time.time(),'hashes':manifest},indent=2))
 return p

def main(v):
 p=freeze(v);t=time.time();rows=[]
 check=K.verify();(p/'results/preflight.json').write_text(json.dumps(check,indent=2))
 if v=='V1':cases=[(k,d) for k in ['k20','k50'] for d in [.1,.2,.3,.4,.5]];methods=['raw','phase','scalar','doppler','csi']
 elif v=='V2':cases=[('broad',d) for d in [.1,.2,.3,.4,.5]];methods=['raw','phase','scalar','doppler','csi']
 elif v=='V3':cases=[('broad',d) for d in [.3,.45,.55]];methods=['phase','phase2','phase1','scalar','doppler','csi']
 elif v=='V5':cases=[('isotropic',d) for d in [.35,.4,.45,.55,.7,.85,.9,1.05,1.2,1.35,1.4,1.45]];methods=['raw','phase','csi']
 else:raise ValueError(v)
 for spectrum,d in cases:
  g=geo(spectrum,d)
  for s in SEEDS:
   for m in methods:rows.append(train_geo(s,g,m))
  (p/'results/partial_runs.json').write_text(json.dumps(rows));print(v,spectrum,d,len(rows),round(time.time()-t,2),flush=True)
  # Stop V1 if high-coherence adverse-phase raw reliably learns, prespecified sufficient falsifier.
  if v=='V1' and g['info']['magnitude']>.9 and abs(g['info']['phase_degrees'])>90:
   rr=rows[-8*len(methods):];raw=[x for x in rr if x['method']=='raw'];cc=[x for x in rr if x['method']=='csi'];delta=[a['service']['ce']-b['service']['ce'] for a,b in zip(raw,cc)]
   if sum(x['service']['accuracy']>=.9 for x in raw)>=6 and bootstrap(delta)['ci95'][1]<.02:
    (p/'results/STOP.json').write_text(json.dumps({'reason':'H2 sufficient falsifier','geometry':g['info']}));break
 (p/'results/all_runs.json').write_text(json.dumps(rows,indent=2));(p/'results/summary.json').write_text(json.dumps(summary(rows),indent=2))
 (p/'results/runtime.json').write_text(json.dumps({'host':platform.node(),'pid':os.getpid(),'wall_seconds':time.time()-t,'rows':len(rows)}));print('DONE',v,len(rows),flush=True)
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException:
  (R/sys.argv[1]/'results/failure.txt').write_text(traceback.format_exc());raise
