"""Frozen G16 conditional single-mode spherical-wave experiment."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import sys,pathlib,json,time,hashlib,traceback,platform,resource,signal
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1]
os.environ['TMPDIR']=str(R/'tmp');os.environ['MPLCONFIGDIR']=str(R/'figures/cache')
sys.path.insert(0,str(R/'shared/src'))
import learning_core as K
import numpy as np
from scipy.special import j0
C=K.C;N=C['nr'];SEEDS=C['seeds'];METHODS=['farfield_polar','nearfield_polar','CSI','S0'];mm=K.mm

def dump(path,x):(R/path).write_text(json.dumps(x,indent=2,allow_nan=False,ensure_ascii=False))
def pack(x):
 x=np.asarray(x);return {'real':x.real.tolist(),'imag':x.imag.tolist()}
def unpack(x):return np.asarray(x['real'])+1j*np.asarray(x['imag'])
def stats(v):
 v=np.asarray(v,float);assert v.shape==(8,) and np.isfinite(v).all()
 ix=np.random.default_rng(C['bootstrap_seed']).integers(0,8,(20000,8))
 return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),min=float(v.min()),max=float(v.max()),per_seed=v.tolist())
def vals(rows,m,key='ce'):return np.array([next(r['service'][key] for r in rows if r['seed']==s and r['method']==m) for s in SEEDS])
def geometry():
 x=(np.arange(N)-(N-1)/2)*.5;theta=np.pi/3;r0=3.;r1=3.383;d=.383
 dist=lambda r:np.sqrt(r*r+x*x-2*r*x*np.sin(theta))
 rn0,rn1=dist(r0),dist(r1);u0=np.exp(-2j*np.pi*rn0);u1=np.exp(-2j*np.pi*rn1);ff=np.exp(2j*np.pi*x*np.sin(theta))
 phi=2*np.pi*np.arange(128)/128;rho=np.mean(np.exp(-2j*np.pi*d*np.cos(phi)))
 c00=np.outer(u0,u0.conj());c11=np.outer(u1,u1.conj());c10=rho*np.outer(u1,u0.conj());a=c10/N
 joint=np.block([[c00,c10.conj().T],[c10,c11]])
 dn=np.diag(u1/u0);df=np.exp(-2j*np.pi*d)*np.eye(N)
 geo=dict(x=x,rn0=rn0,rn1=rn1,u0=u0,u1=u1,aNF0=u0/np.sqrt(N),aNF1=u1/np.sqrt(N),aFF=ff/np.sqrt(N),rho=rho,c00=c00,c11=c11,c10=c10,a=a,joint=joint,D_NF=dn,D_FF=df)
 cart=lambda r:np.sqrt((r*np.sin(theta)-x)**2+(r*np.cos(theta))**2)
 e=float(max(np.max(abs(cart(r0)-rn0)),np.max(abs(cart(r1)-rn1))))
 eig=np.linalg.eigvalsh(joint);bessel=float(abs(rho-j0(2*np.pi*d)))
 assert e<1e-13 and eig.min()>-1e-12 and bessel<1e-13
 assert np.linalg.norm(mm(dn,u0[:,None])[:,0]-u1)<1e-13
 assert np.linalg.norm(mm(a,c00)-c10)<1e-12
 info=dict(model_scope='single_spatial_mode_with_scalar_isotropic_temporal_fading',nr=N,nt=2,r0=r0,r1=r1,theta_deg=60,delta_r=d,aperture=3.5,rayleigh_distance=24.5,x=x.tolist(),distances_initial=rn0.tolist(),distances_return=rn1.tolist(),radial_distance_rates=((r0-x*np.sin(theta))/rn0).tolist(),path_displacement=(rn1-rn0).tolist(),rho=pack(rho),rho_abs=float(abs(rho)),J0=float(j0(2*np.pi*d)),bessel_error=bessel,cartesian_distance_error=e,joint_eigenvalues=eig.tolist(),C00_rank=int(np.linalg.matrix_rank(c00)),A_singular_values=np.linalg.svd(a,compute_uv=False).tolist(),initial_NF_FF_power_overlap=float(abs(np.vdot(ff,u0)/N)**2),radial_NF_FF_power_overlap=float(abs(np.vdot(u1,mm(df,u0[:,None])[:,0])/N)**2),unitary_coherence_upper_bound=float(abs(rho)),geometry_aligned_effective_coherence=pack(np.trace(mm(c10.conj().T,dn))/N))
 return geo,info

class World(K.World):
 def __init__(self,seed,g):
  def pair(rng,n):
   h0=K.cn(rng,(n,2));z=K.cn(rng,(n,2));h1=g['rho']*h0+np.sqrt(1-abs(g['rho'])**2)*z
   return g['u0'][None,:,None]*h0[:,None,:],g['u1'][None,:,None]*h1[:,None,:]
  rng=np.random.default_rng(seed+C['train_seed_offset']);n=C['rf_observation_quota']
  self.inputs=rng.choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
  self._f,self._b=pair(rng,n);self._nf=.1*K.cn(rng,(n,N));self._nb=.03*K.cn(rng,(n,2));self._np=.1*K.cn(rng,(n,N,2))
  crng=np.random.default_rng(seed+C['calibration_seed_offset']);cf,cb=pair(crng,64)
  self.cal_fhat=cf+.1*K.cn(crng,(64,N,2));self.cal_bhat=cb+.1*K.cn(crng,(64,N,2))
  trng=np.random.default_rng(seed+C['test_seed_offset']);self.test_inputs=trng.choice([-1.,1.],(8192,3));self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
  self._test_f=g['u0'][None,:,None]*K.cn(trng,(8192,2))[:,None,:];self._test_n=.1*K.cn(trng,(8192,N));self.test_epsilon=K.cn(trng,(8192,2))
 def save(self,seed):np.savez_compressed(R/f'results/world_{seed}.npz',**{k:v for k,v in vars(self).items() if isinstance(v,np.ndarray)})

def estimate(w,d):
 f,clips=K.qcomplex(w.cal_fhat,4.)
 cross=np.einsum('bmn,bln->ml',w.cal_bhat,f.conj())/128
 raw=np.einsum('bmn,bln->ml',f,f.conj())/128-.01*np.eye(N)
 reg=(raw+raw.conj().T)/2+.02*np.eye(N);e,v=np.linalg.eigh(reg);dc=np.maximum(e,.02)
 creg=np.einsum('ik,k,jk->ij',v,dc,v.conj())
 gamma=np.trace(mm(d.conj().T,cross))/np.trace(creg);a=gamma*d
 if abs(gamma)==0:exact=d.copy()
 else:
  u,s,vh=np.linalg.svd(a);exact=mm(u,vh)
  assert np.linalg.norm(exact-np.exp(1j*np.angle(gamma))*d)<1e-11
 quant,mc=K.qcomplex(exact,32.);uq,_,vhq=np.linalg.svd(quant);m=mm(uq,vhq)
 err=float(np.linalg.norm(mm(m.conj().T,m)-np.eye(N)));assert err<1e-11
 info=dict(gamma=pack(gamma),A_model=pack(a),D=pack(d),M=pack(m),C10hat=pack(cross),Creg=pack(creg),channel_clips=clips,matrix_clips=mc,covariance_floor_count=int((e<.02).sum()),unitarity_error=err,quantization_error_fro=float(np.linalg.norm(m-exact)),formula='A_model=gamma D; svd(A_model)=U S Vh; M=U Vh; quantize/reproject',nullspace_completion='fixed geometric D')
 return a,m,info

def preflight(g):
 out=K.verify();rng=np.random.default_rng(1616888888);n=32768
 h0=K.cn(rng,(n,));z=K.cn(rng,(n,));h1=g['rho']*h0+np.sqrt(1-abs(g['rho'])**2)*z
 both=np.c_[h0[:,None]*g['u0'],h1[:,None]*g['u1']]
 empir=np.einsum('bi,bj->ij',both,both.conj())/n
 errors=abs(empir-g['joint']);zmax=float(errors.max()*np.sqrt(n));assert zmax<7
 # Exact no-fading identity: B=D_NF F, hence B^H D_NF s = F^H s.
 f=g['u0'][None,:,None]*np.array([[1+.3j,.4-.2j]])[:,None,:]
 b=g['u1'][None,:,None]*np.array([[1+.3j,.4-.2j]])[:,None,:]
 s=np.array([np.arange(N)+1j*np.arange(N)[::-1]])
 err=float(np.linalg.norm(K.pb(b,K.mv(g['D_NF'],s))-K.pb(f,s)));assert err<1e-12
 out.update(empirical_joint_covariance_max_complex_z=zmax,empirical_covariance_n=n,adjoint_direction_error=err,receiver_dimension=N,steering_unit_norm_error=float(abs(np.linalg.norm(g['aNF0'])-1)))
 np.savez_compressed(R/'results/covariance_probe.npz',empirical=empir,target=g['joint'],standard_error=np.ones((2*N,2*N))/np.sqrt(n))
 return out

def train(seed,g,method,w):
 if method=='S0':a=g['a'];m=np.eye(N);info={};internal='raw'
 elif method=='CSI':a=g['a'];m=np.eye(N);info={};internal='csi_ideal'
 else:
  a,m,info=estimate(w,g['D_NF'] if method=='nearfield_polar' else g['D_FF']);internal='matrix_est'
 K.fit_temporal=lambda *args,**kw:(a,m,1.,info)
 row=K.train(w,seed,'NF/r3/theta60/dr0.383/single_mode',internal,g['a']);row['method']=method
 row['effective_coherence']=None if method=='CSI' else pack(np.trace(mm(g['c10'].conj().T,m))/N)
 row['actual_effective_M']=None if method=='CSI' else pack(m)
 co=row['costs'];co.update(encoder_updates=int(np.ceil(co['training_labels']/32)),test_forward_evaluations=8192,model_scope='conditional_single_mode',geometry_side_information=method in ['farfield_polar','nearfield_polar'])
 if method=='CSI':
  assert co['rf_observations']==7998 and co['training_labels']==2666
  co.update(oracle_statistics=True,ideal_exact_current_CSI=True,reference_noisy_CSI_schedule_rf=7998,reference_noisy_CSI_control_bits=170624,reference_pilot_observations=5332,current_csi_pilot_observations=0,rf_observations=2666,digital_gradient_bits=None,control_bits=None,oracle_acquisition_cost='undefined; not a deployable finite-bit baseline')
  for key in list(co):
   if key.startswith('serial_slots_if_'):del co[key]
 else:
  expected=(7360,58880) if method=='S0' else (8000,93696)
  assert (co['rf_observations'],co['control_bits'])==expected
  assert co['training_labels']==3680 and co['encoder_updates']==115
 assert row['gain_quant_clips']==0 and row['max_reverse_energy']<=1+1e-12
 return row

def decisions(rows):
 have={m for m in METHODS if sum(r['method']==m for r in rows)==8};out={}
 if 'farfield_polar' in have:
  acc=stats(vals(rows,'farfield_polar','accuracy'))
  out['G16.1']=dict(full=bool(.50<=acc['mean']<=.80),falsified=bool(acc['mean']>=.88),farfield_accuracy=acc)
 if {'farfield_polar','nearfield_polar'}<=have:
  acc=stats(vals(rows,'nearfield_polar','accuracy'));gap=stats(vals(rows,'nearfield_polar','accuracy')-vals(rows,'farfield_polar','accuracy'));gain=stats(vals(rows,'farfield_polar')-vals(rows,'nearfield_polar'))
  out['G16.2']=dict(full=bool(acc['mean']>=.95 and gap['mean']>=.15 and gain['mean']>=.25),falsified=bool(acc['mean']<.85 or gap['mean']<.08),nearfield_accuracy=acc,NF_minus_FF_accuracy=gap,FF_minus_NF_CE=gain)
 if {'nearfield_polar','CSI'}<=have:
  acc=stats(vals(rows,'nearfield_polar','accuracy'));gap=stats(vals(rows,'nearfield_polar')-vals(rows,'CSI'))
  out['G16.3']=dict(full=bool(acc['mean']>=.95 and gap['mean']<=.05),falsified=bool(gap['mean']>.08),nearfield_accuracy=acc,NF_minus_CSI_CE=gap)
 for v in out.values():
  v['status']='rejected' if v['falsified'] else ('numerical conditions met' if v['full'] else 'not supported')
  v['scope']='Specified model only; mechanism not independently established.'
 return out

def main():
 start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900));signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit 1800s')));signal.alarm(1800)
 g,info=geometry();dump('results/geometry.json',info);np.savez_compressed(R/'results/geometry.npz',**g);dump('results/preflight.json',preflight(g))
 worlds={s:World(s,g) for s in SEEDS}
 for s,w in worlds.items():w.save(s)
 rows=[];events=[];seen=set()
 for method in METHODS:
  for seed in SEEDS:
   row=train(seed,g,method,worlds[seed]);rows.append(row);dump('results/partial_runs.json',rows)
   print(json.dumps(dict(method=method,seed=seed,ce=row['service']['ce'],accuracy=row['service']['accuracy'],rows=len(rows),wall_seconds=time.time()-start)),flush=True)
  claims=decisions(rows)
  for name,v in claims.items():
   if name not in seen:
    seen.add(name);events.append(dict(claim=name,batch=method,rows=len(rows),wall_seconds=time.time()-start,triggered=v['falsified'],decision=v,action='close_line_no_new_experiments' if v['falsified'] else 'fixed_endpoint_complete'))
  dump('results/claims.json',claims);dump('results/stop_events.json',events)
  print(json.dumps({'batch_complete':method,'claims':{k:v['status'] for k,v in claims.items()}},ensure_ascii=False),flush=True)
 dump('results/all_runs.json',rows);usage=resource.getrusage(resource.RUSAGE_SELF)
 dump('results/runtime.json',dict(pid=os.getpid(),host=platform.node(),python=sys.version,numpy=np.__version__,wall_seconds=time.time()-start,cpu_seconds=usage.ru_utime+usage.ru_stime,rows=len(rows)))
 print('DONE 32',flush=True)
if __name__=='__main__':
 try:main()
 except BaseException:dump('results/failure.json',dict(traceback=traceback.format_exc()));raise
