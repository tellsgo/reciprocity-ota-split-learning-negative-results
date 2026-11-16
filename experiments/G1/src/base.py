"""G31: frozen conditional moving-aperture test. No historical imports."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import pathlib,sys,json,time,hashlib,traceback,platform,resource,signal,contextlib
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import j0
import learning_core as K
R=pathlib.Path(__file__).resolve().parents[1]
C=K.C;NR=C['nr'];SEEDS=C['seeds']
METHODS=['fixed_S0','MA_S0','CSI','MA_polar','fixed_polar']

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def pack(v):return {'real':np.asarray(v).real.tolist(),'imag':np.asarray(v).imag.tolist()}
def stats(v):
    v=np.asarray(v,float);assert v.shape==(8,) and np.isfinite(v).all()
    ix=np.random.default_rng(C['bootstrap_seed']).integers(0,8,(C['bootstrap_replicates'],8))
    return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),min=float(v.min()),max=float(v.max()),per_seed=v.tolist())
def check_hashes():
    manifest=json.loads((R/'freeze_manifest.json').read_text())
    mismatches=[p for p,h in manifest['hashes'].items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=h]
    assert not mismatches,mismatches
    return True

def response(x):
    th=np.arange(128)*2*np.pi/128
    s0=np.exp(-2j*np.pi*np.arange(NR)[:,None]*.5*np.cos(th))/np.sqrt(128)
    return s0,s0*np.exp(2j*np.pi*(.383-x)*np.sin(th))

def geo(x):
    s0,st=response(x);r=s0@s0.conj().T;t=st@s0.conj().T
    def root(m):
        e,v=np.linalg.eigh(m);assert e.min()>-1e-10
        return (v*np.sqrt(np.maximum(e,0)))@v.conj().T
    rt=root(r)
    if x==.383:a=np.eye(NR,dtype=complex);q=np.zeros((NR,NR),complex);qr=q.copy()
    else:
        a=np.linalg.solve(r.T,t.T).T;q=r-a@r@a.conj().T;q=(q+q.conj().T)/2;qr=root(q)
    jr=np.block([[rt,np.zeros((NR,NR))],[a@rt,qr]])
    target=np.block([[r,t.conj().T],[t,r]])
    err=float(np.linalg.norm(jr@jr.conj().T-target));assert err<1e-10
    info=dict(rail_displacement=x,terminal_displacement=.383,residual=.383-x,joint_factor_error=err,
              rho=pack(np.trace(t)/NR),C10_offdiag_fro=float(np.linalg.norm(t-np.diag(np.diag(t)))),
              C10_spectral_norm=float(np.linalg.norm(t,2)),A_singular_values=np.linalg.svd(a,compute_uv=False).tolist(),
              model='conditional shared-angle difference kernel')
    return dict(s0=s0,st=st,r0=r,rt=t,a=a,q=q,jr=jr,root=rt,info=info)

class World(K.World):
    def __init__(self,seed,g):
        rng=np.random.default_rng(C['train_seed_offset']+seed);n=C['rf_observation_quota']
        self.inputs=rng.choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
        both=np.einsum('ij,bjn->bin',g['jr'],K.cn(rng,(n,2*NR,2)))
        self._f=both[:,:NR];self._b=both[:,NR:]
        self._nf=C['forward_noise_std']*K.cn(rng,(n,NR));self._nb=C['reverse_noise_std']*K.cn(rng,(n,2))
        self._np=C['pilot_noise_std']*K.cn(rng,(n,NR,2))
        crng=np.random.default_rng(C['calibration_seed_offset']+seed);k=C['matrix_calibration_pairs']
        pair=np.einsum('ij,bjn->bin',g['jr'],K.cn(crng,(k,2*NR,2)))
        self.cal_fhat=pair[:,:NR]+C['pilot_noise_std']*K.cn(crng,(k,NR,2))
        self.cal_bhat=pair[:,NR:]+C['pilot_noise_std']*K.cn(crng,(k,NR,2))
        trng=np.random.default_rng(C['test_seed_offset']+seed)
        self.test_inputs=trng.choice([-1.,1.],(C['test_samples'],3));self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
        self._test_f=np.einsum('ij,bjn->bin',g['root'],K.cn(trng,(C['test_samples'],NR,2)))
        self._test_n=C['forward_noise_std']*K.cn(trng,(C['test_samples'],NR));self.test_epsilon=K.cn(trng,(C['test_samples'],2))

def estimate(w):
    f,clip=K.qcomplex(w.cal_fhat[:64],C['channel_quant_bound']);b=w.cal_bhat[:64]
    raw=np.einsum('bmn,bln->ml',f,f.conj())/128-.01*np.eye(NR)
    cross=np.einsum('bmn,bln->ml',b,f.conj())/128
    reg=(raw+raw.conj().T)/2+.02*np.eye(NR)
    e,v=np.linalg.eigh(reg);ec=np.maximum(e,.02);creg=K.mm(v*ec,v.conj().T)
    a=K.mm(cross,K.mm(v*(1/ec),v.conj().T))
    u,sv,vh=np.linalg.svd(a.conj().T);m=K.mm(vh.conj().T,u.conj().T)
    unquantized=m.copy();mq,mc=K.qcomplex(m,32.);uq,_,vhq=np.linalg.svd(mq);m=K.mm(uq,vhq)
    info=dict(Ahat=pack(a),Creg=pack(creg),C10hat=pack(cross),M=pack(m),
              Ahat_singular_values=sv.tolist(),channel_clips=clip,matrix_clips=mc,
              M_unitarity_error=float(np.linalg.norm(K.mm(m.conj().T,m)-np.eye(NR))),
              quantization_error_fro=float(np.linalg.norm(m-unquantized)),covariance_floor_count=int((e<.02).sum()))
    assert np.linalg.norm(K.mm(a,creg)-cross)<1e-10
    return a,m,info

@contextlib.contextmanager
def forbid_decompositions():
    names=['svd','eigh','eig','eigvals','eigvalsh','solve','inv','pinv','qr','cholesky','lstsq']
    originals={n:getattr(np.linalg,n) for n in names};calls=[]
    def forbidden(*args,**kwargs):
        calls.append('forbidden');raise AssertionError('S0 called matrix decomposition or solve')
    for n in names:setattr(np.linalg,n,forbidden)
    try:yield calls
    finally:
        for n,f in originals.items():setattr(np.linalg,n,f)

def train(seed,g,method,w):
    if method.endswith('S0'):
        m=np.eye(NR,dtype=complex)
        info=dict(M=pack(m),public_normalizer=16.,no_temporal_statistics_consumed=True)
        K.fit_temporal=lambda *a,**kw:(np.zeros((NR,NR)),m,1.,info)
        with forbid_decompositions() as calls:row=K.train(w,seed,method,'shift_zero',g['a'])
        row['algorithm_matrix_decomposition_or_solve_calls']=len(calls)
        row['channel_correction_matrix_filter_calls']=0
        row['calibration_relative_error']=None
        assert abs(row['restored_noise_variance_per_complex_coordinate']-.2304)<1e-12
    elif method.endswith('polar'):
        ah,m,info=estimate(w)
        K.fit_temporal=lambda *a,**kw:(ah,m,1.,info)
        row=K.train(w,seed,method,'matrix_est',g['a'])
        row['algorithm_matrix_decomposition_or_solve_calls']=dict(calibration_eigh=1,polar_svd=1,decode_svd=1,normalizer_spectral_norm=1)
        row['channel_correction_matrix_filter_calls']=120
    else:
        m=None;row=K.train(w,seed,method,'csi_rf_budget',g['a'])
        row['CSI_definition']='perfect instantaneous F; inherited quantized gradient feedback and noisy training observations'
    row.update(method=method,geometry=g['info'],optimizer='projected Adam')
    row['effective_M']=None if m is None else pack(m)
    row['rho_B']=None if m is None else pack(np.trace(K.mm(K.mm(g['r0'],g['a'].conj().T),m))/NR)
    row['costs'].update(encoder_updates=int(np.ceil(row['costs']['training_labels']/32)),
        calibration_used=method.endswith('polar'),zero_channel_feedback=method.endswith('S0'),mechanical_time_energy_accounted=False)
    expected=(7680,0) if method.endswith('S0') else (8000,78336) if method.endswith('polar') else (7998,170624)
    assert (row['costs']['rf_observations'],row['costs']['control_bits'])==expected
    assert row['gain_quant_clips']==0 and row['gradient_or_loss_quant_clips']==0
    assert row['max_reverse_energy']<=1+1e-12
    return row

def preflight(gs):
    out=K.verify();checks=[]
    for x,g in gs.items():
        ix=np.arange(NR);dist=np.sqrt(((ix[:,None]-ix[None,:])*.5)**2+(.383-x)**2)
        err=float(np.max(abs(g['rt']-j0(2*np.pi*dist))));assert err<1e-12
        block=np.block([[g['r0'],g['rt'].conj().T],[g['rt'],g['r0']]])
        emin=float(np.linalg.eigvalsh(block).min());assert emin>-1e-10
        h=g['a'].conj().T;u,_,vh=np.linalg.svd(h);m=K.mm(vh.conj().T,u.conj().T)
        assert np.linalg.norm(h@m-(h@m).conj().T)<1e-12
        assert np.linalg.eigvalsh(h@m).min()>-1e-12
        checks.append(dict(x=x,bessel_max_error=err,joint_min_eigenvalue=emin,**g['info']))
    w0=World(731888888,gs[0.]);wm=World(731888888,gs[.383])
    assert np.array_equal(wm._f,wm._b)
    fields=['inputs','labels','_f','_nf','_nb','_np','cal_fhat','test_inputs','test_labels','_test_f','_test_n']
    for field in fields:assert np.array_equal(getattr(w0,field),getattr(wm,field)),field
    a,m,info=estimate(wm);assert info['M_unitarity_error']<1e-12
    # Validate smooth field derivative at an interior point; not another training condition.
    x=.2;eps=1e-6;_,st=response(x);th=np.arange(128)*2*np.pi/128
    numeric=(response(x+eps)[1]-response(x-eps)[1])/(2*eps)
    derr=float(np.max(abs(numeric-(-2j*np.pi*np.sin(th))*st)));assert derr<1e-8
    theory=dict(fixed_diagonal_correlation=float(j0(2*np.pi*.383)),conditional_compensated_correlation=1.,
                opposite_direction_endpoint_correlation=float(j0(4*np.pi*.383)),
                opposite_direction_128_sum=pack(np.exp(4j*np.pi*.383*np.sin(th)).mean()),
                logical_status='shared-angle result is conditional; end-to-end rigid translation alone is insufficient with fixed scatterers')
    assert abs(theory['opposite_direction_endpoint_correlation']-theory['opposite_direction_128_sum']['real'])<1e-12
    out.update(geometry=checks,shared_streams=fields,MA_forward_equals_reverse_exact=True,estimated_M_unitarity=info['M_unitarity_error'],continuous_derivative_max_error=derr,theory=theory)
    return out

def decisions(rows):
    by={m:sorted([r for r in rows if r['method']==m],key=lambda r:r['seed']) for m in METHODS}
    def vals(m,key):return np.array([r['service'][key] for r in by[m]])
    out={}
    if len(by['fixed_S0'])==8:
        a=stats(vals('fixed_S0','accuracy'));out['G31.1']=dict(complete_prediction=a['mean']<=.60,falsified=a['mean']>=.75,fixed_accuracy=a)
    if len(by['MA_S0'])==8:
        a=stats(vals('MA_S0','accuracy'));d=stats(vals('MA_S0','accuracy')-vals('fixed_S0','accuracy'));ce=stats(vals('fixed_S0','ce')-vals('MA_S0','ce'))
        out['G31.2']=dict(complete_prediction=a['mean']>=.92 and d['mean']>=.30 and ce['mean']>=.35,
            falsified=a['mean']<.85 or d['mean']<.15,MA_accuracy=a,accuracy_gain=d,CE_gain=ce)
    if len(by['MA_polar'])==8 and len(by['CSI'])==8:
        a=stats(vals('MA_polar','accuracy'));d=stats(vals('MA_polar','ce')-vals('CSI','ce'))
        out['G31.3']=dict(complete_prediction=a['mean']>=.97 and d['mean']<=.05,falsified=d['mean']>.08,MA_accuracy=a,CE_gap_to_CSI=d)
    for key,v in out.items():
        v['status']='archived' if v['falsified'] else 'archived' if v['complete_prediction'] else 'archived，archived'
    return out

def main():
    resource.setrlimit(resource.RLIMIT_CPU,(900,900));signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1800)
    start=time.time();check_hashes();gs={x:geo(x) for x in [0.,.383]}
    dump(R/'results/preflight.json',preflight(gs))
    dump(R/'ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,host=platform.node(),pid=os.getpid(),blas_threads=1))
    rows=[];mats={};events=[];closed=set()
    for x,g in gs.items():
        for field in ['s0','st','r0','rt','a','q','jr']:mats[f'x{x}_{field}']=g[field]
    for method in METHODS:
        x=.383 if method.startswith('MA') else 0.;g=gs[x]
        for seed in SEEDS:
            w=World(seed,g)
            mats[f'x{x}_{seed}_cal_fhat']=w.cal_fhat;mats[f'x{x}_{seed}_cal_bhat']=w.cal_bhat
            row=train(seed,g,method,w);rows.append(row)
            with (R/'results/raw_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print(json.dumps(dict(method=method,seed=seed,ce=row['service']['ce'],accuracy=row['service']['accuracy'],elapsed=time.time()-start)),flush=True)
        for line,decision in decisions(rows).items():
            if line not in closed:
                events.append(dict(line=line,after_method=method,decision=decision,action='close_line_falsified' if decision['falsified'] else 'close_line_completed'))
                closed.add(line)
        dump(R/'results/partial_runs.json',rows);dump(R/'results/stop_events.json',events)
        np.savez_compressed(R/'results/calibration_and_geometry.npz',**mats)
    dump(R/'results/all_runs.json',rows);dump(R/'results/primary.json',decisions(rows))
    dump(R/'results/runtime.json',dict(rows=len(rows),seconds=time.time()-start,pid=os.getpid(),frozen_hashes_unchanged=check_hashes()))
    print('DONE',len(rows),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        dump(R/'results/failure.json',dict(traceback=traceback.format_exc()));raise
