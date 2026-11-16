"""G50 frozen bias versus jitter experiment. Only local copied modules are imported."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import sys,pathlib,json,hashlib,time,traceback,platform,resource,signal
sys.dont_write_bytecode=True
import numpy as np
from scipy.special import j0,jn_zeros
from numpy.polynomial.hermite import hermgauss
import base as B
K=B.K; C=K.C; R=pathlib.Path(__file__).resolve().parents[1]; SEEDS=C['seeds']; dump=B.dump
ROLES=['train_inputs','forward_latent','reverse_innovation','forward_noise','reverse_noise','pilot_noise','cal_forward_latent','cal_reverse_innovation','cal_forward_noise','cal_reverse_noise','train_error','cal_error','test_inputs','test_forward_latent','test_forward_noise','test_epsilon','initialization','bootstrap','diagnostics','score']
REGISTRY={}
def rng(seed,role):
    ss=np.random.SeedSequence(C['master_entropy'],spawn_key=(int(seed),ROLES.index(role)))
    key=ss.generate_state(2,dtype=np.uint64)
    REGISTRY[f'{seed}:{role}']=dict(seed=int(seed),role=role,spawn_key=list(ss.spawn_key),key=[int(x) for x in key],first_raw_sha256=hashlib.sha256(np.random.Philox(key=key).random_raw(16).tobytes()).hexdigest())
    return np.random.Generator(np.random.Philox(key=key))
K.stream_rng=rng
def check(name):
    h=json.loads((R/name).read_text())['hashes']
    bad=[p for p,v in h.items() if hashlib.sha256((R/p).read_bytes()).hexdigest()!=v]
    assert not bad,bad
    return True
def audit_registry(name):
    vals=list(REGISTRY.values())
    assert len({tuple(x['key']) for x in vals})==len(vals)
    assert len({x['first_raw_sha256'] for x in vals})==len(vals)
    dump(R/name,dict(passed=True,master_entropy=C['master_entropy'],roles=ROLES,streams=vals,unique_streams=len(vals)))
def stats(v):
    v=np.asarray(v,float);assert v.shape==(8,) and np.isfinite(v).all()
    ix=rng(C['bootstrap_seed'],'bootstrap').integers(0,8,(C['bootstrap_replicates'],8))
    return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),min=float(v.min()),max=float(v.max()),per_seed=v.tolist())
def arrayroot(m):
    e,v=np.linalg.eigh(m);assert e.min()>-1e-9,float(e.min())
    return np.einsum('...ik,...jk->...ij',v*np.sqrt(np.maximum(e,0))[...,None,:],v.conj())
D2=(.5*(np.arange(4)[:,None]-np.arange(4)[None,:]))**2
R0=j0(2*np.pi*np.sqrt(D2)).astype(complex); ROOT=arrayroot(R0); RINV=np.linalg.inv(R0)
def conditional(residual):
    r=np.asarray(residual)
    cross=j0(2*np.pi*np.sqrt(D2+r[..., :,None]**2)).astype(complex)
    back=j0(2*np.pi*np.sqrt(D2+(r[..., :,None]-r[...,None,:])**2)).astype(complex)
    a=np.einsum('...ij,jk->...ik',cross,RINV)
    q=back-np.einsum('...ij,jk,...lk->...il',a,R0,a.conj());q=(q+q.conj().swapaxes(-2,-1))/2
    return a,arrayroot(q),cross,back
def mean_geo(delta,sigma):
    nodes,weights=hermgauss(100)
    residual=delta-np.sqrt(2)*sigma*nodes
    cross=np.einsum('k,kij->ij',weights/np.sqrt(np.pi),j0(2*np.pi*np.sqrt(D2+residual[:,None,None]**2))).astype(complex)
    return dict(r0=R0,a=np.einsum('ij,jk->ik',cross,RINV),info=dict(terminal_displacement=.5,mean_residual=delta,sigma=sigma,mean_diagonal_rho=float(np.trace(cross).real/4),model='conditional shared-angle isotropic Gaussian field; four independent rail errors'))
class World(K.World):
    def __init__(self,seed,bias=.5,sigma=0.):
        n=C['fixed_training_examples'];k=64;ntest=C['test_samples']
        self.inputs=rng(seed,'train_inputs').choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
        self._f=np.einsum('ij,bjn->bin',ROOT,K.cn(rng(seed,'forward_latent'),(n,4,2)))
        self.cf=np.einsum('ij,bjn->bin',ROOT,K.cn(rng(seed,'cal_forward_latent'),(k,4,2)))
        delta=bias
        self.error=sigma*rng(seed,'train_error').normal(size=(n,4))
        self.cal_error=sigma*rng(seed,'cal_error').normal(size=(k,4))
        self.residual=delta-self.error;self.cal_residual=delta-self.cal_error
        a,q,_,_=conditional(self.residual);ac,qc,_,_=conditional(self.cal_residual)
        if bias==0. and sigma==0.:
            a[:]=np.eye(4);q[:]=0.;ac[:]=np.eye(4);qc[:]=0.
        self._b=np.einsum('bij,bjn->bin',a,self._f)+np.einsum('bij,bjn->bin',q,K.cn(rng(seed,'reverse_innovation'),(n,4,2)))
        self.cb=np.einsum('bij,bjn->bin',ac,self.cf)+np.einsum('bij,bjn->bin',qc,K.cn(rng(seed,'cal_reverse_innovation'),(k,4,2)))
        self._nf=.1*K.cn(rng(seed,'forward_noise'),(n,4));self._nb=.03*K.cn(rng(seed,'reverse_noise'),(n,2));self._np=.1*K.cn(rng(seed,'pilot_noise'),(n,4,2))
        self.cal_fhat=self.cf+.1*K.cn(rng(seed,'cal_forward_noise'),(k,4,2));self.cal_bhat=self.cb+.1*K.cn(rng(seed,'cal_reverse_noise'),(k,4,2))
        self.test_inputs=rng(seed,'test_inputs').choice([-1.,1.],(ntest,3));self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
        self._test_f=np.einsum('ij,bjn->bin',ROOT,K.cn(rng(seed,'test_forward_latent'),(ntest,4,2)))
        self._test_n=.1*K.cn(rng(seed,'test_forward_noise'),(ntest,4));self.test_epsilon=K.cn(rng(seed,'test_epsilon'),(ntest,2))
def theory():
    from scipy.special import i0e
    rows=[];th=np.arange(4096)*2*np.pi/4096
    for kind,grid in [('bias',C['bias_grid']),('jitter',C['jitter_grid'])]:
        for v in grid:
            bias=v if kind=='bias' else 0.;sigma=v if kind=='jitter' else 0.
            g=mean_geo(bias,sigma);rho=g['info']['mean_diagonal_rho']
            analytic=float(j0(2*np.pi*bias)) if kind=='bias' else float(i0e(np.pi**2*sigma**2))
            angular=float(np.mean(np.cos(2*np.pi*bias*np.sin(th))*np.exp(-2*np.pi**2*sigma**2*np.sin(th)**2)))
            cross=np.einsum('ij,jk->ik',g['a'],R0)
            assert abs(rho-analytic)<1e-12 and abs(rho-angular)<1e-12
            rows.append(dict(kind=kind,value=v,rho=rho,analytic=analytic,angular=angular,integration_abs_error=abs(rho-analytic),C10=B.pack(cross),A=B.pack(g['a']),C10_eigenvalues=np.linalg.eigvalsh(cross).tolist(),A_singular_values=np.linalg.svd(g['a'],compute_uv=False).tolist(),offdiagonal_fro=float(np.linalg.norm(cross-np.diag(np.diag(cross))))))
    return dict(rows=rows,first_zero_lambda=float(jn_zeros(0,1)[0]/(2*np.pi)))
def diagnostics():
    out=K.verify();rg=rng(505018888,'diagnostics');r=.383-.45*rg.normal(size=(16,4));a,q,c,b=conditional(r)
    th=np.arange(2048)*2*np.pi/2048;s=np.exp(2j*np.pi*np.arange(4)[:,None]*.5*np.cos(th))/np.sqrt(2048)
    st=s[None,:,:]*np.exp(2j*np.pi*r[:,:,None]*np.sin(th))
    ca=np.einsum('bil,jl->bij',st,s.conj());ba=np.einsum('bil,bjl->bij',st,st.conj())
    recon=np.einsum('bij,jk,blk->bil',a,R0,a.conj())+np.einsum('bij,blj->bil',q,q.conj())
    errs=[float(np.max(abs(c-ca))),float(np.max(abs(b-ba))),float(np.max(abs(recon-b)))]
    assert max(errs)<1e-11
    az,qz,_,_=conditional(np.zeros((1,4)));assert np.max(abs(az-np.eye(4)))<1e-12 and np.max(abs(qz))<1e-7
    out.update(angular_cross_error=errs[0],angular_backward_error=errs[1],covariance_reconstruction_error=errs[2],zero_residual_identity=True)
    dump(R/'results/diagnostics.json',out)
def vals(rows,label,key='accuracy'):
    out=sorted([r for r in rows if r['condition']==label],key=lambda r:r['seed'])
    assert [r['seed'] for r in out]==SEEDS,(label,len(out))
    return np.array([r['service'][key] for r in out])
def save_row(rows,row):
    rows.append(row)
    with (R/'results/raw_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    dump(R/'results/partial_runs.json',rows)
    print(json.dumps(dict(condition=row['condition'],seed=row['seed'],accuracy=row['service']['accuracy'],ce=row['service']['ce'],alias=row.get('structural_alias',False))),flush=True)
def batch(rows,label,method,bias=.5,sigma=0.):
    for seed in SEEDS:
        worldpath=f'results/worlds/b{bias}_s{sigma}_{seed}.npz';w=World(seed,bias,sigma)
        if not (R/worldpath).exists():np.savez_compressed(R/worldpath,**vars(w))
        g=mean_geo(bias,sigma)
        row=B.train(seed,g,method,w);row.update(condition=label,bias=bias,sigma=sigma,world=worldpath,structural_alias=False)
        row['residual_audit']=dict(mean=float(w.residual.mean()),std=float(w.residual.std(ddof=1)),error_mean=float(w.error.mean()),error_std=float(w.error.std(ddof=1)),conditional_mean_rho=g['info']['mean_diagonal_rho'],empirical_diagonal_rho=float(np.vdot(w._f,w._b).real/w._f.size))
        save_row(rows,row)
def alias(rows,source,target):
    for row in [r for r in rows if r['condition']==source]:
        out=json.loads(json.dumps(row));out.update(condition=target,structural_alias=True,alias_of=source);save_row(rows,out)
def precheck():
    check('protocol_freeze.json');assert not (R/'results/raw_rows.jsonl').exists()
    start=time.time();dump(R/'results/theory.json',theory());diagnostics();rows=[]
    batch(rows,'fixed_S0','fixed_S0');batch(rows,'CSI','CSI')
    gap=stats(vals(rows,'fixed_S0','ce')-vals(rows,'CSI','ce'))
    out=dict(passed=gap['mean']>=.10,delta_CE=gap,fixed_S0=dict(accuracy=stats(vals(rows,'fixed_S0')),ce=stats(vals(rows,'fixed_S0','ce'))),CSI=dict(accuracy=stats(vals(rows,'CSI')),ce=stats(vals(rows,'CSI','ce'))),training_rows=16,time=time.time(),elapsed=time.time()-start)
    dump(R/'results/precheck.json',out);audit_registry('results/precheck_rng_audit.json');print('PRECHECK',json.dumps(out),flush=True)
def formal():
    check('formal_freeze.json');assert json.loads((R/'results/precheck.json').read_text())['passed']
    rows=json.loads((R/'results/partial_runs.json').read_text());assert len(rows)==16
    events=[];decisions={};start=time.time()
    def close(line,falsified,supported,**metrics):
        status='archived' if falsified else 'archived' if supported else 'archived，archived'
        decisions[line]=dict(status=status,falsified=bool(falsified),supported=bool(supported),**metrics)
        events.append(dict(line=line,rows_at_close=len(rows),time=time.time(),action='stop_line_falsified' if falsified else 'close_line_completed',decision=decisions[line]))
        dump(R/'results/primary.json',decisions);dump(R/'results/stop_events.json',events);print('CLOSE',line,status,flush=True)
    batch(rows,'MA_bias_0.0','MA_S0',0.,0.)
    batch(rows,'MA_bias_0.15','MA_S0',.15,0.)
    batch(rows,'MA_bias_0.383','MA_S0',.383,0.)
    ref=vals(rows,'MA_bias_0.0');crit=vals(rows,'MA_bias_0.383');fail=crit.mean()>=.75
    if not fail:alias(rows,'fixed_S0','MA_bias_0.5')
    close('G50.1',fail,crit.mean()<=.65 and (ref-crit).mean()>=.25,accuracy=stats(crit),drop_from_zero=stats(ref-crit),bias_05='not_added_stop_rule' if fail else 'exact_structural_alias_of_fixed_S0')
    alias(rows,'MA_bias_0.0','MA_jitter_0.0')
    batch(rows,'MA_jitter_0.15','MA_S0',0.,.15)
    jm={};fail=False
    for sigma in [.383,.45]:
        label=f'MA_jitter_{sigma}';batch(rows,label,'MA_S0',0.,sigma);v=vals(rows,label)
        jm[str(sigma)]=dict(accuracy=stats(v),gain_vs_fixed_bias_0383=stats(v-crit))
        if v.mean()<.75:fail=True;break
    supported=(not fail) and len(jm)==2 and all(m['accuracy']['mean']>=.85 for m in jm.values()) and jm['0.383']['gain_vs_fixed_bias_0383']['mean']>=.20
    close('G50.2',fail,supported,jitter=jm,unrun_sigma_points=[v for v in [.383,.45] if str(v) not in jm])
    batch(rows,'MA_polar_crit','MA_polar',.383,0.)
    polar=vals(rows,'MA_polar_crit');gain=polar-crit
    close('G50.3',polar.mean()<.80 or gain.mean()<.10,polar.mean()>=.88 and gain.mean()>=.20,accuracy=stats(polar),gain_vs_bias=stats(gain))
    dump(R/'results/all_runs.json',rows);audit_registry('results/formal_rng_audit.json')
    dump(R/'results/runtime.json',dict(seconds=time.time()-start,rows=len(rows),new_training_fits=sum(not r['structural_alias'] for r in rows),frozen_hashes_unchanged=check('formal_freeze.json')))
    print('DONE',len(rows),flush=True)
if __name__=='__main__':
    phase=sys.argv[1];resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1800)
    dump(R/f'logs/{phase}_environment.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,host=platform.node(),pid=os.getpid(),threads=1,phase=phase))
    try:
        if phase=='precheck':precheck()
        elif phase=='formal':formal()
        else:raise ValueError(phase)
    except BaseException:
        dump(R/f'results/{phase}_failure.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
