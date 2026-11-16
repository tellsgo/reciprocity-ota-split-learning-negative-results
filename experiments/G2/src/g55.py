"""G55 frozen rigid versus independent jitter experiment. Only local copied modules are imported."""
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
    def __init__(self,seed,bias=.5,sigma=0.,rigid=False):
        n=C['fixed_training_examples'];k=64;ntest=C['test_samples']
        self.inputs=rng(seed,'train_inputs').choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
        self._f=np.einsum('ij,bjn->bin',ROOT,K.cn(rng(seed,'forward_latent'),(n,4,2)))
        self.cf=np.einsum('ij,bjn->bin',ROOT,K.cn(rng(seed,'cal_forward_latent'),(k,4,2)))
        delta=bias
        z=rng(seed,'train_error').normal(size=(n,4))
        self.error=sigma*(np.repeat(z[:,:1],4,axis=1) if rigid else z)
        z=rng(seed,'cal_error').normal(size=(k,4))
        self.cal_error=sigma*(np.repeat(z[:,:1],4,axis=1) if rigid else z)
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
    out=K.verify();rg=rng(555598888,'diagnostics');r=.383-.45*rg.normal(size=(16,4));a,q,c,b=conditional(r)
    th=np.arange(2048)*2*np.pi/2048;s=np.exp(2j*np.pi*np.arange(4)[:,None]*.5*np.cos(th))/np.sqrt(2048)
    st=s[None,:,:]*np.exp(2j*np.pi*r[:,:,None]*np.sin(th))
    ca=np.einsum('bil,jl->bij',st,s.conj());ba=np.einsum('bil,bjl->bij',st,st.conj())
    recon=np.einsum('bij,jk,blk->bil',a,R0,a.conj())+np.einsum('bij,blj->bil',q,q.conj())
    errs=[float(np.max(abs(c-ca))),float(np.max(abs(b-ba))),float(np.max(abs(recon-b)))]
    assert max(errs)<1e-11
    az,qz,_,_=conditional(np.zeros((1,4)));assert np.max(abs(az-np.eye(4)))<1e-12 and np.max(abs(qz))<1e-7
    out.update(angular_cross_error=errs[0],angular_backward_error=errs[1],covariance_reconstruction_error=errs[2],zero_residual_identity=True)
    dump(R/'results/diagnostics.json',out)

def rigid_checks():
    from scipy.special import ndtr
    diagnostics(); rg=rng(555598891,'diagnostics');r=np.repeat(.45*rg.normal(size=(24,1)),4,axis=1)
    a,q,c,b=conditional(r);err=float(np.max(abs(b-R0)))
    assert err==0.
    th=np.arange(2048)*2*np.pi/2048;s=np.exp(2j*np.pi*np.arange(4)[:,None]*.5*np.cos(th))/np.sqrt(2048)
    st=s[None,:,:]*np.exp(2j*np.pi*r[:,:,None]*np.sin(th))
    errs=[float(np.max(abs(c-np.einsum('bil,jl->bij',st,s.conj())))),float(np.max(abs(b-np.einsum('bil,bjl->bij',st,st.conj()))))]
    assert max(errs)<1e-11
    roots=jn_zeros(0,64)/(2*np.pi);out=theory()
    for row in out['rows']:
        if row['kind']=='jitter':
            sig=row['value'];row['probability_instant_J0_negative']=0. if sig==0 else float(2*sum(ndtr(-roots[k]/sig)-ndtr(-roots[k+1]/sig) for k in range(0,64,2)))
            row['omitted_probability_upper_bound']=0. if sig==0 else float(2*ndtr(-roots[-1]/sig))
    out['rigid_checks']=dict(C11_C00_max_abs=err,angular_errors=errs,common_error_exact=True,indep_and_rigid_C10_expectation_equal='same marginal r_i; exact argument, no empirical-fit assertion')
    dump(R/'results/theory.json',out)

def vals(rows,label,key='accuracy'):
    out=sorted([r for r in rows if r['condition']==label],key=lambda r:r['seed'])
    assert len(out)==8 and len(set(r['seed'] for r in out))==8,(label,len(out))
    return np.array([r['service'][key] for r in out])

def batch(rows,label,phase,seeds,bias=.5,sigma=0.,rigid=False):
    for seed in seeds:
        wp=f'results/worlds/{phase}_{label}_{seed}.npz';w=World(seed,bias,sigma,rigid)
        assert not (R/wp).exists(),wp
        np.savez_compressed(R/wp,**vars(w))
        g=mean_geo(bias,sigma)
        if label=='CSI':
            row=K.train(w,seed,'CSI','csi_exact',g['a']);row['CSI_definition']='exact instantaneous F^H delta; 3840 labels / 120 steps; no reverse noise or gradient quantization; ideal free oracle'
            row['costs'].update(rf_observations=3840,current_csi_pilot_observations=0,control_bits=0,digital_gradient_bits=0,oracle_instantaneous_CSI=True,encoder_updates=120)
            for k in list(row['costs']):
                if k.startswith('serial_slots_if_'):del row['costs'][k]
        else:row=B.train(seed,g,'fixed_S0' if label=='fixed_S0' else 'MA_S0',w)
        row.update(condition=label,phase=phase,bias=bias,sigma=sigma,rigid=rigid,world=wp,structural_alias=False)
        row['residual_audit']=dict(error_std=float(w.error.std()),mean=float(w.error.mean()),same_columns=bool(np.array_equal(w.error,np.repeat(w.error[:,:1],4,axis=1))),empirical_rho=float(np.vdot(w._f,w._b).real/w._f.size),instant_negative_fraction=float(np.mean(j0(2*np.pi*w.residual[:,0])<0)))
        if rigid:assert row['residual_audit']['same_columns']
        rows.append(row)
        with (R/f'results/{phase}_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        dump(R/f'results/{phase}_partial.json',rows)
        print(json.dumps(dict(phase=phase,condition=label,seed=seed,**row['service'])),flush=True)

def precheck():
    check('protocol_freeze.json');rigid_checks();rows=[]
    for label in ['fixed_S0','CSI']:batch(rows,label,'precheck',C['precheck_seeds'])
    gap=stats(vals(rows,'fixed_S0','ce')-vals(rows,'CSI','ce'))
    out=dict(passed=gap['mean']>=.10,delta_CE=gap,summary={label:{k:stats(vals(rows,label,k)) for k in ['ce','accuracy']} for label in ['fixed_S0','CSI']},finished=time.time())
    dump(R/'results/precheck.json',out);audit_registry('results/precheck_rng_audit.json');print('PRECHECK',json.dumps(out),flush=True)

def formal():
    check('formal_freeze.json');assert json.loads((R/'results/precheck.json').read_text())['passed']
    rows=[];decisions={};events=[];metrics1={};metrics2={}
    def close(line,fail,support,**info):
        assert line not in decisions
        decisions[line]=dict(falsified=bool(fail),supported=bool(support),status='archived' if fail else 'archived' if support else 'archived，archived',**info)
        events.append(dict(line=line,action='stop_line_falsified' if fail else 'close_line_complete',time=time.time(),rows=len(rows)))
        dump(R/'results/primary.json',decisions);dump(R/'results/stop_events.json',events);print('CLOSE',line,json.dumps(decisions[line]),flush=True)
    for label in ['fixed_S0','CSI']:batch(rows,label,'formal',SEEDS)
    batch(rows,'MA_rigid_jitter_0.0','formal',SEEDS,0.,0.,True)
    for row in list(rows):
        if row['condition']=='MA_rigid_jitter_0.0':
            a=json.loads(json.dumps(row));a.update(condition='MA_indep_jitter_0.0',rigid=False,structural_alias=True,alias_of='MA_rigid_jitter_0.0');rows.append(a)
            with (R/'results/formal_rows.jsonl').open('a') as f:f.write(json.dumps(a,allow_nan=False)+'\n')
    for sig in [.15,.383,.45]:
        if sig==.45 and 'G55.1' in decisions and 'G55.2' in decisions:break
        rigid=f'MA_rigid_jitter_{sig}';indep=f'MA_indep_jitter_{sig}'
        batch(rows,rigid,'formal',SEEDS,0.,sig,True)
        if sig in [.383,.45] and 'G55.1' not in decisions:
            metrics1[str(sig)]=dict(accuracy=stats(vals(rows,rigid)),ce=stats(vals(rows,rigid,'ce')))
            fail=vals(rows,rigid).mean()<.80
            if fail or sig==.45:close('G55.1',fail,len(metrics1)==2 and all(v['accuracy']['mean']>=.88 for v in metrics1.values()),points=metrics1)
        if 'G55.2' not in decisions:
            batch(rows,indep,'formal',SEEDS,0.,sig,False)
            da=vals(rows,rigid)-vals(rows,indep);dc=vals(rows,rigid,'ce')-vals(rows,indep,'ce')
            metrics2[str(sig)]=dict(accuracy_difference=stats(da),ce_difference=stats(dc),mean_seed_absolute_acc_difference=float(abs(da).mean()),mean_seed_absolute_ce_difference=float(abs(dc).mean()))
            fail=abs(da.mean())>.02
            if fail or sig==.45:close('G55.2',fail,len(metrics2)==3 and all(abs(m['accuracy_difference']['mean'])<=.01 and abs(m['ce_difference']['mean'])<=.01 for m in metrics2.values()),points=metrics2)
        if sig==.383:
            batch(rows,'MA_bias_crit','formal',SEEDS,.383,0.,True)
            gain=vals(rows,rigid)-vals(rows,'MA_bias_crit')
            close('G55.3',gain.mean()<.20,gain.mean()>=.30,gain=stats(gain),rigid_accuracy=stats(vals(rows,rigid)),bias_accuracy=stats(vals(rows,'MA_bias_crit')))
    dump(R/'results/all_runs.json',rows);audit_registry('results/formal_rng_audit.json');check('formal_freeze.json')
    print('DONE',len(rows),flush=True)

if __name__=='__main__':
    phase=sys.argv[1];start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(1800)
    dump(R/f'logs/{phase}_environment.json',dict(python=sys.version,executable=sys.executable,numpy=np.__version__,host=platform.node(),pid=os.getpid(),threads=1,phase=phase,start=start))
    try:{'precheck':precheck,'formal':formal}[phase]()
    except BaseException:
        dump(R/f'results/{phase}_failure_{time.time_ns()}.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
    finally:dump(R/f'logs/{phase}_runtime.json',dict(elapsed=time.time()-start,pid=os.getpid(),end=time.time()))
