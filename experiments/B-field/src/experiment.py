"""Frozen U6MIN channel-only experiment. All writes confined to this run."""
import os,sys
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
sys.dont_write_bytecode=True
import pathlib,json,time,hashlib,ast,importlib.util,traceback,signal,platform
import numpy as np
from scipy.special import j0
R=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((R/'config.json').read_text())
D=C['displacement']; SP=C['spacings']; NS=C['samples_per_seed']; ARMS=['max_snr','random','fixed','slide']

def dump(path,x):
    path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))
def cn(rr,shape):return (rr.normal(size=shape)+1j*rr.normal(size=shape))/np.sqrt(2)
def root(a):
    e,v=np.linalg.eigh((a+a.conj().T)/2)
    assert e.min()>-1e-10, e.min()
    return (v*np.sqrt(np.maximum(e,0)))@v.conj().T

def freeze_check():
    m=json.loads((R/'freeze_manifest.json').read_text())
    for p,h in m['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p

def cosine(f,b):
    den=np.linalg.norm(f,axis=-1)*np.linalg.norm(b,axis=-1)
    assert (den>0).all()
    z=np.sum(f.conj()*b,axis=-1).real/den
    assert np.isfinite(z).all() and np.max(abs(z))<1+1e-12
    return z

def select(f,b,sp,random_idx,scalar=False):
    if scalar:f=f[:,:1];b=b[:,:,:1]
    p=D-np.arange(8)*sp;n=len(f)
    idx=np.stack([np.argmax(np.sum(abs(b)**2,axis=2),axis=1),random_idx,np.zeros(n,int),np.full(n,np.argmin(abs(p)),int)],axis=1)
    selected=b[np.arange(n)[:,None],idx]
    cos=cosine(f[:,None,:],selected)
    return idx,cos

def iso_geometry():
    pos=np.unique(np.round(np.concatenate([np.array([0.]),*[D-np.arange(8)*s for s in SP]]),12))
    angle=2*np.pi*np.arange(256)/256
    k=2*np.pi*np.stack([np.cos(angle),np.sin(angle)])
    kernel=np.concatenate([np.exp(1j*pos[:,None]*k[0]),1j*k],axis=0)/16
    cov=kernel@kernel.conj().T;rr=root(cov)
    ct=j0(np.pi*abs(np.arange(4)[:,None]-np.arange(4)[None,:]));rt=root(ct)
    err=float(np.max(abs(cov[:len(pos),:len(pos)]-j0(2*np.pi*abs(pos[:,None]-pos[None,:])))))
    eps=1e-6
    deriv=(np.exp(1j*eps*k)-np.exp(-1j*eps*k))/(2*eps)/16
    de=float(np.max(abs(deriv-kernel[-2:])))
    checks=dict(kernel_j0_error=err,root_error=float(np.max(abs(rr@rr.conj().T-cov))),derivative_fd_error=de,minimum_eigenvalue=float(np.linalg.eigvalsh(cov).min()),derivative_variance=np.diag(cov)[-2:].real.tolist())
    assert err<1e-10 and checks['root_error']<1e-10 and de<1e-8
    np.savez_compressed(R/'results/raw/isotropic_geometry.npz',positions=pos,kernel=kernel,covariance=cov,root=rr,ct=ct,rt=rt)
    return pos,cov,rr,rt,checks

def run_iso():
    pos,cov,rr,rt,checks=iso_geometry();nf=len(pos);origin=int(np.argmin(abs(pos)))
    maps=np.array([[np.argmin(abs(pos-x)) for x in D-np.arange(8)*s] for s in SP])
    for seed in C['isotropic_seeds']:
        z=cn(rng(seed,10),(NS,nf+2,4))
        fields=np.einsum('ij,bjt,ut->biu',rr,z,rt,optimize=True)
        f=fields[:,origin,:];random_idx=rng(seed,20).integers(0,8,NS)
        indices=[];cosines=[];sindices=[];scosines=[]
        for sp,ids in zip(SP,maps):
            b=fields[:,ids,:];idx,cos=select(f,b,sp,random_idx);si,sc=select(f,b,sp,random_idx,True)
            indices.append(idx);cosines.append(cos);sindices.append(si);scosines.append(sc)
        h=f[:,0];dh=fields[:,-2:,0];grad=np.imag(h.conj()[:,None]*dh)/abs(h[:,None])**2;norm=np.linalg.norm(grad,axis=1)
        assert np.isfinite(norm).all()
        emp=np.einsum('bi,bj->ij',fields[:,:,0],fields[:,:,0].conj())/NS
        np.savez_compressed(R/'results/raw'/f'isotropic_{seed}.npz',fields=fields,positions=pos,maps=maps,f=f,random_idx=random_idx,indices=indices,cosines=cosines,scalar_indices=sindices,scalar_cosines=scosines,h=h,dh=dh,amplitude=abs(h),phase_gradient=grad,gradient_norm=norm)
        checks[str(seed)]={'empirical_field_cov_max_error':float(np.max(abs(emp[:nf,:nf]-cov[:nf,:nf]))),'mean_power':float(np.mean(abs(f)**2)),'finite_samples':len(norm)}
        print('ISO',seed,flush=True)
    dump(R/'results/isotropic_checks.json',checks)

def g53_geometry(seed):
    source=(R/'reference/g53/src/experiment.py').read_text();tree=ast.parse(source)
    node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='geometry')
    cfg=json.loads((R/'reference/g53/configs/config.json').read_text())
    env={'np':np,'C':cfg,'PROFILE_ID':1,'rng':lambda s,t:np.random.default_rng(np.random.SeedSequence([s,1,t])),'mm':lambda a,b:np.einsum('ij,jk->ik',a,b,optimize=False)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'copied_g53_geometry','exec'),env)
    return env['geometry'](seed,'A')

def run_cdl():
    checks={};sps=[.2,.5];pos=np.unique(np.round(np.concatenate([np.array([0.]),*[D-np.arange(8)*s for s in sps]]),12));origin=np.argmin(abs(pos))
    maps=np.array([[np.argmin(abs(pos-x)) for x in D-np.arange(8)*s] for s in sps])
    for seed in C['cdl_seeds']:
        g=g53_geometry(seed);at=np.exp(1j*np.pi*np.arange(4)[:,None]*g['txdir'][:,1]);nr=len(g['pr'])
        kernel=np.exp(2j*np.pi*pos[:,None,None]*g['nu'][None,None,:])*at[None,:,:]*np.sqrt(g['pr'])[None,None,:]
        phases=rng(seed,30).uniform(-np.pi,np.pi,(NS,nr));z=np.exp(1j*phases)
        fields=(z@kernel.reshape(-1,nr).T).reshape(NS,len(pos),4);f=fields[:,origin,:]
        direct=np.sum(np.exp(1j*phases[0])*kernel[-1,3]);err=float(abs(direct-fields[0,-1,3]));assert err<1e-12
        assert np.max(abs(at[:2]-g['at']))<1e-12 and nr==460
        random_idx=rng(seed,20).integers(0,8,NS);indices=[];cosines=[];sindices=[];scosines=[]
        for sp,ids in zip(sps,maps):
            idx,co=select(f,fields[:,ids,:],sp,random_idx);si,sc=select(f,fields[:,ids,:],sp,random_idx,True)
            indices.append(idx);cosines.append(co);sindices.append(si);scosines.append(sc)
        cov=kernel[:,0,:]@kernel[:,0,:].conj().T;emp=fields[:,:,0].T@fields[:,:,0].conj()/NS
        checks[str(seed)]={'direct_sum_error':err,'covariance_max_error':float(np.max(abs(cov-emp))),'fixed_rho_real':float(np.sum(g['pr']*np.exp(2j*np.pi*D*g['nu'])).real),'fixed_rho_imag':float(np.sum(g['pr']*np.exp(2j*np.pi*D*g['nu'])).imag)}
        np.savez_compressed(R/'results/raw'/f'cdl_{seed}.npz',fields=fields,phases=phases,positions=pos,maps=maps,f=f,random_idx=random_idx,indices=indices,cosines=cosines,scalar_indices=sindices,scalar_cosines=scosines,kernel=kernel,pr=g['pr'],rxdir=g['rxdir'],txdir=g['txdir'],angles_deg=g['angles_deg'],permutations=g['permutations'],covariance=cov)
        print('CDL',seed,flush=True)
    dump(R/'results/cdl_checks.json',checks)

def run_p4():
    copied=R/'reference/g49';sys.path.insert(0,str(copied/'src'))
    spec=importlib.util.spec_from_file_location('g49_copied',copied/'src/experiment.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    old=R.parent/'20261015_191851_g49';seeds=json.loads((copied/'seeds.json').read_text())['main'];rows=[];sources={}
    for seed in seeds:
        f,b,_=mod.fields(seed,100,7680);back,idx=mod.pick({'f':f,'b':b},'fas_max_snr');co=cosine(f,back)
        paths=[old/'results/worlds'/f'main_{seed}.npz',old/'results/models'/f'main_{seed}_fas_max_snr.npz',old/'results'/f'metric_main_{seed}_fas_max_snr.json']
        for p in paths:sources[str(p.relative_to(R.parent.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(paths[0]) as w:
            fe=float(np.max(abs(w['f']-f)));be=float(np.max(abs(w['b']-b)));labels=w['test_labels']
        with np.load(paths[1]) as model:
            mismatches=int(np.sum(idx!=model['selected_ports']));acc=float(np.mean(model['predictions']==labels))
        metric=json.loads(paths[2].read_text());assert fe<1e-12 and be<1e-12 and mismatches==0 and acc==metric['accuracy']
        row=dict(seed=seed,n=len(f),p_positive=float(np.mean(co>0)),mean_cosine=float(co.mean()),median_cosine=float(np.median(co)),normalized_correlation=float(np.sum(f.conj()*back).real/np.sqrt(np.sum(abs(f)**2)*np.sum(abs(back)**2))),accuracy=acc,trajectory_cosine=metric['trajectory_cosine'],world_f_max_error=fe,world_b_max_error=be,port_mismatches=mismatches)
        rows.append(row);np.savez_compressed(R/'results/raw'/f'p4_{seed}.npz',f=f,b=b,selected_ports=idx,cosines=co)
        print('P4',seed,flush=True)
    dump(R/'results/p4_rows.json',rows);dump(R/'results/p4_input_hashes.json',sources)

def main():
    freeze_check();start=time.time();signal.alarm(1800)
    dump(R/'logs/compute_start.json',dict(time=start,pid=os.getpid(),python=sys.version,numpy=np.__version__,platform=platform.platform(),cpu_only=True))
    run_iso();run_cdl();run_p4();freeze_check()
    dump(R/'logs/compute_completion.json',dict(time=time.time(),wall_seconds=time.time()-start,pid=os.getpid(),complete=True))
if __name__=='__main__':
    try:main()
    except BaseException:
        (R/'logs'/f'failure_{time.time_ns()}.txt').write_text(traceback.format_exc());raise
