"""Two-stage preregistered finite hexagonal-array translation experiment."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, time, hashlib, itertools, traceback, platform
from pathlib import Path
import numpy as np
from scipy.special import j0
from scipy.optimize import linear_sum_assignment
import base as B
import learning_core as K
R=B.R; C=K.C; N=7
dump=B.dump; mm=B.mm; pack=B.pack
PERMS=np.array(list(itertools.permutations(range(N))))

def verify_freeze(name):
    f=json.loads((R/name).read_text())
    for p,h in f['hashes'].items():
        assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p

def geometry(d,angle,save=True):
    phi=np.arange(6)*np.pi/3
    pos=np.vstack([np.zeros((1,2)),.5*np.array([np.cos(phi),np.sin(phi)]).T])
    t=np.arange(128)*2*np.pi/128; uv=np.array([np.cos(t),np.sin(t)])
    dv=d*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle))])
    s0=np.exp(mm(-2j*np.pi*pos,uv))/np.sqrt(128)
    st=s0*np.exp(2j*np.pi*np.einsum('i,ik->k',dv,uv))
    r=mm(s0,s0.conj().T); cross=mm(st,s0.conj().T)
    a=np.linalg.solve(r.T,cross.T).T
    q=r-mm(mm(a,r),a.conj().T); q=(q+q.conj().T)/2
    def root(x):
        e,v=np.linalg.eigh(x); assert e.min()>-1e-10
        return mm(v*np.sqrt(np.maximum(e,0)),v.conj().T)
    rt=root(r); qr=root(q)
    jr=np.block([[rt,np.zeros((N,N))],[mm(a,rt),qr]])
    residual=pos[:,None,:]-pos[None,:,:]-dv
    distances=np.linalg.norm(residual,axis=2)
    overlap=(distances<1e-10).astype(float)
    rho=np.sum(cross[PERMS,np.arange(N)].real,axis=1)/N
    ix=np.flatnonzero(rho>=rho.max()-1e-12)[0]
    perm=PERMS[ix]; m=np.eye(N)[perm].T
    row,col=linear_sum_assignment(-cross.real)
    assert abs(rho.max()-cross.real[row,col].sum()/N)<1e-12
    joint=np.block([[r,cross.conj().T],[cross,r]])
    e0=float(np.max(abs(r-j0(2*np.pi*np.linalg.norm(pos[:,None]-pos[None,:],axis=2)))))
    e1=float(np.max(abs(cross-j0(2*np.pi*distances))))
    ej=float(np.linalg.norm(mm(jr,jr.conj().T)-joint))
    assert max(e0,e1,ej)<1e-10
    assert np.linalg.eigvalsh(joint).min()>-1e-10
    assert np.linalg.norm(mm(m.T,m)-np.eye(N))<1e-12
    u,s,vh=np.linalg.svd(a); polar=mm(u,vh); herm=mm(a.conj().T,polar)
    assert np.linalg.norm(herm-herm.conj().T)<1e-10 and np.linalg.eigvalsh(herm).min()>-1e-10
    invroot=np.linalg.inv(rt)
    residual_by_perm=np.sum(distances[PERMS,np.arange(N)]**2,axis=1)
    if save:
        tag=f'd{d:.2f}_a{angle}'
        np.savetxt(R/f'results/permutations_{tag}.csv',np.column_stack([PERMS,rho,residual_by_perm]),delimiter=',',header='p0,p1,p2,p3,p4,p5,p6,rho_B,squared_residual_sum',comments='')
    # Independent paired channels at the same test geometry; diagnostic only.
    rng=np.random.default_rng(737888888+angle)
    pair=np.einsum('ij,bjn->bin',jr,K.cn(rng,(8192,2*N,2)))
    empirical=np.einsum('bin,bjn->ij',pair[:,N:],pair[:,:N].conj())/(8192*2)
    g=dict(pos=pos,s0=s0,st=st,r0=r,rt=cross,a=a,q=q,root=rt,qr=qr,jr=jr,M=m,T=overlap,dv=dv)
    info=dict(d=d,angle=angle,rho_S0=float(np.trace(cross).real/N),rho_Sperm=float(rho[ix]),rho_partial=float(np.trace(mm(cross.conj().T,overlap)).real/N),
        rho_S0_empirical=float(np.trace(empirical).real/N),rho_Sperm_empirical=float(np.trace(mm(empirical.conj().T,m)).real/N),
        sigma_max_C00=float(np.linalg.svd(r,compute_uv=False)[0]),sigma_max_C10=float(np.linalg.svd(cross,compute_uv=False)[0]),
        singular_values_A=s.tolist(),canonical_correlations=np.linalg.svd(mm(mm(invroot,cross),invroot),compute_uv=False).tolist(),
        overlap_count=int(overlap.sum()),partial_rank=int(np.linalg.matrix_rank(overlap)),selected_permutation=perm.tolist(),
        min_all_permutations_squared_residual=float(residual_by_perm.min()),selected_squared_residual=float(residual_by_perm[ix]),
        selected_steering_error=float(np.linalg.norm(st-mm(m,s0))),C00_bessel_error=e0,C10_bessel_error=e1,joint_factor_error=ej,
        Q_eigenvalues=np.linalg.eigvalsh(q).tolist(),all_5040_assignments_enumerated=True,hungarian_check=True)
    if save: np.savez_compressed(R/f'results/geometry_{tag}.npz',**g)
    return g,info

def train(seed,g,method,angle,d,phase):
    w=B.World(seed,g)
    m=None
    if method in ['S0','S_hex','S_perm']:
        m=np.eye(N) if method=='S0' else g['M']
        info={'M':pack(m),'fixed_public_normalizer':float(8*np.sqrt(N)),'geometry_only':True}
        K.fit_temporal=lambda *a,**kw:(np.zeros((N,N)),m,1.,info)
        internal='shift_zero'
    elif method=='polar_64':
        aa,m,alpha,info=B.estimate(w)
        K.fit_temporal=lambda *a,**kw:(aa,m,alpha,info)
        internal='matrix_est'
    else: internal='csi_rf_budget'
    r=K.train(w,seed,f'hex7/d{d}/theta{angle}',internal,g['a'])
    r.update(method=method,angle=angle,d=d,phase=phase,actual_effective_M=None if m is None else pack(m))
    if method in ['S0','S_hex','S_perm']: r['calibration_relative_error']=None
    if method=='CSI':
        co=r['costs']; co['nominal_inherited_digital_gradient_bits']=co['digital_gradient_bits']
        co['digital_gradient_bits']=None; co['control_bits']=None; co['ideal_instantaneous_channel']=True
        for k in list(co):
            if k.startswith('serial_slots_'): del co[k]
    r['noiseless_negative_fraction']=float(np.mean([z['noiseless_return_vs_current_cosine']<0 for z in r['gradient_diagnostics']]))
    path=R/f'results/{phase}_raw.jsonl'
    with path.open('a') as f: f.write(json.dumps(r,allow_nan=False)+'\n')
    if phase=='formal' and method=='polar_64':
        np.savez_compressed(R/f'results/calibration_a{angle}_s{seed}.npz',cal_fhat=w.cal_fhat,cal_bhat=w.cal_bhat)
    print(json.dumps(dict(phase=phase,d=d,angle=angle,method=method,seed=seed,accuracy=r['service']['accuracy'],ce=r['service']['ce'])),flush=True)
    return r

def batch(seeds,g,method,angle,d,phase):
    return [train(s,g,method,angle,d,phase) for s in seeds]

def summary(rows):
    return dict(accuracy=B.stats([r['service']['accuracy'] for r in rows]),ce=B.stats([r['service']['ce'] for r in rows]),negative_fraction=B.stats([r['noiseless_negative_fraction'] for r in rows]))

def qualification():
    verify_freeze('freeze_A.json')
    assert not (R/'results/preflight_raw.jsonl').exists()
    checks=K.verify(); dump(R/'results/implementation_checks.json',checks)
    entries=[]; common=None; chosen=None
    for d in C['displacement_candidates']:
        gs={}; metrics=[]
        for angle in C['angles']:
            g,info=geometry(d,angle); gs[angle]=g; metrics.append(info)
        if common is None:
            common=batch(C['preflight_seeds'],gs[0],'CSI',0,d,'preflight')
        cs=summary(common)
        ss={}
        for angle in C['angles']:
            ss[str(angle)]=summary(batch(C['preflight_seeds'],gs[angle],'S0',angle,d,'preflight'))
        passed=cs['accuracy']['mean']>=.9 and all(v['accuracy']['mean']<=.7 for v in ss.values())
        entry=dict(d=d,geometry=metrics,CSI=cs,S0=ss,passed=passed)
        entries.append(entry)
        dump(R/'results/preflight.json',dict(candidates=entries,selected_d=d if passed else None,passed=passed))
        print('PREFLIGHT_GATE '+json.dumps(entry),flush=True)
        if passed: chosen=d; break
        if cs['accuracy']['mean']<.9: break
    assert chosen is not None or not json.loads((R/'results/preflight.json').read_text())['passed']

def formal():
    verify_freeze('freeze_B.json')
    gate=json.loads((R/'results/preflight.json').read_text()); assert gate['passed']
    assert not (R/'results/formal_raw.jsonl').exists()
    d=gate['selected_d']; seeds=C['seeds']; rows=[]; baseline={}; gs={}; events=[]
    for angle in C['angles']: gs[angle]=geometry(d,angle,False)[0]
    csi=batch(seeds,gs[0],'CSI',0,d,'formal'); rows+=csi
    for angle in C['angles']:
        baseline[angle]=batch(seeds,gs[angle],'S0',angle,d,'formal'); rows+=baseline[angle]
    schedules=[('G37.1','S_hex',[60,0,120]),('G37.2','S_perm',[30,90]),('G37.3','polar_64',[60,0,120,30,90])]
    for line,method,angles in schedules:
        for i,angle in enumerate(angles):
            rr=batch(seeds,gs[angle],method,angle,d,'formal'); rows+=rr
            s=summary(rr); acc=s['accuracy']['mean']
            gain=B.stats([x['service']['ce']-y['service']['ce'] for x,y in zip(baseline[angle],rr)])
            gap=B.stats([x['service']['ce']-y['service']['ce'] for x,y in zip(rr,csi)])
            if line=='G37.1': hit=angle==60 and (acc<.85 or gain['mean']<.20)
            elif line=='G37.2': hit=angle==30 and acc>=.75
            else: hit=acc<.90
            event=dict(line=line,angle=angle,method=method,summary=s,CE_gain_vs_S0=gain,CE_gap_vs_CSI=gap,triggered=bool(hit),skipped_angles=angles[i+1:] if hit else [],closed=bool(hit or i==len(angles)-1))
            events.append(event); dump(R/'results/stop_events.json',events); dump(R/'results/all_runs.json',rows)
            print('LINE_EVENT '+json.dumps(event),flush=True)
            if hit: break
    dump(R/'results/all_runs.json',rows)

if __name__=='__main__':
    phase=sys.argv[1]; start=time.time()
    dump(R/f'logs/{phase}_process.json',dict(pid=os.getpid(),start=start,host=platform.node()))
    try:
        {'preflight':qualification,'formal':formal}[phase]()
        dump(R/f'results/{phase}_runtime.json',dict(seconds=time.time()-start,success=True))
    except BaseException:
        dump(R/f'results/{phase}_failure.json',dict(traceback=traceback.format_exc()))
        raise
