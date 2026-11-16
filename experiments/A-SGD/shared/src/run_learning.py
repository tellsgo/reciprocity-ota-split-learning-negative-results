"""Frozen geometry-derived joint learning; all cases/arms and failures retained."""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import json,pathlib,hashlib,time,resource,signal,platform,csv
ROOT=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'configs/config.json').read_text())
resource.setrlimit(resource.RLIMIT_CPU,(C['cpu_limit_seconds'],C['cpu_limit_seconds']))
signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall budget')))
signal.alarm(C['wall_limit_seconds'])
import numpy as np
import learning_core as learning
import geometry_core as geometry

def factor(g):
    r=g['r0'];d,v=np.linalg.eigh(r);root=np.einsum('ik,k,jk->ij',v,np.sqrt(np.maximum(d,0)),v.conj())
    if g['info']['displacement_lambda']==0: lower=root.copy();qroot=np.zeros((3,3),dtype=complex)
    else:
        lower=geometry.mm(g['a'],root);dq,vq=np.linalg.eigh(g['q']);dq=np.where(dq<1e-12,0,dq)
        qroot=np.einsum('ik,k,jk->ij',vq,np.sqrt(np.maximum(dq,0)),vq.conj())
    jointroot=np.block([[root,np.zeros((3,3))],[lower,qroot]])
    target=np.block([[r,g['rt'].conj().T],[g['rt'],r]])
    err=np.linalg.norm(geometry.mm(jointroot,jointroot.conj().T)-target)/np.linalg.norm(target)
    assert err<1e-10
    return jointroot,float(err)

class World(learning.World):
    def __init__(self,seed,g,jointroot):
        rng=np.random.default_rng(C['train_seed_offset']+seed);n=C['rf_observation_quota']
        self.inputs=rng.choice([-1.,1.],(n,3));self.labels=(self.inputs[:,0]*self.inputs[:,1]>0).astype(float)
        both=np.einsum('ij,bjn->bin',jointroot,learning.cn(rng,(n,6,2)))
        self._f=both[:,:3];self._b=both[:,3:]
        self._nf=C['forward_noise_std']*learning.cn(rng,(n,3));self._nb=C['reverse_noise_std']*learning.cn(rng,(n,2))
        self._np=C['pilot_noise_std']*learning.cn(rng,(n,3,2))
        crng=np.random.default_rng(C['calibration_seed_offset']+seed);k=C['matrix_calibration_pairs']
        pair=np.einsum('ij,bjn->bin',jointroot,learning.cn(crng,(k,6,2)))
        self.cal_fhat=pair[:,:3]+C['pilot_noise_std']*learning.cn(crng,(k,3,2))
        self.cal_bhat=pair[:,3:]+C['pilot_noise_std']*learning.cn(crng,(k,3,2))
        trng=np.random.default_rng(C['test_seed_offset']+seed)
        self.test_inputs=trng.choice([-1.,1.],(C['test_samples'],3));self.test_labels=(self.test_inputs[:,0]*self.test_inputs[:,1]>0).astype(float)
        self._test_f=np.einsum('ij,bjn->bin',jointroot[:3,:3],learning.cn(trng,(C['test_samples'],3,2)))
        self._test_n=C['forward_noise_std']*learning.cn(trng,(C['test_samples'],3));self.test_epsilon=learning.cn(trng,(C['test_samples'],2))
        self.fit_diagnostics={}

def fit_temporal(world,k,scalar=False):
    f,clips=geometry.qcomplex(world.cal_fhat[:k],C['channel_quant_bound']);b=world.cal_bhat[:k]
    r=np.einsum('bmn,bln->ml',f,f.conj())/(2*k)-C['pilot_noise_std']**2*np.eye(3)
    cross=np.einsum('bmn,bln->ml',b,f.conj())/(2*k)
    reg=(r+r.conj().T)/2+C['calibration_ridge']*np.eye(3)
    d,v=np.linalg.eigh(reg);dc=np.maximum(d,C['calibration_covariance_floor'])
    alpha=np.trace(cross)/np.sum(dc)
    info={'channel_clips':clips,'matrix_clips':0,'covariance_floor_count':int(np.count_nonzero(d<C['calibration_covariance_floor']))}
    if scalar:
        post,state=geometry.scalar_post(alpha,True)
        adjusted=0j if post==0 else 1/post.conjugate()
        info.update(floor_count=int(state['floor']),unavailable_zero=state['unavailable_zero'])
        world.fit_diagnostics['scalar_est']={'estimated_alpha':{'real':float(alpha.real),'imag':float(alpha.imag)},'effective_alpha':{'real':float(adjusted.real),'imag':float(adjusted.imag)}}
        return alpha*np.eye(3),np.eye(3),adjusted,info
    inverse=np.einsum('ik,k,jk->ij',v,1/dc,v.conj());ahat=geometry.mm(cross,inverse)
    corr,ci=geometry.correction(ahat,True)
    info.update(matrix_clips=ci['quant_clips'],floor_count=3-ci['rank'],rank=ci['rank'],singular_values=ci['singular_values'])
    world.fit_diagnostics['matrix_est']={'estimated_A':geometry.pack(ahat)}
    return ahat,corr,1.,info

learning.fit_temporal=fit_temporal

def rng_audit():
    offsets=[C[k] for k in ['calibration_seed_offset','train_seed_offset','test_seed_offset','initialization_seed_offset']]+[151000]
    streams=[seed+off for seed in C['environment_seeds'] for off in offsets]
    assert len(set(streams))==len(C['environment_seeds'])*len(offsets)
    old=set(range(5100,5132))|{7102,9301,121014,987654321,132001,142001,491000}
    for seeds in [range(6100,6116),range(271000,271032)]:
        for seed in seeds:
            for off in offsets+[71000]:old.add(seed+off)
    for seed in range(381000,381016):old.update([seed,seed+10000,seed+20000])
    independent_covariance_seed=2300000
    assert not (set(streams)|{C['bootstrap_seed'],independent_covariance_seed})&old
    assert C['bootstrap_seed'] not in streams and independent_covariance_seed not in streams
    return {'training_stream_count':len(streams),'derived_training_seeds':streams,'known_prior_stream_overlap':[],
            'bootstrap_seed':C['bootstrap_seed'],'covariance_check_seed':independent_covariance_seed,
            'unit_derivative_check_note':'Original deterministic derivative/score unit checks rerun unchanged; they do not generate training/calibration/test data.'}

def checks(models):
    checks=learning.verify();assert geometry.scalar_post(0j,True)[0]==0
    rng=np.random.default_rng(2300000);z=learning.cn(rng,(8192,6,2));worst=0.
    for g,root,_ in models:
        pair=np.einsum('ij,bjn->bin',root,z)
        for covcheck in geometry.model_checks(pair[:,:3],pair[:,3:],g).values():worst=max(worst,covcheck['max_z'])
    assert worst<6
    # Explicit all-zero measured cross-covariance must not leave a zero divisor.
    fake=type('Probe',(),{})();fake.cal_fhat=np.zeros((8,3,2),complex);fake.cal_bhat=np.zeros((8,3,2),complex);fake.fit_diagnostics={}
    _,_,alpha,info=fit_temporal(fake,8,True);assert alpha==0 and info['unavailable_zero']
    checks.update(scalar_zero_guard=True,joint_factor_max_relative_error=max(e for _,_,e in models),fresh_covariance_max_z=float(worst))
    return checks

def analyze(rows,models):
    rng=np.random.default_rng(C['bootstrap_seed']);ids=rng.integers(0,len(C['environment_seeds']),(C['bootstrap_replicates'],len(C['environment_seeds'])))
    lookup={(r['seed'],r['case'],r['method']):r for r in rows};out=[]
    def contrast(case,method):return np.array([lookup[(s,case,'raw')]['service']['ce']-lookup[(s,case,method)]['service']['ce'] for s in C['environment_seeds']])
    def stat(d):return {'mean_gain':float(d.mean()),'median_gain':float(np.median(d)),'ci95':np.percentile(d[ids].mean(axis=1),[2.5,97.5]).tolist(),'wins':int((d>0).sum()),'all_paired_gains':d.tolist()}
    for g,_,_ in models:
        case=g['info']['case'];armout={}
        for m in C['methods']:
            rr=[lookup[(seed,case,m)] for seed in C['environment_seeds']]
            cal=[r['calibration_relative_error'] for r in rr if r['calibration_relative_error'] is not None]
            armout[m]={'mean_ce':float(np.mean([r['service']['ce'] for r in rr])),'mean_accuracy':float(np.mean([r['service']['accuracy'] for r in rr])),
              'calibration_relative_error':None if not cal else float(np.mean(cal)),'costs':rr[0]['costs'],
              'matrix_rank_counts':{str(rank):sum(r['calibration'].get('rank')==rank for r in rr) for rank in range(4)},
              'alpha_mean_absolute_error':float(np.mean([r.get('alpha_absolute_error',0) for r in rr])) if m=='scalar_est' else None,
              'mean_score_clipping':float(np.mean([r['score_train_clip_fraction'] for r in rr]))}
        out.append({'case':case,'geometry':g['info'],'arms':armout,'scalar_vs_raw':stat(contrast(case,'scalar_est'))})
    half=[g['info']['case'] for g,_,_ in models if g['info']['displacement_lambda']==.5]
    d=np.mean([contrast(case,'scalar_est') for case in half],axis=0);primary=stat(d)
    passed=primary['mean_gain']>=.02 and primary['ci95'][0]>0 and primary['wins']>=C['primary_min_winning_environments']
    # Exact expected copies across delay/direction: they are not extra replications.
    for seed in C['environment_seeds']:
        for spectrum in C['spectra']:
            cases=[g['info']['case'] for g,_,_ in models if g['info']['spectrum']==spectrum]
            for m in ['csi_rf_budget','csi_same_samples','score_0.1','score_0.3','rx_only']:
                vals=[lookup[(seed,case,m)]['service']['ce'] for case in cases]
                assert max(vals)-min(vals)<1e-12
    return {'primary':primary,'primary_gate_passed':bool(passed),'primary_four_cases':half,'independent_units':len(C['environment_seeds']),'baseline_copy_check':True,'cases':out}

def main():
    started=time.time();files=['configs/config.json','configs/SPEC.md','src/learning_core.py','src/geometry_core.py','src/run_learning.py']
    frozen={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
    ra=rng_audit();(ROOT/'results/freeze_manifest.json').write_text(json.dumps({'hashes':frozen,'rng_audit':ra,'task':C['task_name'],'pid':os.getpid()},indent=2))
    models=[]
    for spectrum in C['spectra']:
        for direction in C['directions']:
            for d in C['displacements_lambda']:
                g=geometry.geometry(spectrum,direction,d);root,err=factor(g);models.append((g,root,err))
    verification=checks(models);rows=[]
    for seed in C['environment_seeds']:
        for g,root,_ in models:
            case=g['info']['case'];world=World(seed,g,root)
            for method in C['methods']:
                row=learning.train(world,seed,case,method,g['a'])
                row['statistic_diagnostics']=world.fit_diagnostics.get(method,{})
                if method=='scalar_est':
                    al=world.fit_diagnostics[method]['estimated_alpha'];ahat=al['real']+1j*al['imag'];row['alpha_absolute_error']=float(abs(ahat-g['alpha']))
                    row['alpha_relative_error']=float(abs(ahat-g['alpha'])/max(abs(g['alpha']),1e-30))
                n=row['costs']['training_labels'];row['costs']['encoder_updates']=0 if method=='rx_only' else int(np.ceil(n/C['batch_size']))
                row['costs']['receiver_optimizer']='bounded_LBFGS' if method=='rx_only' else 'Adam'
                row['costs']['unused_rf_quota']=max(0,C['rf_observation_quota']-row['costs']['rf_observations'])
                row['costs']['rf_quota_excess']=max(0,row['costs']['rf_observations']-C['rf_observation_quota'])
                row['costs']['free_oracle_matrix_knowledge']=method=='matrix_oracle'
                rows.append(row)
        (ROOT/'results/partial_runs.json').write_text(json.dumps(rows))
        print(json.dumps({'seed_complete':seed,'rows':len(rows),'wall_seconds':time.time()-started}),flush=True)
    report=analyze(rows,models);usage=resource.getrusage(resource.RUSAGE_SELF)
    report['metadata']={'task':C['task_name'],'host':platform.node(),'pid':os.getpid(),'rows':len(rows),'hashes':frozen,'checks':verification,'rng_audit':ra,
        'user_cpu_seconds':usage.ru_utime,'system_cpu_seconds':usage.ru_stime,'wall_seconds':time.time()-started,'config':C}
    (ROOT/'results/all_runs.json').write_text(json.dumps(rows,indent=2));(ROOT/'results/summary.json').write_text(json.dumps(report,indent=2))
    with (ROOT/'results/summary.csv').open('w') as file:
        cols=['case','method','mean_ce','mean_accuracy','rf','control_bits','examples','updates'];wr=csv.DictWriter(file,fieldnames=cols);wr.writeheader()
        for case in report['cases']:
            for m,r in case['arms'].items():
                co=r['costs'];wr.writerow({'case':case['case'],'method':m,'mean_ce':r['mean_ce'],'mean_accuracy':r['mean_accuracy'],'rf':co['rf_observations'],'control_bits':co['control_bits'],'examples':co['training_labels'],'updates':co['encoder_updates']})
    print(json.dumps({'status':'complete','rows':len(rows),'wall_seconds':time.time()-started,'primary_gate_passed':report['primary_gate_passed'],'primary':report['primary'],'checks':verification}),flush=True)

if __name__=='__main__':main()
