"""Post-run independent replay and frozen-endpoint statistics; never trains."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import pathlib,json,hashlib,csv,datetime,sys,traceback
import numpy as np
np.seterr(all='raise')
R=pathlib.Path(__file__).resolve().parents[1];ROOT=R.parents[1]
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def norm(x):return float(np.linalg.norm(x))
def read(p):return json.loads(p.read_text())
def mv(m,x):return np.einsum('ij,bj->bi',m,x)
def independent_service(wo,W,w,b):
    tx=mv(W,wo['test_inputs']);y=np.sum(wo['test_h'].conj()*tx,axis=1)+wo['test_n']
    logits=b+w[0]*np.tanh(abs(y)**2)
    return dict(accuracy=float(np.mean((logits>=0)==wo['test_labels'])),ce=float(np.mean(np.logaddexp(0,logits)-wo['test_labels']*logits)))
def run():
    rows=read(R/'results/all_runs.json');pre=read(R/'results/preflight/all_runs.json')
    assert len(rows)==56 and len(pre)==16
    keys=[(r['seed'],r['arm']) for r in rows];assert len(set(keys))==56
    endpoint=[];grad_errors=[];update_errors=[];state_errors=[];rho_errors=[];power=0.
    a=np.exp(-2j*np.pi*.5*np.arange(4)[None,:]*np.cos(np.deg2rad([20,45,70]))[:,None]);ph=np.exp(2j*np.pi*.383*np.cos(np.deg2rad([20,45,70])))
    world_cache={}
    for group,records in [('main',rows),('preflight',pre)]:
        folder=R/('results/models' if group=='main' else 'results/preflight')
        worldfolder=R/('results/worlds' if group=='main' else 'results/preflight')
        for row in records:
            seed=row['seed'];arm=row['arm'];key=group,seed
            if key not in world_cache:world_cache[key]=dict(np.load(worldfolder/f'world_{seed}.npz'))
            wo=world_cache[key];model=dict(np.load(folder/f'model_{seed}_{arm}.npz'));trace=dict(np.load(folder/f'trace_{seed}_{arm}.npz'))
            for v in list(wo.values())+list(model.values())+list(trace.values()):assert np.isfinite(v).all()
            service=independent_service(wo,model['W'],model['w'],float(model['b']))
            endpoint.extend(abs(service[k]-row['service'][k]) for k in ['accuracy','ce'])
            T=model['T'];rho=np.sum(a[0].conj()*np.einsum('ij,j->i',ph[0]*T,a[0]))/4
            rho_errors.append(abs(rho-complex(row['physics']['rho_re'],row['physics']['rho_im'])))
            is_sgd=arm.startswith('SGD');is_multi=arm.endswith('_ZF') or arm.endswith('_MMSE')
            mw=np.zeros((4,3),complex);vw=np.zeros((4,3));mr=np.zeros(2);vr=np.zeros(2)
            additive=.03*wo['white']
            if is_multi:additive=additive+np.einsum('bu,ua->ba',wo['users'],a[1:])/np.sqrt(2)
            for j in range(120):
                ii=slice(32*j,32*(j+1));inp=wo['inputs'][ii];h=wo['h0'][ii,0];W=trace['W'][j];p=trace['receiver'][j]
                tx=mv(W,inp);power=max(power,float(np.max(np.sum(abs(tx)**2,axis=1))))
                y=np.sum(h.conj()*tx,axis=1)+wo['nf'][ii];u=np.tanh(abs(y)**2);z=p[1]+p[0]*u;error=1/(1+np.exp(-z))-wo['labels'][ii]
                s=2*error*p[0]*(1-u*u)*y;gx=h*s[:,None];truth=np.einsum('ba,bi->ai',gx,inp)/32
                if arm!='CSI':
                    # Equivalent linear expression avoids the training unit-phase/gain path.
                    raw=wo['h1'][ii,0]*s[:,None]+additive[ii]*abs(s)[:,None];gx=mv(T,raw)
                grad=np.einsum('ba,bi->ai',gx,inp)/32;gr=np.array([np.mean(error*u)+.001*p[0],np.mean(error)])
                grad_errors.extend([norm(grad-trace['gradient'][j]),norm(truth-trace['true_gradient'][j]),norm(gr-trace['receiver_gradient'][j])])
                if is_sgd:dw=-.025*grad;dr=-.06*gr
                else:
                    mw=.9*mw+.1*grad;vw=.99*vw+.01*abs(grad)**2;mr=.9*mr+.1*gr;vr=.99*vr+.01*gr**2
                    dw=-.025*(mw/(1-.9**(j+1)))/(np.sqrt(vw/(1-.99**(j+1)))+1e-8)
                    dr=-.06*(mr/(1-.9**(j+1)))/(np.sqrt(vr/(1-.99**(j+1)))+1e-8)
                update_errors.extend([norm(dw-trace['unprojected_W_step'][j]),norm(dr-trace['unprojected_receiver_step'][j])])
                wn=W+dw;wn*=min(1.,(1/np.sqrt(3))/max(norm(wn),1e-15));pn=np.clip(p+dr,-4,4)
                state_errors.extend([norm(wn-trace['W'][j+1]),norm(pn-trace['receiver'][j+1])])
            state_errors.extend([norm(trace['W'][-1]-model['W']),norm(trace['receiver'][-1]-np.r_[model['w'],float(model['b'])])])
    assert max(endpoint)<1e-12 and max(grad_errors)<1e-10 and max(update_errors)<1e-10 and max(state_errors)<1e-10
    assert max(rho_errors)<1e-12 and power<=1+1e-12
    freeze_checks={}
    for name in ['freeze_A.json','freeze_B.json']:
        freeze_checks[name]={p:hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in read(R/name)['hashes'].items()}
        assert all(freeze_checks[name].values())
    source_checks={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in read(R/'SOURCE_MAP.json').items()}
    assert all(source_checks.values())
    pair_errors=[]
    for seed in sorted({r['seed'] for r in rows}):
        for suffix in ['ZF','MMSE','single']:
            x=dict(np.load(R/f'results/models/model_{seed}_SGD_{suffix}.npz'));y=dict(np.load(R/f'results/models/model_{seed}_Adam_{suffix}.npz'))
            pair_errors.append(norm(x['T']-y['T']))
    assert max(pair_errors)==0.
    dump(R/'results/verification.json',dict(models_verified=72,updates_verified=72*120,max_endpoint_error=max(endpoint),max_gradient_error=max(grad_errors),
        max_update_error=max(update_errors),max_state_error=max(state_errors),max_rho_error=max(rho_errors),max_forward_power=power,
        paired_optimizer_T_max_error=max(pair_errors),freeze_checks=freeze_checks,source_checks=source_checks,all_passed=True))
    seeds=sorted({r['seed'] for r in rows});arms=['SGD_ZF','SGD_MMSE','SGD_single','Adam_ZF','Adam_MMSE','Adam_single','CSI'];lookup={(r['seed'],r['arm']):r for r in rows}
    idx=np.random.default_rng(943439999).integers(0,8,(20000,8))
    def stat(v):
        v=np.array(v);return dict(mean=float(v.mean()),sd=float(v.std(ddof=1)),min=float(v.min()),max=float(v.max()),per_seed=v.tolist(),ci95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
    means={arm:{k:stat([lookup[s,arm]['service'][k] for s in seeds]) for k in ['accuracy','ce']} for arm in arms}
    ds=np.array([lookup[s,'SGD_single']['service']['accuracy']-lookup[s,'SGD_ZF']['service']['accuracy'] for s in seeds])
    da=np.array([lookup[s,'Adam_single']['service']['accuracy']-lookup[s,'Adam_ZF']['service']['accuracy'] for s in seeds])
    bs=ds[idx].mean(1);ba=da[idx].mean(1);valid=(ba>0)&(bs>=0)
    ratio=bs[valid]/ba[valid];contrasts=dict(SGD_loss=stat(ds),Adam_loss=stat(da),SGD_minus_Adam_loss=stat(ds-da),
        ratio_of_mean_losses=float(ds.mean()/da.mean()),ratio_bootstrap95=np.quantile(ratio,[.025,.975]).tolist(),
        ratio_interval_kind='unconditional percentile' if valid.all() else 'conditional on valid resamples',invalid_ratio_resample_fraction=float(1-valid.mean()),
        Adam_ZF_minus_SGD_ZF=stat([lookup[s,'Adam_ZF']['service']['accuracy']-lookup[s,'SGD_ZF']['service']['accuracy'] for s in seeds]))
    for opt in ['SGD','Adam']:
        contrasts[opt+'_MMSE_minus_ZF']=stat([lookup[s,opt+'_MMSE']['service']['accuracy']-lookup[s,opt+'_ZF']['service']['accuracy'] for s in seeds])
    diags=read(R/'results/main_initial_diagnostics.json');physics={}
    for arm in arms:
        rr=[d for d in diags if d['arm']==arm]
        physics[arm]={k:stat([d[k] for d in rr]) for k in ['rho_re','rho_im','rho_abs','scalar_residual','mean_gradient_cosine','batch_noisy_cosine_mean','batch_negative_fraction','gsnr','expected_gradient_norm_ratio']}
    claims=read(R/'results/claims.json');assert means['SGD_ZF']['accuracy']['mean']>=.75 and claims['G43.1']['falsified']
    assert means['Adam_ZF']['accuracy']['mean']>=.8 and da.mean()<=.12 and claims['G43.2']['support']
    assert contrasts['ratio_of_mean_losses']<2 and claims['G43.3']['falsified']
    dump(R/'results/summary.json',means);dump(R/'results/contrasts.json',contrasts);dump(R/'results/physics_summary.json',physics)
    metrics=[dict(seed=r['seed'],arm=r['arm'],**r['service'],rho_re=r['physics']['rho_re'],rho_im=r['physics']['rho_im'],mean_update_norm=r['mean_update_norm'],pilot_clips=r['calibration'].get('pilot_clips',0)) for r in rows]
    with (R/'results/by_seed.csv').open('w') as f:
        out=csv.DictWriter(f,fieldnames=list(metrics[0]));out.writeheader();out.writerows(metrics)
    with (R/'results/paired_losses.csv').open('w') as f:
        out=csv.writer(f);out.writerow(['seed','SGD_loss','Adam_loss','SGD_minus_Adam_loss'])
        out.writerows(zip(seeds,ds,da,ds-da))
    print(json.dumps(dict(verification=read(R/'results/verification.json'),means={k:v['accuracy'] for k,v in means.items()},contrasts=contrasts),indent=2))
if __name__=='__main__':
    try:run()
    except BaseException:
        (R/'logs/verification_failure.txt').write_text(traceback.format_exc());raise
