"""Frozen G43 experiment; no parameter selection from observed performance."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys, json, pathlib, time, hashlib, resource, signal, traceback
import numpy as np
import g35_reference as L
K=L.K; G=L.G
R=pathlib.Path(__file__).resolve().parents[1]
PRE=list(range(943430000,943430008)); SEEDS=list(range(843430000,843430008))
ORDER=['CSI','SGD_MMSE','SGD_single','SGD_ZF','Adam_MMSE','Adam_single','Adam_ZF']
METHOD={'ZF':'ZF_polar','MMSE':'MMSE_polar','single':'single_user'}
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False))
def norm(x): return float(np.linalg.norm(x))
def cosine(x,y): return float(np.vdot(x,y).real/max(norm(x)*norm(y),1e-300))
def check_hash(name):
    for p,h in json.loads((R/name).read_text())['hashes'].items():
        assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p
def arm_info(name):
    if name in ['CSI','S0']:return 'Adam',name
    return name.split('_',1)
def setup(wo,name):
    _,kind=arm_info(name)
    if kind=='CSI':return np.conj(L.geometry()[1][0])*np.eye(4),{},{}
    if kind=='S0':return np.eye(4,dtype=complex),{},{}
    return G.estimate(wo,METHOD[kind])
def noise(wo,name):
    _,kind=arm_info(name)
    if kind=='CSI':return np.zeros_like(wo['white'])
    return L.additive(wo['white'],wo['users'],dict(kind='multi' if kind in ['ZF','MMSE'] else 'baseline',db=0))
def covariance(name):
    _,kind=arm_info(name)
    return L.covariance(dict(kind='ideal' if kind=='CSI' else ('multi' if kind in ['ZF','MMSE'] else 'baseline'),db=0))
def gradients(wo,W,w,b,T,name,ids):
    inp=wo['inputs'][ids];h=wo['h0'][ids,0]
    loss,s,gw,gb,_=K.task(G.forward(h,W,inp,wo['nf'][ids]),wo['labels'][ids],w,b);s=s[:,0]
    true=h*s[:,None]
    if name=='CSI':gx=true
    else:
        mag=abs(s);tx=np.divide(s,mag,out=np.zeros_like(s),where=mag>0)
        assert np.max(abs(tx)**2)<=1+1e-12 and mag.max()<=8+1e-12
        received=wo['h1'][ids,0]*tx[:,None]+noise(wo,name)[ids]
        gx=K.mv(T,received)*mag[:,None]
    return K.pgrad(gx,inp),K.pgrad(true,inp),np.r_[gw,gb],loss,s,true,gx
def initial_diagnostics(wo,seed,name,folder):
    W,w,b=G.initial(seed,4);T,cal,mats=setup(wo,name);a,ph=L.geometry();B=ph[0]*T
    ids=np.arange(L.N);ge,gt,gr,loss,s,true,gx=gradients(wo,W,w,b,T,name,ids)
    expected=K.mv(B,true);gm=K.pgrad(expected,wo['inputs'])
    rho=np.trace(np.outer(a[0],a[0].conj())@B)/4
    per=np.einsum('na,nb->nab',expected,wo['inputs'])
    var_sample=max(float(np.mean(np.sum(abs(per)**2,axis=(1,2)))-norm(gm)**2),0.)/32
    Rz=covariance(name);var_noise=float(np.mean(abs(s)**2*np.sum(wo['inputs']**2,axis=1))*np.trace(T@Rz@T.conj().T).real/32)
    noisy=np.einsum('na,nb->nab',gx,wo['inputs']).reshape(120,32,4,3).mean(1)
    truth=np.einsum('na,nb->nab',true,wo['inputs']).reshape(120,32,4,3).mean(1)
    cos=np.array([cosine(x,y) for x,y in zip(truth,noisy)])
    info=dict(seed=seed,arm=name,**L.diag(T),mean_gradient_cosine=cosine(gt,gm),
        noisy_full_gradient_cosine=cosine(gt,ge),batch_noisy_cosine_mean=float(cos.mean()),
        batch_noisy_cosine_min=float(cos.min()),batch_negative_fraction=float(np.mean(cos<0)),
        expected_gradient_norm_ratio=norm(gm)/norm(gt),noisy_gradient_norm_ratio=norm(ge)/norm(gt),
        gsnr=norm(gm)**2/(var_sample+var_noise),sample_variance=var_sample,noise_variance=var_noise,
        rho_trace_vs_gradient_error=float(abs(rho-np.vdot(gt,gm)/norm(gt)**2)),calibration=cal)
    assert info['rho_trace_vs_gradient_error']<1e-10
    np.savez_compressed(folder/f'init_{seed}_{name}.npz',W=W,w=w,b=b,T=T,B=B,
        true_gradient=gt,expected_gradient=gm,noisy_gradient=ge,batch_true=truth,batch_noisy=noisy,**{k:v for k,v in mats.items() if k!='T'})
    return info
class SGD:
    def __init__(self,*args):pass
    def step(self,x,g,lr):return x-lr*g
def train(wo,seed,name,folder):
    opt,_=arm_info(name);W,w,b=G.initial(seed,4);T,cal,mats=setup(wo,name)
    Opt=SGD if opt=='SGD' else K.Adam;aw=Opt(W.shape,complex);ar=Opt((2,),float)
    history={key:[] for key in ['W','receiver','gradient','true_gradient','receiver_gradient','unprojected_W_step','unprojected_receiver_step']}
    history['W'].append(W.copy());history['receiver'].append(np.r_[w,b])
    trace=[];initial=G.service(wo,W,w,b)
    for step in range(120):
        ids=np.arange(step*32,(step+1)*32)
        grad,gt,gr,loss,*_=gradients(wo,W,w,b,T,name,ids)
        proposed=aw.step(W,grad,.025);dw=proposed-W
        W=proposed*min(1.,(1/np.sqrt(3))/max(norm(proposed),1e-15))
        par=ar.step(np.r_[w,b],gr,.06);dr=par-np.r_[w,b];par=np.clip(par,-4,4);w=par[:1];b=float(par[1])
        assert np.isfinite(W).all() and np.isfinite(par).all()
        trace.append(dict(step=step+1,train_ce=float(loss.mean()),gradient_norm=norm(grad),true_gradient_norm=norm(gt),
            gradient_cosine=cosine(gt,grad),update_norm=norm(dw),projected_update_norm=norm(W-history['W'][-1]),encoder_norm=norm(W)))
        for key,val in [('W',W),('receiver',par),('gradient',grad),('true_gradient',gt),('receiver_gradient',gr),
                        ('unprojected_W_step',dw),('unprojected_receiver_step',dr)]: history[key].append(val.copy())
    service=G.service(wo,W,w,b)
    np.savez_compressed(folder/f'model_{seed}_{name}.npz',W=W,w=w,b=b,T=T,**{k:v for k,v in mats.items() if k!='T'})
    np.savez_compressed(folder/f'trace_{seed}_{name}.npz',**{k:np.asarray(v) for k,v in history.items()})
    row=dict(seed=seed,arm=name,optimizer=opt,initial_service=initial,service=service,physics=L.diag(T),calibration=cal,
        training_examples=3840,updates=120,trace=trace,mean_update_norm=float(np.mean([x['update_norm'] for x in trace])))
    print('TRAIN',seed,name,service,flush=True)
    return row
def checks():
    wo=L.world(PRE[0]);W,w,b=G.initial(PRE[0],4);ids=np.arange(23);x=wo['inputs'][ids];h=wo['h0'][ids,0];lab=wo['labels'][ids]
    def objective(W,w,b):return K.task(G.forward(h,W,x,wo['nf'][ids]),lab,w,b)[0].mean()+.001*np.sum(w*w)/2
    _,s,gw,gb,_=K.task(G.forward(h,W,x,wo['nf'][ids]),lab,w,b);gt=K.pgrad(h*s,x)
    eps=1e-6;errors=[]
    for ix in np.ndindex(W.shape):
        for z in [1.,1j]:
            e=np.zeros_like(W);e[ix]=z
            errors.append(abs((objective(W+eps*e,w,b)-objective(W-eps*e,w,b))/(2*eps)-np.vdot(gt,e).real))
    errors.extend([abs((objective(W,w+eps,b)-objective(W,w-eps,b))/(2*eps)-gw[0]),abs((objective(W,w,b+eps)-objective(W,w,b-eps))/(2*eps)-gb)])
    assert max(errors)<1e-7
    matrices={}
    for name in ['Adam_ZF','Adam_MMSE','Adam_single']:
        T,d,m=setup(wo,name);assert d['Q_unitarity_error']<1e-12
        if 'P' in m:
            assert d['P_idempotence_error']<1e-12 and d['P_rank']==2 and d['estimated_null_residual']<1e-12
        else:assert d['linear_solve_relative_residual']<1e-12
        matrices[name]=d
    # Fixed gradient sequence test; no additional training configurations.
    seq=K.cn(L.rng(PRE[0],80),(17,4,3));adam1=K.Adam((4,3),complex);adam2=K.Adam((4,3),complex)
    m1=np.zeros((4,3),complex);v1=np.zeros((4,3));m2=m1.copy();v2=v1.copy();diff=[];actual=[];sgd=[]
    for t,g in enumerate(seq,1):
        c=.01;m1=.9*m1+.1*g;v1=.99*v1+.01*abs(g)**2;m2=.9*m2+.1*c*g;v2=.99*v2+.01*abs(c*g)**2
        d1=(m1/(1-.9**t))/np.sqrt(v1/(1-.99**t));d2=(m2/(1-.9**t))/np.sqrt(v2/(1-.99**t));diff.append(norm(d1-d2))
        actual.append(norm(adam1.step(np.zeros_like(g),g,.025)-adam2.step(np.zeros_like(g),c*g,.025)))
        sgd.append(norm(SGD().step(np.zeros_like(g),c*g,.025))/norm(SGD().step(np.zeros_like(g),g,.025)))
    assert max(diff)<1e-12 and max(abs(np.array(sgd)-.01))<1e-14
    a,ph=L.geometry();H=a[1:].T;P=np.eye(4)-H@np.linalg.solve(H.conj().T@H,H.conj().T)
    from scipy.special import j0
    return dict(finite_difference_max_error=max(errors),matrices=matrices,optimizer_scale_check=dict(scale=.01,
        SGD_step_norm_ratios=sgd,Adam_eps0_max_step_difference=max(diff),Adam_actual_eps_max_step_difference=max(actual)),
        nominal_isotropic_J0_at_d=float(j0(2*np.pi*.383)),actual_rank1_temporal_correlation=dict(real=float(ph[0].real),imag=float(ph[0].imag),magnitude=float(abs(ph[0]))),
        ideal_ZF_target_retained_energy=norm(P@a[0])**2/4,paired_world_exact=all(np.array_equal(v,L.world(PRE[0])[k]) for k,v in wo.items()))
def world_power(wo,seed):
    a,_=L.geometry();inter=np.einsum('bu,ua->ba',wo['users'],a[1:])/np.sqrt(2)
    ps=float(np.mean(np.sum(abs(wo['h1'][:,0])**2,axis=1)));pi=float(np.mean(np.sum(abs(inter)**2,axis=1)))
    return dict(seed=seed,expected_SIR_dB=0.,sample_signal_power=ps,sample_interference_power=pi,sample_SIR_dB=float(10*np.log10(ps/pi)))
def preflight():
    check_hash('freeze_A.json');dump(R/'results/numerical_checks.json',checks())
    rows=[];diags=[];powers=[];folder=R/'results/preflight'
    for seed in PRE:
        wo=L.world(seed);np.savez_compressed(folder/f'world_{seed}.npz',**wo);powers.append(world_power(wo,seed))
        for name in ['S0']+ORDER:diags.append(initial_diagnostics(wo,seed,name,folder))
        dump(R/'results/preflight_diagnostics.json',diags)
        for name in ['S0','CSI']:
            row=train(wo,seed,name,folder);rows.append(row)
            with (R/'results/preflight_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    means={m:float(np.mean([r['service']['accuracy'] for r in rows if r['arm']==m])) for m in ['S0','CSI']}
    gate=dict(means=means,passed=means['S0']<=.70 and means['CSI']>=.90,powers=powers)
    dump(R/'results/preflight.json',gate);dump(folder/'all_runs.json',rows);print('GATE',gate,flush=True)
def verdicts(rows):
    acc={name:np.array([r['service']['accuracy'] for r in rows if r['arm']==name]) for name in ORDER}
    claims={}
    def record(key,support,false,**values):
        claims[key]=dict(verdict='archived' if false else ('archived' if support else 'archived，archived'),support=bool(support),falsified=bool(false),**values)
        if false and not (R/f'results/STOP_{key}.json').exists():
            dump(R/f'results/STOP_{key}.json',dict(time=time.time(),claim=key,result=claims[key],action='closed immediately after fixed eight-seed batch; no further tests of this claim'))
    if len(acc['SGD_ZF'])==8 and len(acc['SGD_single'])==8:
        z=float(acc['SGD_ZF'].mean());d=float(np.mean(acc['SGD_single']-acc['SGD_ZF']))
        record('G43.1',z<.65 and d>=.25,z>=.75,SGD_ZF_accuracy=z,SGD_loss=d)
    if len(acc['Adam_ZF'])==8 and len(acc['Adam_single'])==8:
        z=float(acc['Adam_ZF'].mean());d=float(np.mean(acc['Adam_single']-acc['Adam_ZF']))
        record('G43.2',z>=.80 and d<=.12,z<.75,Adam_ZF_accuracy=z,Adam_loss=d)
    if 'G43.1' in claims and 'G43.2' in claims:
        ds=claims['G43.1']['SGD_loss'];da=claims['G43.2']['Adam_loss']
        if da>0 and ds>=0:record('G43.3',ds/da>=3,ds/da<2,ratio=ds/da,SGD_loss=ds,Adam_loss=da)
        else:claims['G43.3']=dict(verdict='archived',support=False,falsified=False,ratio=None,SGD_loss=ds,Adam_loss=da)
    dump(R/'results/claims.json',claims);return claims
def main():
    check_hash('freeze_B.json');assert json.loads((R/'results/preflight.json').read_text())['passed']
    worlds={};diags=[];powers=[]
    for seed in SEEDS:
        wo=L.world(seed);worlds[seed]=wo;np.savez_compressed(R/f'results/worlds/world_{seed}.npz',**wo);powers.append(world_power(wo,seed))
        for name in ORDER:diags.append(initial_diagnostics(wo,seed,name,R/'results/models'))
    dump(R/'results/main_initial_diagnostics.json',diags);dump(R/'results/main_power.json',powers)
    rows=[]
    for name in ORDER:
        for seed in SEEDS:
            row=train(worlds[seed],seed,name,R/'results/models');rows.append(row)
            with (R/'results/main_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('CLAIMS',verdicts(rows),flush=True)
    dump(R/'results/all_runs.json',rows)
if __name__=='__main__':
    stage=sys.argv[1];start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    try:
        {'preflight':preflight,'main':main}[stage]()
        dump(R/f'results/runtime_{stage}.json',dict(pid=os.getpid(),seconds=time.time()-start,finished=time.time()))
    except BaseException:
        (R/f'logs/failure_{stage}_{int(time.time())}.txt').write_text(traceback.format_exc());raise
