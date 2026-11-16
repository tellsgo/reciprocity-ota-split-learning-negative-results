"""G35 preregistered independent additive interference experiment."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import sys, pathlib, json, time, hashlib, traceback, signal, resource
import numpy as np
from scipy.stats import spearmanr
import learning_core as K
import g24_reference as G
R=pathlib.Path(__file__).resolve().parents[1]
SEEDS=list(range(263500000,263500008)); PRE=list(range(263590000,263590008))
D=.383; N=3840; NT=8192; NM=4096
ARMS=[dict(name=f'white_naive_{p}',kind='white',method='naive_polar',db=p) for p in [0,10,20]]+[dict(name=f'multi_{m}_{p}',kind='multi',method=m,db=p) for m in ['naive_polar','ZF_polar','MMSE_polar'] for p in [0,10,20]]+[dict(name='CSI',kind='ideal',method='CSI',db=None)]
S0=dict(name='S0',kind='baseline',method='S0',db=None)
def dump(path,data): path.write_text(json.dumps(data,indent=2,allow_nan=False))
def norm(x):return float(np.linalg.norm(x))
def rng(s,t):return np.random.default_rng(np.random.SeedSequence([s,t]))
def geometry():
    a=np.array([G.steering(t) for t in [20,45,70]])
    return a,np.exp(2j*np.pi*D*np.cos(np.deg2rad([20,45,70])))
G.geometry=geometry

def world(seed):
    gen=rng(seed,1);x=gen.choice([-1.,1.],(N,3));h0,h1=G.channels(gen,N)
    out=dict(inputs=x,labels=(x[:,0]*x[:,1]>0).astype(float),h0=h0,h1=h1,nf=.1*K.cn(gen,(N,)))
    gen=rng(seed,2);c0,c1=G.channels(gen,64)
    out.update(cal0=c0+.1*K.cn(gen,c0.shape),cal1=c1+.1*K.cn(gen,c1.shape),cal_true0=c0,cal_true1=c1)
    gen=rng(seed,3);x=gen.choice([-1.,1.],(NT,3));h0,_=G.channels(gen,NT)
    out.update(test_inputs=x,test_labels=(x[:,0]*x[:,1]>0).astype(float),test_h=h0[:,0],test_n=.1*K.cn(gen,(NT,)))
    out.update(white=K.cn(rng(seed,10),(N,4)),users=K.cn(rng(seed,11),(N,2)))
    return out

def matrix(wo,arm):
    a,ph=geometry()
    if arm['method']=='CSI':return ph[0].conjugate()*np.eye(4),{},{}
    if arm['method']=='S0':return np.eye(4),{},{}
    return G.estimate(wo,arm['method'])

def covariance(arm):
    a,_=geometry()
    if arm['kind']=='ideal':return np.zeros((4,4),complex)
    if arm['kind']=='white':return 10**(-arm['db']/10)*np.eye(4)
    if arm['kind']=='baseline':return .0009*np.eye(4)
    return 10**(arm['db']/10)/2*(a[1:].T@a[1:].conj())+.0009*np.eye(4)

def additive(white,users,arm):
    a,_=geometry()
    if arm['kind']=='ideal':return np.zeros_like(white)
    if arm['kind']=='white':return 10**(-arm['db']/20)*white
    if arm['kind']=='baseline':return .03*white
    return np.sqrt(10**(arm['db']/10)/2)*np.einsum('...u,ua->...a',users,a[1:])+.03*white

def diag(T):
    a,ph=geometry();B=ph[0]*T;rho=np.vdot(a[0],B@a[0])/4
    residual=norm(B@a[0]-rho*a[0])/2
    return dict(rho_re=float(rho.real),rho_im=float(rho.imag),rho_abs=float(abs(rho)),scalar_residual=residual,
                sigma_max_B=float(np.linalg.norm(B,2)),noise_energy_gain=float(np.trace(T@T.conj().T).real/4))

def probe(wo,W,w,b,T,arm):
    inp=wo['test_inputs'];h=wo['test_h'];_,ph=geometry()
    _,s,_,_,_=K.task(G.forward(h,W,inp,wo['test_n']),wo['test_labels'],w,b);s=s[:,0]
    truth=h*s[:,None];mu0=K.pgrad(truth,inp)
    v=np.einsum('na,nb->nab',K.mv(ph[0]*T,truth),inp);mu=v.mean(0)
    Rz=covariance(arm);trz=float(np.trace(T@Rz@T.conj().T).real)
    vn=float(np.mean(abs(s)**2*np.sum(inp**2,axis=1))*trz)
    vs=max(float(np.mean(np.sum(abs(v)**2,axis=(1,2)))-norm(mu)**2),0.)
    variance=(vn+vs)/32;gsnr=norm(mu)**2/max(variance,1e-300)
    rho=complex(diag(T)['rho_re'],diag(T)['rho_im']);scalar=abs(rho)**2*norm(mu0)**2/max(variance,1e-300)
    return dict(gsnr=gsnr,gsnr_scalar=scalar,signal_power=norm(mu)**2,sample_variance=vs/32,noise_variance=vn/32,
                variance=variance,true_gradient_norm=norm(mu0),scalar_formula_relative_error=abs(scalar-gsnr)/max(gsnr,1e-300)),(truth,s,mu0)

def mc(wo,seed,W,w,b,T,arm):
    _,(truth,s,mu0)=probe(wo,W,w,b,T,arm);inp=wo['test_inputs'];_,ph=geometry()
    gen=rng(seed,20);ids=gen.integers(0,NT,(NM,32));white=K.cn(rng(seed,21),(NM,32,4));users=K.cn(rng(seed,22),(NM,32,2))
    noise=additive(white,users,arm)
    out={};arrays={}
    for kind,ii in [('unconditional',ids),('conditional',np.broadcast_to(np.arange(32),(NM,32)))]:
        mean_terms=np.einsum('ab,rnb->rna',ph[0]*T,truth[ii]);noisy=np.einsum('ab,rnb->rna',T,noise)*abs(s[ii])[...,None]
        grads=np.einsum('rna,rni->rai',mean_terms+noisy,inp[ii])/32
        true=mu0 if kind=='unconditional' else K.pgrad(truth[:32],inp[:32])
        means=grads.mean(0);rhoh=np.einsum('ai,rai->r',true.conj(),grads)/norm(true)**2
        den=np.linalg.norm(grads,axis=(1,2))*norm(true)
        cos=np.einsum('ai,rai->r',true.conj(),grads).real/np.maximum(den,1e-300)
        meanrho=rhoh.mean();se=np.std(rhoh,ddof=1)/np.sqrt(NM)
        out[kind]=dict(rho_re=float(meanrho.real),rho_im=float(meanrho.imag),rho_se=float(se),cosine_mean=float(cos.mean()),
            scalar_vector_error=norm(means-(diag(T)['rho_re']+1j*diag(T)['rho_im'])*true)/max(norm(true),1e-300),
            vector_gsnr=norm(means)**2/ max(float(np.mean(np.sum(abs(grads-means)**2,axis=(1,2)))),1e-300))
        arrays[kind+'_gradients']=grads;arrays[kind+'_true_gradient']=true
    np.savez_compressed(R/'results'/f'mc_{seed}_{arm["name"]}.npz',**arrays)
    return out

def train(wo,seed,arm,folder,measure=True):
    W,w,b=G.initial(seed,4);T,cal,mats=matrix(wo,arm);di=diag(T)
    n=2666 if arm['method']=='CSI' else N;aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float)
    tr=[];probes=[];mcr=None
    if measure:
        p,_=probe(wo,W,w,b,T,arm);probes.append(dict(step=0,**p));mcr=mc(wo,seed,W,w,b,T,arm)
    for start in range(0,n,32):
        ids=np.arange(start,min(start+32,n));inp=wo['inputs'][ids]
        y=G.forward(wo['h0'][ids,0],W,inp,wo['nf'][ids]);loss,s,gw,gb,_=K.task(y,wo['labels'][ids],w,b);s=s[:,0]
        truegx=wo['h0'][ids,0]*s[:,None]
        if arm['method']=='CSI':gx=truegx
        else:
            mag=abs(s);tx=np.divide(s,mag,out=np.zeros_like(s),where=mag>0)
            received=wo['h1'][ids,0]*tx[:,None]+additive(wo['white'][ids],wo['users'][ids],arm)
            gx=K.mv(T,received)*mag[:,None]
            assert np.max(abs(tx))<=1+1e-12 and np.max(mag)<8+1e-12
        grad=K.pgrad(gx,inp);gt=K.pgrad(truegx,inp)
        W=aw.step(W,grad,.025);W*=min(1.,(1/np.sqrt(3))/max(norm(W),1e-15))
        par=np.clip(ar.step(np.r_[w,b],np.r_[gw,gb],.06),-4,4);w=par[:1];b=float(par[1])
        assert np.isfinite(W).all() and np.isfinite(par).all()
        step=len(tr)+1;tr.append(dict(step=step,ce=float(loss.mean()),gradient_cosine=float(np.vdot(gt,grad).real/max(norm(gt)*norm(grad),1e-300)),gradient_norm=norm(grad),true_gradient_norm=norm(gt),encoder_norm=norm(W)))
        if measure and (step in [40,80,120] or start+32>=n):
            p,_=probe(wo,W,w,b,T,arm);probes.append(dict(step=step,**p))
    service=G.service(wo,W,w,b)
    np.savez_compressed(folder/f'model_{seed}_{arm["name"]}.npz',W=W,w=w,b=b,T=T,**{k:v for k,v in mats.items() if k!='T'})
    return dict(seed=seed,arm=arm,service=service,physics=di,calibration=cal,trace=tr,probes=probes,mc=mcr,training_examples=n,updates=len(tr))

def numerical_checks():
    seed=PRE[0];wo=world(seed);W,w,b=G.initial(seed,4);x=wo['inputs'][:23];h=wo['h0'][:23,0];noise=wo['nf'][:23];lab=wo['labels'][:23]
    def obj(W,w,b):return K.task(G.forward(h,W,x,noise),lab,w,b)[0].mean()+.001*sum(w*w)/2
    _,s,gw,gb,_=K.task(G.forward(h,W,x,noise),lab,w,b);g=K.pgrad(h*s,x);errs=[];eps=1e-6
    for ix in np.ndindex(W.shape):
        for z in [1.,1j]:
            e=np.zeros_like(W);e[ix]=z
            errs.append(abs((obj(W+eps*e,w,b)-obj(W-eps*e,w,b))/(2*eps)-np.vdot(g,e).real))
    errs.extend([abs((obj(W,w+eps,b)-obj(W,w-eps,b))/(2*eps)-gw[0]),abs((obj(W,w,b+eps)-obj(W,w,b-eps))/(2*eps)-gb)])
    assert max(errs)<1e-7
    checks={}
    for meth in ['naive_polar','ZF_polar','MMSE_polar']:
        T,d,m=G.estimate(wo,meth);checks[meth]=dict(**diag(T),calibration=d)
        assert d['Q_unitarity_error']<1e-12
        if meth=='MMSE_polar':assert d['linear_solve_relative_residual']<1e-12
        else:assert d['P_idempotence_error']<1e-12 and d['P_rank']==2
    assert all(np.array_equal(v,world(seed)[k]) for k,v in wo.items())
    return dict(finite_difference_max_error=max(errs),matrices=checks)

def check_hash(file):
    for rel,h in json.loads((R/file).read_text())['hashes'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel

def preflight():
    global D
    check_hash('preregistration.json');dump(R/'results/numerical_checks.json',numerical_checks())
    attempts=[]
    for d in [.383,.5]:
        D=d;folder=R/'results/preflight'/f'd_{d}';folder.mkdir()
        rows=[]
        for seed in PRE:
            wo=world(seed);np.savez_compressed(folder/f'world_{seed}.npz',**wo)
            for arm in [S0,ARMS[-1]]:
                row=train(wo,seed,arm,folder,False);rows.append(row);print('PREFLIGHT',d,seed,arm['name'],row['service'],flush=True)
        acc={m:float(np.mean([r['service']['accuracy'] for r in rows if r['arm']['name']==m])) for m in ['S0','CSI']}
        metrics={}
        for seed in PRE:
            wo=world(seed);metrics[str(seed)]={}
            for arm in [S0,ARMS[0],ARMS[6],ARMS[9],ARMS[-1]]:
                T,cal,mats=matrix(wo,arm);metrics[str(seed)][arm['method']]=dict(**diag(T),sigma_max_A_est=max(cal.get('A_singular_values',[1.])))
        dump(folder/'all_runs.json',rows);dump(folder/'physical_metrics.json',metrics)
        gate=acc['S0']<=.70 and acc['CSI']>=.90
        attempts.append(dict(d=d,accuracy=acc,gate=gate,folder=str(folder.relative_to(R))))
        dump(R/'results/preflight.json',dict(attempts=attempts,passed=gate,selected_d=d if gate else None))
        if gate:break
        if acc['CSI']<.90:break
    print('PREFLIGHT RESULT',attempts,flush=True)

def corr(x,y):return float(spearmanr(x,y).statistic)
def summary(rows):
    primary=[r for r in rows if r['arm']['name']!='S0'];means={}
    for arm in ARMS:
        rr=[r for r in primary if r['arm']['name']==arm['name']]
        means[arm['name']]={key:float(np.mean([fun(r) for r in rr])) for key,fun in dict(accuracy=lambda r:r['service']['accuracy'],ce=lambda r:r['service']['ce'],gsnr=lambda r:r['probes'][0]['gsnr'],gsnr_scalar=lambda r:r['probes'][0]['gsnr_scalar'],rho_re=lambda r:r['physics']['rho_re'],rho_im=lambda r:r['physics']['rho_im'],trajectory_gsnr=lambda r:float(np.exp(np.mean(np.log([p['gsnr'] for p in r['probes']]))))).items()}
    ac=[v['accuracy'] for v in means.values()];gs=[v['gsnr'] for v in means.values()];ce=[v['ce'] for v in means.values()]
    r=corr(gs,ac);rce=corr(gs,[-x for x in ce]);rs=corr([v['gsnr_scalar'] for v in means.values()],ac)
    correlations=dict(primary_accuracy=r,primary_minus_ce=rce,scalar_proxy_accuracy=rs,trajectory_accuracy=corr([v['trajectory_gsnnr'] if 'trajectory_gsnnr' in v else v['trajectory_gsnr'] for v in means.values()],ac),
        pooled_accuracy=corr([x['probes'][0]['gsnr'] for x in primary],[x['service']['accuracy'] for x in primary]),per_seed_accuracy={str(s):corr([x['probes'][0]['gsnr'] for x in primary if x['seed']==s],[x['service']['accuracy'] for x in primary if x['seed']==s]) for s in SEEDS})
    for kind in ['white','multi']:
        aa=[a for a in ARMS if a['kind']==kind];vals=[means[a['name']] for a in aa]
        correlations[kind]=dict(gsnr_accuracy=corr([v['gsnr'] for v in vals],[v['accuracy'] for v in vals]),noise_power_accuracy=corr([(-a['db'] if kind=='white' else a['db']) for a in aa],[v['accuracy'] for v in vals]))
    invariance={}
    for kind,lo,hi in [('white',20,0),('multi',0,20)]:
        ends=[]
        for p in [lo,hi]:
            rr=[r for r in primary if r['arm']['kind']==kind and r['arm']['method']=='naive_polar' and r['arm']['db']==p]
            rho=np.mean([complex(r['mc']['conditional']['rho_re'],r['mc']['conditional']['rho_im']) for r in rr]);co=float(np.mean([r['mc']['conditional']['cosine_mean'] for r in rr]))
            ends.append(dict(rho_re=float(rho.real),rho_im=float(rho.imag),cosine=co,aggregate_mc_se=float(np.sqrt(sum(r['mc']['conditional']['rho_se']**2 for r in rr))/8)))
        v=abs(complex(ends[1]['rho_re'],ends[1]['rho_im'])-complex(ends[0]['rho_re'],ends[0]['rho_im']))/abs(complex(ends[0]['rho_re'],ends[0]['rho_im']))
        invariance[kind]=dict(relative_deviation=v,low=ends[0],high=ends[1],falsified=bool(v>.10 or ends[0]['cosine']*ends[1]['cosine']<0),numerical_target_met=bool(v<=.03))
    comparisons=[]
    for db in [0,10,20]:
        z=means[f'multi_ZF_polar_{db}'];m=means[f'multi_MMSE_polar_{db}'];ratio=z['rho_re']/m['rho_re'];pred=m['accuracy']*ratio;chance=.5+(m['accuracy']-.5)*ratio
        comparisons.append(dict(db=db,rho_ratio=ratio,accuracy_ZF=z['accuracy'],accuracy_MMSE=m['accuracy'],gap=m['accuracy']-z['accuracy'],predicted_direct=pred,direct_error=abs(z['accuracy']-pred),predicted_chance=chance,chance_error=abs(z['accuracy']-chance),falsifier=bool(ratio>=.90 and m['accuracy']-z['accuracy']>.05)))
    # Resample independent seeds jointly for all arms, preserving pairings.
    lookup={(r['seed'],r['arm']['name']):r for r in primary};aa=np.array([[lookup[s,a['name']]['service']['accuracy'] for a in ARMS] for s in SEEDS]);gg=np.array([[lookup[s,a['name']]['probes'][0]['gsnr'] for a in ARMS] for s in SEEDS]);cc=np.array([[lookup[s,a['name']]['service']['ce'] for a in ARMS] for s in SEEDS]);idx=rng(263599999,1).integers(0,8,(20000,8))
    from scipy.stats import rankdata
    ba=aa[idx].mean(1);bg=gg[idx].mean(1);ra=rankdata(ba,axis=1);rg=rankdata(bg,axis=1);ra-=ra.mean(1)[:,None];rg-=rg.mean(1)[:,None];br=np.sum(ra*rg,1)/np.sqrt(np.sum(ra**2,1)*np.sum(rg**2,1))
    correlations['bootstrap95']=np.quantile(br,[.025,.975]).tolist()
    for j,a in enumerate(ARMS):means[a['name']]['accuracy_ci95']=np.quantile(ba[:,j],[.025,.975]).tolist();means[a['name']]['ce_ci95']=np.quantile(cc[idx,j].mean(1),[.025,.975]).tolist()
    claims={'G35.1':dict(strict_scalar_identity='counterexample; closed before main training',additive_invariance=invariance),'G35.2':dict(falsified=r<.8,support_sorting=r>=.9 and rce>=.9,rank=r),'G35.3':dict(falsified=any(c['falsifier'] for c in comparisons),rho_reduction_met=all(c['rho_ratio']<=.7 for c in comparisons),direct_prediction_within_5pp=all(c['direct_error']<=.05 for c in comparisons),comparisons=comparisons)}
    for key in ['G35.2','G35.3']:
        if claims[key]['falsified']:dump(R/'results'/f'STOP_{key}.json',dict(time=time.time(),reason=claims[key],action='closed at eight-seed batch; no further training'))
    if any(v['falsified'] for v in invariance.values()):dump(R/'results/STOP_G35.1_invariance.json',invariance)
    dump(R/'results/means.json',means);dump(R/'results/correlations.json',correlations);dump(R/'results/claims.json',claims)
    return claims

def main():
    global D
    check_hash('freeze_manifest.json');pf=json.loads((R/'results/preflight.json').read_text());assert pf['passed'];D=pf['selected_d']
    # A direct algebraic counterexample is enough to stop the universal scalar theorem line.
    dump(R/'results/STOP_G35.1_strict.json',dict(time=time.time(),C00=[[1,0],[0,1]],B=[[1,0],[0,0]],gradient=[1,0],rho=.5,expected_gradient=[1,0],claimed_gradient=[.5,0],relative_error=.5,action='strict universal scalar equality falsified; remaining frozen common experiments serve G35.2/3 and additive-invariance diagnostic'))
    rows=[]
    for seed in SEEDS:
        wo=world(seed);np.savez_compressed(R/'results/worlds'/f'world_{seed}.npz',**wo)
        for arm in ARMS+[S0]:
            row=train(wo,seed,arm,R/'results/models',arm['name']!='S0');rows.append(row)
            with (R/'results/runs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            print('MAIN',seed,arm['name'],row['service'],'GSNR',row['probes'][0]['gsnr'] if row['probes'] else '-',flush=True)
    dump(R/'results/all_runs.json',rows);claims=summary(rows);print('CLAIMS',json.dumps(claims),flush=True)

if __name__=='__main__':
    start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(900,900));signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    try:
        {'preflight':preflight,'main':main}[sys.argv[1]]()
        dump(R/'results'/f'runtime_{sys.argv[1]}.json',dict(pid=os.getpid(),wall_seconds=time.time()-start,finished=time.time()))
    except BaseException:
        (R/'logs'/f'failure_{sys.argv[1]}_{int(time.time())}.txt').write_text(traceback.format_exc());raise
