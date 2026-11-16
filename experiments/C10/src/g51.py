"""Frozen G51 two-user geometric classification; exact spatial operators."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import sys, pathlib, json, time, hashlib, traceback, signal, resource
import numpy as np
from scipy.stats import spearmanr, rankdata
import learning_core as K
R=pathlib.Path(__file__).resolve().parents[1]
C=json.loads((R/'configs/g51.json').read_text())
SEEDS=C['seeds']; PRE=C['preflight_seeds']; ARMS=C['arms']
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def rng(s,t):return np.random.default_rng(np.random.SeedSequence([s,t]))
def norm(x):return float(np.linalg.norm(x))
def cosine(x,y):
    assert norm(x)>1e-12 and norm(y)>1e-12, 'gradient direction undefined'
    return float(np.clip(np.vdot(x,y).real/(norm(x)*norm(y)),-1,1))
def geometry(delta):
    th=np.deg2rad([C['target_angle'],C['target_angle']+delta])
    a=np.exp(-2j*np.pi*C['spacing']*np.cos(th)[:,None]*np.arange(4))
    ph=np.exp(2j*np.pi*C['displacement']*np.cos(th))
    return a,ph

def initial(seed,stream=4):
    W=K.cn(rng(seed,stream),(4,3));return W/norm(W)*.4,np.array([.8]),-.4

def forward(h,W,x,n):return (np.einsum('bi,bi->b',h.conj(),K.mv(W,x))+n)[:,None]
def normalized(s):return np.divide(s,abs(s),out=np.zeros_like(s),where=abs(s)>0)
def dataset(seed,stream,n,delta):
    g=rng(seed,stream);a,ph=geometry(delta)
    x=g.choice([-1.,1.],(n,3));z=K.cn(g,(n,2));nf=.1*K.cn(g,(n,));nb=.03*K.cn(g,(n,4))
    x2=g.choice([-1.,1.],(n,3));nf2=.1*K.cn(g,(n,));h=z[:,:,None]*a[None,:,:]
    W2,w2,b2=initial(seed,5)
    _,s2,_,_,_=K.task(forward(h[:,1],W2,x2,nf2),(x2[:,0]*x2[:,1]>0).astype(float),w2,b2)
    inter=ph[1]*h[:,1]*normalized(s2[:,0])[:,None]
    return dict(x=x,lab=(x[:,0]*x[:,1]>0).astype(float),h=h,nf=nf,nb=nb,x2=x2,nf2=nf2,inter=inter)
def world(seed,delta):
    out={}
    for prefix,stream,n in [('train',1,C['train_n']),('test',3,C['test_n']),('probe',7,C['probe_n'])]:
        out.update({prefix+'_'+k:v for k,v in dataset(seed,stream,n,delta).items()})
    return out

def matrix(delta,arm):
    a,ph=geometry(delta);I=np.eye(4,dtype=complex);H=(ph[:,None]*a).T
    P=I-np.outer(H[:,1],H[:,1].conj())/np.vdot(H[:,1],H[:,1])
    F=H@np.linalg.solve(H.conj().T@H+C['reverse_std']**2*np.eye(2),np.eye(2))
    if arm=='S0':T=I
    elif arm=='ZF':T=ph[0].conjugate()*P
    elif arm=='MMSE':T=np.outer(a[0],F[:,0].conj())
    else:T=ph[0].conjugate()*I
    return T,dict(P=P,H=H,F=F)
def target_terms(wo,W,w,b,prefix,ids=None):
    x=wo[prefix+'_x'];h=wo[prefix+'_h'][:,0];nf=wo[prefix+'_nf'];lab=wo[prefix+'_lab']
    if ids is not None:x=x[ids];h=h[ids];nf=nf[ids];lab=lab[ids]
    loss,s,gw,gb,prob=K.task(forward(h,W,x,nf),lab,w,b)
    s=s[:,0];gx=h*s[:,None];gt=K.pgrad(gx,x)
    return x,s,gx,gt,np.r_[gw,gb],loss,prob

def gradients(wo,W,w,b,T,arm,delta,ids):
    x,s,gx,gt,gr,loss,_=target_terms(wo,W,w,b,'train',ids)
    if arm=='CSI':physical=gx
    else:
        _,ph=geometry(delta);inter=0 if arm=='single_user' else wo['train_inter'][ids]
        recv=ph[0]*wo['train_h'][ids,0]*normalized(s)[:,None]+inter+wo['train_nb'][ids]
        physical=K.mv(T,recv)*abs(s)[:,None]
        assert float(np.max(abs(normalized(s))**2))<=1+1e-12
    return K.pgrad(physical,x),gt,gr,float(loss.mean())
def service(wo,W,w,b):
    _,_,_,_,_,loss,prob=target_terms(wo,W,w,b,'test')
    power=np.sum(abs(K.mv(W,wo['test_x']))**2,axis=1)
    assert power.max()<=1+1e-12
    return dict(ce=float(loss.mean()),accuracy=float(np.mean((prob>=.5)==wo['test_lab'])),max_power=float(power.max()))

def mc_bank(seed,delta):
    # Original independent draws; their symmetric partners have the same marginal law.
    m=C['mc_independent'];n=C['probe_n'];d=dataset(seed,20,m*n,delta)
    return (d['inter']+d['nb']).reshape(m,n,4),d['nb'].reshape(m,n,4)
def probe(wo,W,w,b,T,arm,delta,bank):
    x,s,gx,gt,_,loss,_=target_terms(wo,W,w,b,'probe');_,ph=geometry(delta)
    gm=gt.copy() if arm=='CSI' else ph[0]*K.mm(T,gt)
    if arm=='CSI':noise=np.zeros((C['mc_independent'],4,3),complex)
    else:
        raw=bank[1] if arm=='single_user' else bank[0]
        noise=np.einsum('ab,rnb,n,ni->rai',T,raw,abs(s),x,optimize=True)/len(x)
    plus=gm[None,:,:]+noise;minus=gm[None,:,:]-noise
    samples=np.stack([plus,minus],axis=1).reshape(-1,4,3)
    empirical=samples.mean(0);iid=plus.mean(0)
    cos=cosine(empirical,gt);alpha=float(np.vdot(gt,empirical).real/norm(gt)**2)
    row=dict(expected_cosine=cos,analytic_cosine=cosine(gm,gt),iid_cosine=cosine(iid,gt),
        true_gradient_norm=norm(gt),expected_gradient_norm=norm(empirical),norm_ratio=norm(empirical)/norm(gt),
        signed_projection=alpha,positive_projection=max(alpha,0.),positive_projection_energy_fraction=max(cos,0.)**2,
        orthogonal_relative_residual=norm(empirical-alpha*gt)/norm(gt),
        empirical_analytic_relative_error=norm(empirical-gm)/norm(gm),
        iid_mean_relative_error=norm(iid-gm)/norm(gm),
        return_variance=float(np.mean(np.sum(abs(noise)**2,axis=(1,2)))),probe_ce=float(loss.mean()))
    assert row['empirical_analytic_relative_error']<1e-10
    return row,dict(true_gradient=gt,analytic_gradient=gm,empirical_gradient=empirical,iid_gradient=iid,mc_gradients=samples)

def train(wo,seed,delta,arm,folder,measure):
    W,w,b=initial(seed);T,mats=matrix(delta,arm);aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float)
    statesW=[W.copy()];statesR=[np.r_[w,b]];grads=[];truegrads=[];rgrads=[];trace=[];probes=[];parr={}
    initial_service=service(wo,W,w,b);bank=mc_bank(seed,delta) if measure else None
    def measure_at(step):
        row,arr=probe(wo,W,w,b,T,arm,delta,bank);probes.append(dict(step=step,**row))
        parr.update({f'step{step}_{k}':v for k,v in arr.items()})
    if measure:measure_at(0)
    for step in range(120):
        ids=np.arange(step*32,(step+1)*32);g,gt,gr,loss=gradients(wo,W,w,b,T,arm,delta,ids)
        old=W.copy();W=aw.step(W,g,.025);unprojected=norm(W-old)
        W*=min(1.,1/np.sqrt(3)/max(norm(W),1e-15))
        par=np.clip(ar.step(np.r_[w,b],gr,.06),-4,4);w=par[:1];b=float(par[1])
        assert np.isfinite(W).all() and np.isfinite(par).all()
        statesW.append(W.copy());statesR.append(par.copy());grads.append(g);truegrads.append(gt);rgrads.append(gr)
        trace.append(dict(step=step+1,ce=loss,gradient_norm=norm(g),true_norm=norm(gt),encoder_norm=norm(W),update_norm=norm(W-old),unprojected_update_norm=unprojected))
        if measure and step+1 in C['checkpoints']:measure_at(step+1)
    row=dict(seed=seed,delta=delta,arm=arm,initial_service=initial_service,service=service(wo,W,w,b),trace=trace,probes=probes)
    if measure:
        late=[p for p in probes if p['step'] in C['primary_checkpoints']]
        row['late']={k:float(np.mean([p[k] for p in late])) for k in late[0] if k!='step'}
    name=f'{seed}_{delta}_{arm}'
    np.savez_compressed(folder/f'model_{name}.npz',W=W,w=w,b=b,T=T,**mats)
    np.savez_compressed(folder/f'trace_{name}.npz',W=np.array(statesW),receiver=np.array(statesR),gradient=np.array(grads),true_gradient=np.array(truegrads),receiver_gradient=np.array(rgrads))
    if measure:np.savez_compressed(folder/f'probe_{name}.npz',**parr)
    print('TRAIN',name,row['service'],'late_cos',row.get('late',{}).get('expected_cosine'),flush=True)
    return row

def check_hash(file):
    for rel,h in json.loads((R/file).read_text())['hashes'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==h,rel

def checks():
    wo=world(PRE[0],15);W,w,b=initial(PRE[0]);ids=np.arange(23)
    _,_,_,gt,gr,_,_=target_terms(wo,W,w,b,'train',ids)
    def obj(W,w,b):return float(target_terms(wo,W,w,b,'train',ids)[5].mean()+.001*np.sum(w*w)/2)
    errors=[];e=1e-6
    for idx in np.ndindex(W.shape):
        for z in [1.,1j]:
            d=np.zeros_like(W);d[idx]=z
            errors.append(abs((obj(W+e*d,w,b)-obj(W-e*d,w,b))/(2*e)-np.vdot(gt,d).real))
    errors.extend([abs((obj(W,w+e,b)-obj(W,w-e,b))/(2*e)-gr[0]),abs((obj(W,w,b+e)-obj(W,w,b-e))/(2*e)-gr[1])])
    assert max(errors)<1e-7
    geom=[]
    for delta in C['angles']:
        a,ph=geometry(delta);T,m=matrix(delta,'ZF');P=m['P'];H=m['H'];F=m['F']
        alt=np.linalg.solve(H@H.conj().T+.03**2*np.eye(4),H)
        d=dict(delta=delta,overlap=float(abs(np.vdot(a[0],a[1]))/4),target_retained=norm(P@a[0])**2/4,
            analytic_ZF_cosine=norm(P@a[0])/2,projector_idempotence=norm(P@P-P),projector_hermitian=norm(P-P.conj().T),
            null_residual=norm(P@H[:,1]),rank=int(np.linalg.matrix_rank(P,tol=1e-10)),mmse_equivalence=norm(F-alt))
        assert max(d['projector_idempotence'],d['projector_hermitian'],d['null_residual'])<1e-12
        assert d['rank']==3 and d['mmse_equivalence']<1e-9
        geom.append(d)
    # Check the actual user-2 local-gradient sign symmetry, including its forward noise.
    d=dataset(PRE[0],20,257,15);W2,w2,b2=initial(PRE[0],5);lab=(d['x2'][:,0]*d['x2'][:,1]>0).astype(float)
    _,sp,*_=K.task(forward(d['h'][:,1],W2,d['x2'],d['nf2']),lab,w2,b2)
    _,sm,*_=K.task(forward(d['h'][:,1],W2,-d['x2'],-d['nf2']),lab,w2,b2)
    sym=norm(sp+sm);assert sym<1e-12
    return dict(finite_difference_max=max(errors),geometry=geom,interferer_antithetic_symmetry_error=sym)

def preflight():
    check_hash('freeze_A.json');dump(R/'results/numerical_checks.json',checks());rows=[]
    for seed in PRE:
        wo=world(seed,15);np.savez_compressed(R/f'results/preflight/world_{seed}_15.npz',**wo)
        for arm in ['S0','CSI']:
            row=train(wo,seed,15,arm,R/'results/preflight',False);rows.append(row)
            with (R/'results/preflight_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    means={m:{k:float(np.mean([r['service'][k] for r in rows if r['arm']==m])) for k in ['ce','accuracy']} for m in ['S0','CSI']}
    diff=means['S0']['ce']-means['CSI']['ce']
    gate=dict(means=means,delta_ce=diff,passed=diff>=.10,threshold=.10)
    dump(R/'results/preflight.json',gate);print('GATE',gate,flush=True)

def canonical(v):
    v=np.array(v,dtype=float);s=sorted(range(len(v)),key=lambda i:v[i]);out=v.copy();last=None
    for i in s:
        if last is None or v[i]-last>C['tie_tol']:last=v[i]
        out[i]=last
    return out

def summarize(rows):
    out=[]
    for delta in C['angles']:
        for arm in ARMS:
            rr=[r for r in rows if r['delta']==delta and r['arm']==arm]
            if len(rr)!=8:continue
            out.append(dict(delta=delta,arm=arm,accuracy=float(np.mean([r['service']['accuracy'] for r in rr])),
                ce=float(np.mean([r['service']['ce'] for r in rr])),**{k:float(np.mean([r['late'][k] for r in rr])) for k in rr[0]['late']}))
    return out

def verdicts(rows):
    cells=summarize(rows);lookup={(x['delta'],x['arm']):x for x in cells};claims={}
    oldpath=R/'results/claims.json'
    if oldpath.exists():claims=json.loads(oldpath.read_text())
    def record(key,support,falsified,**vals):
        if key in claims:return
        c=dict(verdict='archived' if falsified else ('archived' if support else 'archived'),support=bool(support),falsified=bool(falsified),**vals)
        claims[key]=c
        if falsified:dump(R/f'results/STOP_{key}.json',dict(time=time.time(),claim=key,result=c,completed_runs=len(rows),action='close claim immediately; remaining fixed-grid data only for other open claims, no new configurations'))
    inversions=[];complete=all((d,m) in lookup for d in C['angles'] for m in ['naive','ZF','MMSE']);strict=True
    for d in C['angles']:
        if not all((d,m) in lookup for m in ['naive','ZF','MMSE']):continue
        mm,nn,zz=[lookup[d,m] for m in ['MMSE','naive','ZF']]
        strict &= all(mm[k]>nn[k]+C['tie_tol'] and nn[k]>zz[k]+C['tie_tol'] for k in ['accuracy','expected_cosine'])
        for i,j in [('MMSE','naive'),('MMSE','ZF'),('naive','ZF')]:
            cd=lookup[d,i]['expected_cosine']-lookup[d,j]['expected_cosine'];ad=lookup[d,i]['accuracy']-lookup[d,j]['accuracy']
            if abs(cd)>C['tie_tol'] and abs(ad)>C['tie_tol'] and cd*ad<0:inversions.append(dict(delta=d,arm1=i,arm2=j,cosine_difference=cd,accuracy_difference=ad))
    if inversions or complete:record('G51.1',strict and complete,bool(inversions),inversions=inversions,strict_all_angles=bool(strict and complete))
    if len(cells)==15:
        z=float(np.mean([x['expected_cosine'] for x in cells if x['arm']=='ZF']));m=float(np.mean([x['expected_cosine'] for x in cells if x['arm']=='MMSE']))
        record('G51.2',z<=.70 and m>=.88,z>=.80,ZF_cosine=z,MMSE_cosine=m)
        cv=canonical([x['expected_cosine'] for x in cells]);av=canonical([x['accuracy'] for x in cells]);rho=float(spearmanr(cv,av).statistic)
        record('G51.3',rho>=.85,rho<.70,spearman=rho,n_cells=15,cosine_values=cv.tolist(),accuracy_values=av.tolist(),cosine_ranks=rankdata(cv).tolist(),accuracy_ranks=rankdata(av).tolist())
    dump(R/'results/summary.json',cells);dump(R/'results/claims.json',claims);return claims

def main():
    check_hash('freeze_B.json');assert json.loads((R/'results/preflight.json').read_text())['passed'];rows=[];powers=[]
    for delta in C['angles']:
        for seed in SEEDS:
            wo=world(seed,delta);np.savez_compressed(R/f'results/worlds/world_{seed}_{delta}.npz',**wo)
            sig=float(np.mean(np.sum(abs(wo['train_h'][:,0])**2,axis=1)));inter=float(np.mean(np.sum(abs(wo['train_inter'])**2,axis=1)))
            powers.append(dict(seed=seed,delta=delta,signal=sig,interference=inter,SIR_dB=float(10*np.log10(sig/inter))))
            for arm in ARMS:
                row=train(wo,seed,delta,arm,R/'results/models',True);rows.append(row)
                with (R/'results/main_rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print('CLAIMS',verdicts(rows),flush=True)
    dump(R/'results/all_runs.json',rows);dump(R/'results/power.json',powers)

if __name__=='__main__':
    stage=sys.argv[1];start=time.time();resource.setrlimit(resource.RLIMIT_CPU,(1200,1200))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall limit')));signal.alarm(1800)
    try:
        {'preflight':preflight,'main':main}[stage]()
        dump(R/f'results/runtime_{stage}.json',dict(pid=os.getpid(),seconds=time.time()-start,finished=time.time(),status='complete'))
    except BaseException:
        (R/f'logs/failure_{stage}_{int(time.time())}.txt').write_text(traceback.format_exc());raise
