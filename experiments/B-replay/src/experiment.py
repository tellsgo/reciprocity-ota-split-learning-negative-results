import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import sys,json,pathlib,time,hashlib,traceback,resource,signal
import numpy as np
from scipy.special import j0
import primitives as K
R=pathlib.Path(__file__).resolve().parents[1]
ARMS=['fixed_S0','CSI','fas_max_snr','fas_slide','fas_slide_indep']
SEEDS=json.loads((R/'seeds.json').read_text())
N=7680; NT=8192; BATCH=32; D=.383

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def event(kind,**kw):
    item=dict(time=time.time(),kind=kind,**kw)
    with (R/'logs/events.jsonl').open('a') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    print(json.dumps(item,ensure_ascii=False),flush=True)
def rng(seed,stream):return np.random.default_rng(np.random.SeedSequence([seed,stream]))
def check_freeze(file):
    for rel,sha in json.loads((R/file).read_text())['hashes'].items():
        assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
def root(c):
    e,v=np.linalg.eigh(c);assert e.min()>-1e-10
    return (v*np.sqrt(np.maximum(e,0)))@v.conj().T
def geometry():
    pos=np.r_[0,D-np.arange(8)/7]
    cr=j0(2*np.pi*abs(pos[:,None]-pos[None,:]));ct=j0(np.pi*abs(np.arange(4)[:,None]-np.arange(4)[None,:]))
    return pos,cr,ct,root(cr),root(ct)
def fields(seed,stream,n):
    pos,cr,ct,rr,rt=geometry()
    z=K.cn(rng(seed,stream),(n,9,4))
    joint=np.einsum('ij,bjk,lk->bil',rr,z,rt)
    independent=np.einsum('bjk,lk->bjl',K.cn(rng(seed,stream+1),(n,8,4)),rt)
    return joint[:,0,:],joint[:,1:,:],independent
def initial(seed):
    w=K.cn(rng(seed,30),(4,3));w*=.4/np.linalg.norm(w)
    return w,np.array([.8]),-.4
def make_world(seed,sigma):
    inp=rng(seed,10).choice([-1.,1.],(N,3));f,b,bi=fields(seed,100,N)
    tf,_,_=fields(seed,200,NT);ti=rng(seed,20).choice([-1.,1.],(NT,3))
    return dict(inputs=inp,labels=(inp[:,0]*inp[:,1]>0).astype(float),f=f,b=b,bi=bi,
                nf=sigma*K.cn(rng(seed,12),(N,1)),nb=.03*K.cn(rng(seed,13),(N,4)),
                test_inputs=ti,test_labels=(ti[:,0]*ti[:,1]>0).astype(float),test_f=tf,
                test_n=sigma*K.cn(rng(seed,22),(NT,1)))
def pick(world,arm):
    if arm=='fas_max_snr':idx=np.argmax(np.sum(abs(world['b'])**2,axis=2),axis=1)
    else:idx=np.full(len(world['f']),0 if arm=='fixed_S0' else int(np.argmin(abs(D-np.arange(8)/7))),dtype=int)
    back=world['bi'] if arm=='fas_slide_indep' else world['b']
    return back[np.arange(len(idx)),idx],idx
def cosine(a,b):
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.vdot(b,a).real/den) if den>0 else 0.
def service(world,W,w,b):
    x=K.mv(W,world['test_inputs'])
    assert np.max(np.sum(abs(x)**2,axis=1))<=1+1e-10
    y=np.einsum('bt,bt->b',world['test_f'],x)[:,None]+world['test_n']
    losses,_,_,_,prob=K.task(y,world['test_labels'],w,b)
    return losses,prob
def train(world,seed,arm,phase):
    W,w,b=initial(seed);W0=W.copy();aw=K.Adam(W.shape,complex);ar=K.Adam((2,),float)
    back,indices=pick(world,arm);ests=[];trues=[];trace=[];clips=0
    for start in range(0,N,BATCH):
        sl=slice(start,start+BATCH);inp=world['inputs'][sl]
        y=np.einsum('bt,bt->b',world['f'][sl],K.mv(W,inp))[:,None]+world['nf'][sl]
        loss,s,gw,gb,_=K.task(y,world['labels'][sl],w,b)
        truth=K.pgrad(world['f'][sl].conj()*s,inp)
        if arm=='CSI':est=truth.copy()
        else:
            norm=np.linalg.norm(s,axis=1);tx=s/np.maximum(norm,1e-15)[:,None]
            gain,clip=K.qreal(norm,16.,False);clips+=clip
            gx=(back[sl].conj()*tx+world['nb'][sl])*gain[:,None]
            est=K.pgrad(gx,inp)
        ests.append(est);trues.append(truth);trace.append([start//BATCH+1,float(loss.mean()),cosine(est,truth)])
        W=aw.step(W,est,.025);W*=min(1.,1/np.sqrt(3)/max(np.linalg.norm(W),1e-15))
        rx=ar.step(np.r_[w,b],np.r_[gw,gb],.06);rx=np.clip(rx,-4,4);w=rx[:1];b=float(rx[1])
        assert np.isfinite(W).all()
    losses,prob=service(world,W,w,b);prediction=(prob>=.5)
    row=dict(seed=seed,phase=phase,arm=arm,accuracy=float(np.mean(prediction==world['test_labels'])),ce=float(losses.mean()),
             norm_clips=clips,trajectory_cosine=float(np.mean(np.array(trace)[:,2])),
             cosine_of_mean_gradients=cosine(np.mean(ests,axis=0),np.mean(trues,axis=0)))
    np.savez_compressed(R/'results/models'/f'{phase}_{seed}_{arm}.npz',W=W,w=w,b=b,W0=W0,estimated_gradients=ests,
                        true_gradients=trues,trace=trace,selected_ports=indices,losses=losses,probabilities=prob,predictions=prediction)
    dump(R/'results'/f'metric_{phase}_{seed}_{arm}.json',row);event('TRAIN',**row)
    return row
def stats(v):
    v=np.array(v);ids=rng(SEEDS['bootstrap'],800).integers(0,len(v),(20000,len(v)))
    return dict(mean=float(v.mean()),ci95=np.quantile(v[ids].mean(1),[.025,.975]).tolist(),per_seed=v.tolist())
def summaries(rows):
    return {a:{m:stats([r[m] for r in rows if r['arm']==a]) for m in ['accuracy','ce','trajectory_cosine']} for a in dict.fromkeys(r['arm'] for r in rows)}
def diagnostic(seed,arm):
    file=R/'results/worlds'/f'diagnostic_{seed}.npz'
    if not file.exists():
        f,b,bi=fields(seed,300,16384);np.savez_compressed(file,f=f,b=b,bi=bi)
    with np.load(file) as z:world={k:z[k] for k in z.files}
    back,idx=pick(world,arm)
    cross=np.sum(world['f'].conj()*back,axis=1)/4
    fp=np.sum(abs(world['f'])**2,axis=1)/4;bp=np.sum(abs(back)**2,axis=1)/4
    row=dict(seed=seed,arm=arm,cross_real=float(cross.real.mean()),cross_imag=float(cross.imag.mean()),
             normalized_correlation_real=float(cross.real.mean()/np.sqrt(fp.mean()*bp.mean())),forward_power=float(fp.mean()),
             backward_power=float(bp.mean()),negative_cross_fraction=float(np.mean(cross.real<0)),port_counts=np.bincount(idx,minlength=8).tolist())
    dump(R/'results'/f'diagnostic_{seed}_{arm}.json',row)
    return row
def verify():
    seed=SEEDS['verification'];rr=rng(seed,400);W,w,b=initial(seed);w=np.array([.7]);inp=rr.choice([-1.,1.],(32,3));lab=(inp[:,0]*inp[:,1]>0).astype(float)
    f=K.cn(rr,(32,4));noise=.1*K.cn(rr,(32,1))
    def obj(W,w,b):
        y=np.einsum('bt,bt->b',f,K.mv(W,inp))[:,None]+noise
        return K.task(y,lab,w,b)[0].mean()+.001*np.sum(w*w)/2
    y=np.einsum('bt,bt->b',f,K.mv(W,inp))[:,None]+noise
    _,s,gw,gb,_=K.task(y,lab,w,b);grad=K.pgrad(f.conj()*s,inp);err=[];eps=1e-6
    for ix in np.ndindex(W.shape):
        for part in [1,1j]:
            e=np.zeros_like(W);e[ix]=part
            err.append(abs((obj(W+eps*e,w,b)-obj(W-eps*e,w,b))/(2*eps)-np.vdot(grad,e).real))
    err.extend([abs((obj(W,w+eps,b)-obj(W,w-eps,b))/(2*eps)-gw[0]),abs((obj(W,w,b+eps)-obj(W,w,b-eps))/(2*eps)-gb)])
    pos,cr,ct,rc,tc=geometry();angle=np.arange(128)*2*np.pi/128
    angular=np.mean(np.exp(2j*np.pi*(pos[:,None,None]-pos[None,:,None])*np.cos(angle)),axis=2)
    out=dict(fd_checks=len(err),fd_max_error=float(max(err)),Cr_factor_error=float(np.linalg.norm(rc@rc.conj().T-cr)),
             Ct_factor_error=float(np.linalg.norm(tc@tc.conj().T-ct)),angular_error=float(np.max(abs(angular-cr))),
             fixed_j0=float(j0(2*np.pi*D)),slide_port=int(np.argmin(abs(pos[1:]))),slide_residual=float(pos[1:][np.argmin(abs(pos[1:]))]),
             slide_j0=float(j0(2*np.pi*min(abs(pos[1:])))),minimum_eigenvalue=float(np.linalg.eigvalsh(cr).min()))
    assert max(err)<1e-7 and out['Cr_factor_error']<1e-10 and out['Ct_factor_error']<1e-10 and out['angular_error']<1e-12
    assert out['slide_port']==3
    np.savez_compressed(R/'results/geometry.npz',positions=pos,Cr=cr,Ct=ct,Cr_root=rc,Ct_root=tc)
    dump(R/'results/numerical_checks.json',out);return out
def stop(claim,reason):
    dump(R/'results'/f'STOP_{claim}.json',dict(time=time.time(),claim=claim,reason=reason,action='archived'))
    event('STOP',claim=claim,reason=reason)
def decide(rows,which):
    ss=summaries(rows);acc=lambda arm:ss[arm]['accuracy']['mean'];ce=lambda arm:ss[arm]['ce']['mean']
    if which==1:
        bad=acc('fas_max_snr')>=.82;good=acc('fas_max_snr')<=.75 and acc('fas_max_snr')-acc('fixed_S0')<.08
    elif which==2:
        gap=acc('fas_slide')-acc('fas_max_snr');bad=acc('fas_slide')<.80 or gap<.10
        good=acc('fas_slide')>=.88 and (gap>=.15 or ce('fas_max_snr')-ce('fas_slide')>=.15)
    else:
        bad=acc('fas_slide_indep')>=.78;good=acc('fas_slide_indep')<.70 and acc('fas_slide')-acc('fas_slide_indep')>=.20
    out=dict(claim=f'G49.{which}',falsified=bool(bad),numerical_target_met=bool(good),verdict='archived' if bad else 'archived' if good else 'archived，archived',summary=ss)
    dump(R/'results'/f'G49.{which}.json',out)
    if bad:stop(f'G49.{which}',out['verdict'])
    event('DECISION',claim=out['claim'],verdict=out['verdict'])
def main():
    stage=sys.argv[1];started=time.time();check_freeze('preregistration.json' if stage=='precheck' else 'freeze_manifest.json')
    event('START',stage=stage,pid=os.getpid(),python=sys.version,numpy=np.__version__)
    if stage=='precheck':
        verify();attempts=[]
        for ci,sigma in enumerate([.10]):
            rows=[];phase=f'precheck{ci}'
            for seed in SEEDS['precheck']:
                world=make_world(seed,sigma);np.savez_compressed(R/'results/worlds'/f'{phase}_{seed}.npz',**world)
                for arm in ['fixed_S0','CSI']:rows.append(train(world,seed,arm,phase))
            ss=summaries(rows);passed=ss['fixed_S0']['ce']['mean']-ss['CSI']['ce']['mean']>=.10
            attempts.append(dict(candidate=ci,sigma=sigma,passed=passed,summary=ss,rows=rows))
            dump(R/'results/precheck.json',dict(passed=passed,selected_sigma=sigma if passed else None,attempts=attempts))
            event('GATE',phase=phase,passed=passed,summary=ss)
            if passed:break
        if not passed:stop('GLOBAL_PRECHECK','archived CE archived，archived')
    else:
        pre=json.loads((R/'results/precheck.json').read_text());assert pre['passed'];sigma=pre['selected_sigma'];rows=[];worlds={}
        for seed in SEEDS['main']:
            world=make_world(seed,sigma);worlds[seed]=world;np.savez_compressed(R/'results/worlds'/f'main_{seed}.npz',**world)
        for arm in ARMS:
            for seed in SEEDS['main']:
                if arm!='CSI':diagnostic(seed,arm)
                rows.append(train(worlds[seed],seed,arm,'main'))
                dump(R/'results/main_partial.json',rows)
            if arm=='CSI':
                ss=summaries(rows);passed=ss['fixed_S0']['ce']['mean']-ss['CSI']['ce']['mean']>=.10
                dump(R/'results/main_gate.json',dict(passed=passed,summary=ss))
                event('MAIN_BASELINE_DIAGNOSTIC',delta_ce=ss['fixed_S0']['ce']['mean']-ss['CSI']['ce']['mean'],note='archived，archived')
            elif arm=='fas_max_snr':decide(rows,1)
            elif arm=='fas_slide':decide(rows,2)
            elif arm=='fas_slide_indep':decide(rows,3)
        dump(R/'results/main.json',dict(rows=rows,summary=summaries(rows)))
    dump(R/'logs'/f'{stage}_completion.json',dict(pid=os.getpid(),finished=time.time(),wall_seconds=time.time()-started,status='complete'))
    event('DONE',stage=stage,wall_seconds=time.time()-started)
if __name__=='__main__':
    resource.setrlimit(resource.RLIMIT_CPU,(900,910));signal.alarm(1200)
    try:main()
    except BaseException:
        path=R/'logs'/f'failure_{time.time_ns()}.txt';path.write_text(traceback.format_exc());raise
