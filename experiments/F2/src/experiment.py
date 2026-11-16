"""G42: deterministic geometry LAP, paired learning, exhaustive permutation endpoints."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[key]='1'
import sys,json,time,hashlib,itertools,traceback,platform,signal
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from scipy.special import j0
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr,rankdata
R=Path(__file__).resolve().parents[1]
C=json.loads((R/'configs/config.json').read_text())
np.seterr(over='raise',invalid='raise',divide='raise')
def dump(p,x): p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False))
def mm(a,b): return np.einsum('ij,jk->ik',a,b)
def cn(rng,shape): return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(2)
def rng(seed,n,stream):return np.random.default_rng(np.random.SeedSequence([seed,n,stream]))
def root(x):
 e,u=np.linalg.eigh((x+x.conj().T)/2);assert e.min()>-1e-10
 return mm(u*np.sqrt(np.maximum(e,0)),u.conj().T)
def check_freeze(name):
 for p,h in json.loads((R/name).read_text())['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h,p

def geometry(n,ang,d):
 if n==7:
  ph=np.arange(6)*np.pi/3;pos=np.vstack([np.zeros((1,2)),.5*np.array([np.cos(ph),np.sin(ph)]).T])
 else:pos=np.array([(.5*x,.5*y) for x in range(3) for y in range(3)])
 theta=np.arange(128)*2*np.pi/128;uv=np.array([np.cos(theta),np.sin(theta)])
 dv=d*np.array([np.cos(np.deg2rad(ang)),np.sin(np.deg2rad(ang))])
 s0=np.exp(-2j*np.pi*mm(pos,uv))/np.sqrt(128);st=s0*np.exp(2j*np.pi*np.einsum('i,ik->k',dv,uv))
 cov=mm(s0,s0.conj().T);cross=mm(st,s0.conj().T);a=np.linalg.solve(cov.T,cross.T).T
 q=cov-mm(mm(a,cov),a.conj().T);q=(q+q.conj().T)/2
 rt=root(cov);qr=root(q);jr=np.block([[rt,np.zeros((n,n))],[mm(a,rt),qr]])
 dist=np.linalg.norm(pos[:,None]-pos[None,:]-dv,axis=2)
 joint=np.block([[cov,cross.conj().T],[cross,cov]])
 err=max(float(np.max(abs(cross-j0(2*np.pi*dist)))),float(np.max(abs(cov-j0(2*np.pi*np.linalg.norm(pos[:,None]-pos[None,:],axis=2))))))
 ferr=float(np.max(abs(mm(jr,jr.conj().T)-joint)));assert max(err,ferr)<1e-10
 return dict(n=n,angle=ang,d=d,pos=pos,dv=dv,cov=cov,cross=cross,a=a,q=q,root=rt,qr=qr,jr=jr,dist=dist,bessel_error=err,factor_error=ferr,joint_min_eigenvalue=float(np.linalg.eigvalsh(joint).min()))

def assignments(g):
 n=g['n'];perms=np.array(list(itertools.permutations(range(n))),dtype=np.int16)
 score=np.sum(g['cross'].real[perms,np.arange(n)],axis=1)
 row,col=linear_sum_assignment(-g['cross'].real);opt=float(g['cross'].real[row,col].sum())
 lap=int(np.flatnonzero(score>=opt-1e-12)[0]);assert abs(score.max()-opt)<1e-10
 used=set();h=[]
 for i in range(n):
  j=min((j for j in range(n) if j not in used),key=lambda j:(g['dist'][j,i],j));used.add(j);h.append(j)
 heur=int(np.flatnonzero(np.all(perms==h,axis=1))[0]);identity=int(np.flatnonzero(np.all(perms==np.arange(n),axis=1))[0])
 return perms,score,dict(LAP=lap,heuristic=heur,S0=identity,optimum=opt)
def matrix(p):return np.eye(len(p))[p].T

def world(seed,g):
 n=g['n'];tr=rng(seed,n,1);k=C['train_samples']
 inp=tr.choice([-1.,1.],(k,3));f=np.einsum('ij,bjn->bin',g['root'],cn(tr,(k,n,2)))
 b=np.einsum('ij,bjn->bin',g['a'],f)+np.einsum('ij,bjn->bin',g['qr'],cn(tr,(k,n,2)))
 nf=.1*cn(tr,(k,n));nb=.03*cn(tr,(k,2))
 te=rng(seed,n,2);ti=te.choice([-1.,1.],(C['test_samples'],3));tf=np.einsum('ij,bjn->bin',g['root'],cn(te,(len(ti),n,2)));tn=.1*cn(te,(len(ti),n))
 cr=rng(seed,n,3);cf=np.einsum('ij,bjn->bin',g['root'],cn(cr,(64,n,2)));cb=np.einsum('ij,bjn->bin',g['a'],cf)+np.einsum('ij,bjn->bin',g['qr'],cn(cr,(64,n,2)))
 cf+=.1*cn(cr,cf.shape);cb+=.1*cn(cr,cb.shape)
 return dict(inp=inp,lab=(inp[:,0]*inp[:,1]>0).astype(float),f=f,b=b,nf=nf,nb=nb,ti=ti,tl=(ti[:,0]*ti[:,1]>0).astype(float),tf=tf,tn=tn,cf=cf,cb=cb)
def initial(seed,n,k):
 W=cn(rng(seed,n,4),(2,3));W*=.4/np.linalg.norm(W)
 return np.repeat(W[None],k,0),np.full((k,n),.8),np.full(k,-.4)

def evaluate(data,W,w,b):
 ce=np.zeros(len(W));acc=np.zeros(len(W));ne=len(data['ti'])
 for start in range(0,ne,256):
  sl=slice(start,start+256);x=np.einsum('kni,bi->kbn',W,data['ti'][sl]);y=np.einsum('bmn,kbn->kbm',data['tf'][sl],x)+data['tn'][sl]
  z=b[:,None]+np.einsum('kbm,km->kb',np.tanh(abs(y)**2),w)
  ce+=np.sum(np.logaddexp(0,z)-data['tl'][sl]*z,axis=1);acc+=np.sum((z>=0)==data['tl'][sl],axis=1)
 return ce/ne,acc/ne

def train(data,seed,n,mats=None,perms=None,csi=False):
 k=len(perms) if perms is not None else len(mats) if mats is not None else 1
 W,w,b=initial(seed,n,k);mw=np.zeros_like(W);vw=np.zeros(W.shape);mr=np.zeros((k,n+1));vr=mr.copy();trace=[]
 L=8*np.sqrt(n);maxpower=0.;maxrev=0.
 for step,start in enumerate(range(0,C['train_samples'],32),1):
  sl=slice(start,start+32);inp=data['inp'][sl];f=data['f'][sl];x=np.einsum('kni,bi->kbn',W,inp)
  maxpower=max(maxpower,float(np.max(np.sum(abs(x)**2,axis=2))))
  y=np.einsum('bmn,kbn->kbm',f,x)+data['nf'][sl];u=np.tanh(abs(y)**2)
  z=b[:,None]+np.einsum('kbm,km->kb',u,w);err=1/(1+np.exp(-z))-data['lab'][sl]
  delta=2*err[:,:,None]*w[:,None,:]*(1-u*u)*y
  gw=np.einsum('kb,kbm->km',err,u)/len(inp)+.001*w;gb=err.mean(1)
  if csi:gx=np.einsum('bmn,kbm->kbn',f.conj(),delta)
  else:
   if perms is not None:
    # Pi[perm[i],i]=1 => B^H Pi delta=sum_i conj(B[perm[i]]) delta[i].
    bp=data['b'][sl][:,perms,:].transpose(1,0,2,3)
    gx=np.einsum('kbmn,kbm->kbn',bp.conj(),delta)+L*data['nb'][sl]
    power=np.sum(abs(delta)**2,axis=2)/(L*L)
   else:
    sd=np.einsum('kmj,kbj->kbm',mats,delta)
    gx=np.einsum('bmn,kbm->kbn',data['b'][sl].conj(),sd)+L*data['nb'][sl]
    power=np.sum(abs(sd)**2,axis=2)/(L*L)
   maxrev=max(maxrev,float(power.max()))
  grad=np.einsum('kbn,bi->kni',gx,inp)/len(inp)
  mw=.9*mw+.1*grad;vw=.99*vw+.01*abs(grad)**2
  W-=.025*(mw/(1-.9**step))/(np.sqrt(vw/(1-.99**step))+1e-8)
  norms=np.sqrt(np.sum(abs(W)**2,axis=(1,2)));W*=np.minimum(1.,1/np.sqrt(3)/np.maximum(norms,1e-15))[:,None,None]
  gr=np.column_stack([gw,gb]);mr=.9*mr+.1*gr;vr=.99*vr+.01*gr*gr
  pars=np.clip(np.column_stack([w,b])-.06*(mr/(1-.9**step))/(np.sqrt(vr/(1-.99**step))+1e-8),-4,4);w=pars[:,:n];b=pars[:,n]
  if step==1 or step%20==0:trace.append(dict(step=step,mean_train_ce=np.mean(np.logaddexp(0,z)-data['lab'][sl]*z,axis=1).tolist()))
 assert maxpower<=1+1e-10 and maxrev<=1+1e-10 and np.isfinite(W).all() and np.isfinite(w).all()
 ce,acc=evaluate(data,W,w,b)
 return dict(W=W,w=w,b=b,ce=ce,accuracy=acc,max_forward_power=maxpower,max_reverse_power=maxrev,trace=trace)

def quant(x,bound):
 def q(v):return np.rint(np.clip(v,-bound,bound)/bound*32767)/32767*bound
 return q(x.real)+1j*q(x.imag)
def polar(data):
 f=quant(data['cf'],4);cross=np.einsum('bmn,bln->ml',data['cb'],f.conj())/128
 r=np.einsum('bmn,bln->ml',f,f.conj())/128-.01*np.eye(f.shape[1]);r=(r+r.conj().T)/2+.02*np.eye(f.shape[1]);e,u=np.linalg.eigh(r);reg=mm(u*np.maximum(e,.02),u.conj().T)
 a=np.linalg.solve(reg.T,cross.T).T;u,s,vh=np.linalg.svd(a);m0=mm(u,vh);u,_,vh=np.linalg.svd(quant(m0,32));m=mm(u,vh)
 assert np.linalg.norm(mm(m.conj().T,m)-np.eye(len(m)))<1e-10
 return m,dict(Ahat=a,Creg=reg,C10hat=cross,M=m,sv=s,channel_clips=np.count_nonzero(abs(data['cf'].real)>4)+np.count_nonzero(abs(data['cf'].imag)>4))

def stats(vals):
 v=np.asarray(vals);assert v.shape==(8,) and np.isfinite(v).all()
 ix=np.random.default_rng(C['bootstrap_seed']).integers(0,8,(20000,8))
 return dict(mean=float(v.mean()),ci95=np.quantile(v[ix].mean(1),[.025,.975]).tolist(),per_seed=v.tolist(),min=float(v.min()),max=float(v.max()))
def summarize(rows):return {m:{key:stats([r[key] for r in rows if r['method']==m]) for key in ['ce','accuracy']} for m in sorted(set(r['method'] for r in rows))}
def append(p,row):
 with p.open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
def rowout(out,index,seed,n,angle,d,method):
 return dict(seed=seed,n=n,angle=angle,d=d,method=method,ce=float(out['ce'][index]),accuracy=float(out['accuracy'][index]),max_forward_power=out['max_forward_power'],max_reverse_power=out['max_reverse_power'])

def precheck():
 check_freeze('freeze_precheck.json');records=[];csi={}
 for d in C['displacement_candidates']:
  cells=[]
  for n in [7,9]:
   for angle in [15,37,73]:
    g=geometry(n,angle,d);rows=[]
    for seed in C['precheck_seeds']:
     data=world(seed,g);o=train(data,seed,n,mats=np.eye(n)[None]);r=rowout(o,0,seed,n,angle,d,'S0');rows.append(r);append(R/'results/precheck_raw.jsonl',r)
     key=(n,seed)
     if key not in csi:csi[key]=train(data,seed,n,csi=True)
     r=rowout(csi[key],0,seed,n,angle,d,'CSI');rows.append(r);append(R/'results/precheck_raw.jsonl',r)
    s=summarize(rows);passed=s['S0']['accuracy']['mean']<=.70 and s['CSI']['accuracy']['mean']>=.90
    cells.append(dict(n=n,angle=angle,summary=s,passed=passed));print(json.dumps(dict(stage='precheck',d=d,n=n,angle=angle,S0=s['S0']['accuracy']['mean'],CSI=s['CSI']['accuracy']['mean'],passed=passed)),flush=True)
  passed=all(x['passed'] for x in cells);records.append(dict(d=d,cells=cells,passed=passed));dump(R/'results/precheck.json',dict(attempts=records,passed=passed,selected_d=d if passed else None))
  if passed or any(v['summary']['CSI']['accuracy']['mean']<.9 for v in cells):break

def endpoints(rows):
 by={m:[r for r in rows if r['method']==m] for m in ['S0','S_LAP','S_heuristic','polar_64','CSI']}
 s=summarize(rows);gain=stats([x['ce']-y['ce'] for x,y in zip(by['S0'],by['S_LAP'])]);gap=stats([x['ce']-y['ce'] for x,y in zip(by['S_LAP'],by['polar_64'])]);acc=s['S_LAP']['accuracy']['mean']
 return dict(summary=s,CE_gain=gain,CE_gap=gap,G421_falsified=bool(acc<.85 or gain['mean']<.15),G421_supported=bool(acc>=.92 and gain['mean']>=.30),G422_falsified=bool(gap['mean']>.03 and gap['ci95'][0]>0),G422_supported=bool(gap['mean']<=.02))

def formal():
 check_freeze('freeze_formal.json');gate=json.loads((R/'results/precheck.json').read_text());assert gate['passed'];d=gate['selected_d'];closed=set();events=[];all_primary=[];all_corr=[]
 for n in [7,9]:
  for angle in [15,37,73]:
   if len(closed)==3:break
   tag=f'n{n}_a{angle}';dest=R/'results'/tag;dest.mkdir(exist_ok=False);g=geometry(n,angle,d);perms,scores,ids=assignments(g)
   np.savez_compressed(dest/'geometry.npz',**g);np.savez_compressed(dest/'assignments.npz',perms=perms,scores=scores,**ids)
   dump(dest/'selection.json',dict(**ids,permutation_LAP=perms[ids['LAP']].tolist(),permutation_heuristic=perms[ids['heuristic']].tolist(),rho_LAP=float(scores[ids['LAP']]/n),rho_S0=float(scores[ids['S0']]/n),rho_heuristic=float(scores[ids['heuristic']]/n)))
   rows=[]
   if not {'G42.1','G42.2'}.issubset(closed):
    for seed in C['seeds']:
     data=world(seed,g);np.savez_compressed(dest/f'world_{seed}.npz',**data);m,cal=polar(data);np.savez_compressed(dest/f'polar_{seed}.npz',**cal)
     mats=np.stack([np.eye(n),matrix(perms[ids['LAP']]),matrix(perms[ids['heuristic']]),m]);out=train(data,seed,n,mats=mats)
     np.savez_compressed(dest/f'main_models_{seed}.npz',**{k:out[k] for k in ['W','w','b','ce','accuracy']},mats=mats)
     dump(dest/f'main_trace_{seed}.json',out['trace'])
     for i,method in enumerate(['S0','S_LAP','S_heuristic','polar_64']):
      r=rowout(out,i,seed,n,angle,d,method);rows.append(r);append(R/'results/formal_raw.jsonl',r)
     out=train(data,seed,n,csi=True);np.savez_compressed(dest/f'csi_model_{seed}.npz',**{k:out[k] for k in ['W','w','b','ce','accuracy']});r=rowout(out,0,seed,n,angle,d,'CSI');rows.append(r);append(R/'results/formal_raw.jsonl',r)
     print(json.dumps(dict(stage='main',cell=tag,seed=seed,rows=rows[-5:])),flush=True)
    p=endpoints(rows);p.update(n=n,angle=angle);all_primary.append(p);dump(R/'results/primary.json',all_primary)
    for line,key in [('G42.1','G421_falsified'),('G42.2','G422_falsified')]:
     if line not in closed and p[key]:closed.add(line);events.append(dict(line=line,cell=tag,action='falsified_stop',endpoint=p))
    dump(R/'results/stop_events.json',events)
   if 'G42.3' not in closed:
    destc=dest/'exhaustive';destc.mkdir();means=np.zeros(len(perms));ces=np.zeros(len(perms));accs=[];seed_rhos=[]
    for si,seed in enumerate(C['seeds']):
     data=world(seed,g);outpath=dest/f'world_{seed}.npz'
     if not outpath.exists():np.savez_compressed(outpath,**data)
     arr=np.zeros(len(perms));cearr=arr.copy();modelsW=np.zeros((len(perms),2,3),complex);modelsw=np.zeros((len(perms),n));modelsb=arr.copy();starttime=time.time()
     for start in range(0,len(perms),C['permutation_chunk']):
      end=min(start+C['permutation_chunk'],len(perms));o=train(data,seed,n,perms=perms[start:end]);arr[start:end]=o['accuracy'];cearr[start:end]=o['ce'];modelsW[start:end]=o['W'];modelsw[start:end]=o['w'];modelsb[start:end]=o['b']
      np.savez_compressed(destc/f's{seed}_chunk{start:06d}.npz',start=start,end=end,W=o['W'],w=o['w'],b=o['b'],accuracy=o['accuracy'],ce=o['ce'],max_forward_power=o['max_forward_power'],max_reverse_power=o['max_reverse_power'])
      if start%(C['permutation_chunk']*5)==0 or end==len(perms):print(json.dumps(dict(stage='exhaustive',cell=tag,seed=seed,completed=end,total=len(perms),seconds=time.time()-starttime)),flush=True)
     np.savez_compressed(destc/f'seed_{seed}.npz',accuracy=arr,ce=cearr,W=modelsW,w=modelsw,b=modelsb);accs.append(arr);seed_rhos.append(float(spearmanr(np.round(scores,12),arr).statistic));means+=arr/8;ces+=cearr/8
    accs=np.array(accs);rho=float(spearmanr(np.round(scores,12),means).statistic)
    # 256 seed bootstrap descriptors, no added decision gate; all permutations retained.
    boot=[];rr=np.random.default_rng(C['bootstrap_seed']);score_rank=rankdata(np.round(scores,12));score_rank-=score_rank.mean();score_norm=np.linalg.norm(score_rank)
    for _ in range(256):
     v=rankdata(np.mean(accs[rr.integers(0,8,8)],axis=0));v-=v.mean();boot.append(float(np.dot(score_rank,v)/(score_norm*np.linalg.norm(v))))
    co=dict(n=n,angle=angle,candidate_count=len(perms),seed_count=8,spearman=rho,per_seed_rhos=seed_rhos,descriptive_seed_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),falsified=rho<.70,supported=rho>=.85)
    np.savez_compressed(destc/'aggregate.npz',mean_accuracy=means,mean_ce=ces,accuracy_by_seed=accs,scores=scores);dump(destc/'correlation.json',co);all_corr.append(co);dump(R/'results/correlations.json',all_corr)
    print('CORRELATION '+json.dumps(co),flush=True)
    if co['falsified']:closed.add('G42.3');events.append(dict(line='G42.3',cell=tag,action='falsified_stop',endpoint=co));dump(R/'results/stop_events.json',events)
 dump(R/'results/completion.json',dict(closed=sorted(closed),completed_main_cells=[(p['n'],p['angle']) for p in all_primary],completed_correlation_cells=[(p['n'],p['angle']) for p in all_corr],finished=True))

if __name__=='__main__':
 stage=sys.argv[1];start=time.time()
 try:
  signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('4h stage budget')));signal.alarm(14400)
  {'precheck':precheck,'formal':formal}[stage]();dump(R/f'logs/{stage}_runtime.json',dict(seconds=time.time()-start,success=True,pid=os.getpid()))
 except BaseException:
  dump(R/f'logs/{stage}_failure.json',dict(traceback=traceback.format_exc(),seconds=time.time()-start));raise
