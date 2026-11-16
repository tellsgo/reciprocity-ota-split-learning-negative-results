"""Post-run independent algebra/trajectory replay. No training or parameter changes."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import pathlib,json,csv,hashlib,sys,time
import numpy as np
from scipy.stats import rankdata,spearmanr
import g51 as G
R=G.R;C=G.C
def dump(p,v):G.dump(p,v)
def metric(world,W,par,prefix):
 x=world[prefix+'_x'];h=world[prefix+'_h'][:,0];lab=world[prefix+'_lab'];nf=world[prefix+'_nf']
 y=np.sum(h.conj()*np.einsum('ai,ni->na',W,x),axis=1)+nf
 u=np.tanh(abs(y)**2);z=par[1]+par[0]*u;prob=1/(1+np.exp(-z))
 loss=np.logaddexp(0,z)-lab*z
 s=2*(prob-lab)*par[0]*(1-u*u)*y
 return dict(ce=float(loss.mean()),accuracy=float(np.mean((prob>=.5)==lab))),np.einsum('na,n,ni->ai',h,s,x)/len(x),s

def replay(world,tr,T,arm,delta):
 nstep=120;x=world['train_x'].reshape(nstep,32,3);h=world['train_h'][:,0].reshape(nstep,32,4)
 nf=world['train_nf'].reshape(nstep,32);lab=world['train_lab'].reshape(nstep,32)
 W=tr['W'][:-1];rr=tr['receiver'][:-1]
 y=np.sum(h.conj()*np.einsum('tai,tni->tna',W,x),axis=2)+nf
 u=np.tanh(abs(y)**2);z=rr[:,1,None]+rr[:,0,None]*u;pr=1/(1+np.exp(-z));s=2*(pr-lab)*rr[:,0,None]*(1-u*u)*y
 true=h*s[:,:,None];gt=np.einsum('tna,tni->tai',true,x)/32
 gr=np.column_stack((np.mean((pr-lab)*u,axis=1)+.001*rr[:,0],np.mean(pr-lab,axis=1)))
 if arm=='CSI':phys=true
 else:
  phi=np.exp(2j*np.pi*.383*np.cos(np.deg2rad(30)))
  unit=np.divide(s,abs(s),out=np.zeros_like(s),where=abs(s)>0)
  inter=0 if arm=='single_user' else world['train_inter'].reshape(nstep,32,4)
  recv=phi*h*unit[:,:,None]+inter+world['train_nb'].reshape(nstep,32,4)
  phys=np.einsum('ab,tnb->tna',T,recv)*abs(s)[:,:,None]
 grad=np.einsum('tna,tni->tai',phys,x)/32
 errs=dict(gradient=float(np.max(abs(grad-tr['gradient']))),true_gradient=float(np.max(abs(gt-tr['true_gradient']))),receiver_gradient=float(np.max(abs(gr-tr['receiver_gradient']))))
 m=np.zeros((4,3),complex);v=np.zeros((4,3));mr=np.zeros(2);vr=np.zeros(2);errw=0.;errr=0.
 for t in range(120):
  m=.9*m+.1*grad[t];v=.99*v+.01*abs(grad[t])**2
  wp=tr['W'][t]-.025*(m/(1-.9**(t+1)))/(np.sqrt(v/(1-.99**(t+1)))+1e-8)
  wp*=min(1.,1/np.sqrt(3)/float(np.linalg.norm(wp)))
  mr=.9*mr+.1*gr[t];vr=.99*vr+.01*gr[t]**2
  rp=np.clip(tr['receiver'][t]-.06*(mr/(1-.9**(t+1)))/(np.sqrt(vr/(1-.99**(t+1)))+1e-8),-4,4)
  errw=max(errw,float(np.max(abs(wp-tr['W'][t+1]))));errr=max(errr,float(np.max(abs(rp-tr['receiver'][t+1]))))
 errs.update(encoder_update=errw,receiver_update=errr)
 return errs

def main():
 start=time.time();G.check_hash('freeze_A.json');G.check_hash('freeze_B.json')
 rows=json.loads((R/'results/all_runs.json').read_text());pre=[json.loads(l) for l in (R/'results/preflight_rows.jsonl').read_text().splitlines()]
 assert len(rows)==120 and len(pre)==16
 assert {(r['seed'],r['delta'],r['arm']) for r in rows}=={(s,d,m) for s in C['seeds'] for d in C['angles'] for m in C['arms']}
 errors=[];mic=[];minimum_gradient=1e9;maxpow=0;world_cache={};bank_cache={}
 for row in pre+rows:
  seed,d,arm=row['seed'],row['delta'],row['arm'];ispre=seed in C['preflight_seeds'];folder=R/'results'/('preflight' if ispre else 'models')
  key=seed,d
  if key not in world_cache:
   wp=R/f'results/preflight/world_{seed}_{d}.npz' if ispre else R/f'results/worlds/world_{seed}_{d}.npz'
   world_cache[key]=dict(np.load(wp))
   rebuilt=G.world(seed,d);assert all(np.array_equal(v,rebuilt[k]) for k,v in world_cache[key].items())
  wo=world_cache[key];name=f'{seed}_{d}_{arm}';tr=dict(np.load(folder/f'trace_{name}.npz'));md=dict(np.load(folder/f'model_{name}.npz'))
  metric_end,_,_=metric(wo,md['W'],np.r_[md['w'],md['b']],'test')
  ee=max(abs(metric_end[k]-row['service'][k]) for k in ['accuracy','ce'])
  e=replay(wo,tr,md['T'],arm,d);e.update(endpoint=ee,final_state=float(np.max(abs(tr['W'][-1]-md['W']))))
  errors.append(dict(seed=seed,delta=d,arm=arm,**e));maxpow=max(maxpow,row['service']['max_power'])
  assert max(e.values())<1e-10,e
  if ispre:continue
  if key not in bank_cache:bank_cache[key]=G.mc_bank(seed,d)
  pp=dict(np.load(folder/f'probe_{name}.npz'))
  for p in row['probes']:
   t=p['step'];_,gt,s=metric(wo,tr['W'][t],tr['receiver'][t],'probe')
   phi=np.exp(2j*np.pi*.383*np.cos(np.deg2rad(30)))
   gm=gt if arm=='CSI' else phi*np.einsum('ab,bi->ai',md['T'],gt)
   empirical=pp[f'step{t}_mc_gradients'].mean(0)
   raw=bank_cache[key][1] if arm=='single_user' else bank_cache[key][0]
   noise=np.zeros((128,4,3),complex) if arm=='CSI' else np.einsum('ab,rnb,n,ni->rai',md['T'],raw,abs(s),wo['probe_x'])/C['probe_n']
   arr=pp[f'step{t}_mc_gradients'];replay_error=max(float(np.max(abs(arr[::2]-(gm+noise)))),float(np.max(abs(arr[1::2]-(gm-noise)))))
   cos=float(np.vdot(gt,empirical).real/(np.linalg.norm(gt)*np.linalg.norm(empirical)))
   er=max(float(np.max(abs(gt-pp[f'step{t}_true_gradient']))),float(np.max(abs(gm-pp[f'step{t}_analytic_gradient']))),float(np.max(abs(empirical-gm))))
   assert er<1e-10 and replay_error<1e-10 and abs(cos-p['expected_cosine'])<1e-10
   minimum_gradient=min(minimum_gradient,float(np.linalg.norm(gt)))
   mic.append(dict(seed=seed,delta=d,arm=arm,step=t,gradient_error=er,mc_replay_error=replay_error,cosine_error=abs(cos-p['expected_cosine'])))
 print('REPLAY PASS',len(errors),'models',len(mic),'probes',flush=True)
 hashes=json.loads((R/'READONLY_HASHES_BEFORE.json').read_text());root=R.parents[1]
 after={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in hashes};assert after==hashes
 dump(R/'READONLY_HASHES_AFTER.json',after)
 # Statistics from the already completed frozen data. No more training.
 cells=G.summarize(rows);look={(r['seed'],r['delta'],r['arm']):r for r in rows}
 A=np.array([[look[s,x['delta'],x['arm']]['service']['accuracy'] for x in cells] for s in C['seeds']])
 V=np.array([[look[s,x['delta'],x['arm']]['late']['expected_cosine'] for x in cells] for s in C['seeds']])
 E=np.array([[look[s,x['delta'],x['arm']]['service']['ce'] for x in cells] for s in C['seeds']])
 idx=np.random.default_rng(C['bootstrap_seed']).integers(0,8,(10000,8));ab=A[idx].mean(1);eb=E[idx].mean(1)
 # Each expected cosine is a deterministic geometry-only value; fix its exact theoretical tie groups.
 v=G.canonical(V.mean(0));cr=rankdata(v);ar=rankdata(ab,axis=1)
 cr=cr-cr.mean();ar=ar-ar.mean(axis=1)[:,None]
 rb=np.einsum('bi,i->b',ar,cr)/np.sqrt(np.sum(ar*ar,axis=1)*np.sum(cr*cr))
 rho=float(spearmanr(v,G.canonical(A.mean(0))).statistic)
 assert abs(rho-json.loads((R/'results/claims.json').read_text())['G51.3']['spearman'])<1e-12
 stats=[]
 for j,cell in enumerate(cells):stats.append(dict(**cell,accuracy_ci95=np.quantile(ab[:,j],[.025,.975]).tolist(),ce_ci95=np.quantile(eb[:,j],[.025,.975]).tolist(),accuracy_min=float(A[:,j].min()),accuracy_max=float(A[:,j].max())))
 contrasts=[]
 for d in C['angles']:
  pos={x['arm']:j for j,x in enumerate(cells) if x['delta']==d}
  for aa,bb in [('ZF','naive'),('single_user','ZF'),('MMSE','ZF'),('MMSE','single_user')]:
   dif=A[:,pos[aa]]-A[:,pos[bb]];bs=dif[idx].mean(1)
   contrasts.append(dict(delta=d,arm1=aa,arm2=bb,mean=float(dif.mean()),ci95=np.quantile(bs,[.025,.975]).tolist(),per_seed=dif.tolist(),positive_seeds=int((dif>0).sum()),negative_seeds=int((dif<0).sum())))
 raw=[];probes=[]
 for row in rows:
  raw.append(dict(seed=row['seed'],delta=row['delta'],arm=row['arm'],**row['service'],**row['late']))
  for p in row['probes']:probes.append(dict(seed=row['seed'],delta=row['delta'],arm=row['arm'],**p))
 for name,data in [('by_seed.csv',raw),('probes.csv',probes),('replay.csv',errors),('probe_replay.csv',mic)]:
  with (R/'results'/name).open('w') as f:
   wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows(data)
 cp=[]
 for d in C['angles']:
  for arm in C['arms']:
   rr=[r for r in rows if r['delta']==d and r['arm']==arm]
   for t in C['checkpoints']:
    pp=[next(p for p in r['probes'] if p['step']==t) for r in rr]
    cp.append(dict(delta=d,arm=arm,step=t,probe_ce=float(np.mean([p['probe_ce'] for p in pp])),true_gradient_norm=float(np.mean([p['true_gradient_norm'] for p in pp])),cosine=float(np.mean([p['expected_cosine'] for p in pp])),iid_cosine=float(np.mean([p['iid_cosine'] for p in pp])),iid_relative_error=float(np.mean([p['iid_mean_relative_error'] for p in pp])),last_update=float(np.mean([r['trace'][max(t-1,0)]['update_norm'] for r in rr])) if t else None))
 subs=[j for j,x in enumerate(cells) if x['arm'] in ['naive','ZF','MMSE']]
 cor=dict(primary=rho,ci95=np.quantile(rb,[.025,.975]).tolist(),bootstrap_replicates=10000,
          three_strategy_cells=float(spearmanr(G.canonical(V.mean(0)[subs]),G.canonical(A.mean(0)[subs])).statistic),
          seed_level=float(spearmanr(G.canonical(V.ravel()),G.canonical(A.ravel())).statistic),
          note='15 cell primary; paired bootstrap resamples 8 independent seeds, preserves geometry and theory-based cosine ties; no independent-cell p value')
 np.savez_compressed(R/'results/bootstrap.npz',indices=idx,accuracy=ab,ce=eb,spearman=rb)
 dump(R/'results/statistics.json',stats);dump(R/'results/contrasts.json',contrasts);dump(R/'results/correlations.json',cor);dump(R/'results/checkpoint_summary.json',cp)
 maxerrs={k:max(e[k] for e in errors) for k in ['gradient','true_gradient','receiver_gradient','encoder_update','receiver_update','endpoint','final_state']}
 ver=dict(status='PASS',models=len(errors),main_models=len(rows),preflight_models=len(pre),updates=len(errors)*120,probes=len(mic),max_errors=maxerrs,
          mc_replay_max=max(x['mc_replay_error'] for x in mic),minimum_probe_true_gradient_norm=minimum_gradient,max_test_tx_power=maxpow,
          all_worlds_reproduced=True,worlds=len(world_cache),frozen_hashes_match=True,readonly_hashes_match=True,seconds=time.time()-start)
 dump(R/'results/verification.json',ver);print(json.dumps(ver,indent=2),flush=True);print('CORRELATION',cor,flush=True)
if __name__=='__main__':main()
