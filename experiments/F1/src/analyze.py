"""Post-run integrity checks, fixed-endpoint summaries and standalone figures."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'
import json, hashlib, csv, datetime
from pathlib import Path
import numpy as np
import g37 as G
R=G.R; C=G.C; B=G.B; K=G.K

def load(p): return json.loads((R/p).read_text())

def check():
    G.verify_freeze('freeze_B.json')
    a=load('freeze_A.json')
    for p,h in a['hashes'].items():
        target='SPEC_A.md' if p=='SPEC.md' else p
        assert hashlib.sha256((R/target).read_bytes()).hexdigest()==h,p
    W=R.parents[1]; before=load('READONLY_HASHES_BEFORE.json')
    after={p:hashlib.sha256((W/p).read_bytes()).hexdigest() for p in before}
    assert before==after; G.dump(R/'READONLY_HASHES_AFTER.json',after)
    rows=load('results/all_runs.json')
    raw=[json.loads(s) for s in (R/'results/formal_raw.jsonl').read_text().splitlines()]
    pre=[json.loads(s) for s in (R/'results/preflight_raw.jsonl').read_text().splitlines()]
    assert rows==raw
    assert len(rows)==120 and len(pre)==48
    assert len({(r['method'],r['angle'],r['seed']) for r in rows})==len(rows)
    assert len({(r['method'],r['angle'],r['seed']) for r in pre})==len(pre)
    gs={a:G.geometry(.5,a,False)[0] for a in C['angles']}
    errors=[]; polarerrors=[]; powers=[]; initial={}; csi_by_seed={}
    for r in pre+rows:
        g=gs[r['angle']]; w=B.World(r['seed'],g)
        Wf=np.array(r['final_W_real'])+1j*np.array(r['final_W_imag'])
        wr=np.array(r['final_receiver_w']); br=r['final_receiver_b']
        # Independent direct forward/CE path, not World.service or task.
        x=np.einsum('ni,bi->bn',Wf,w.test_inputs)
        energy=np.sum(abs(x)**2,axis=1); powers.append(float(energy.max()))
        assert energy.max()<=1+1e-12
        y=np.einsum('bmn,bn->bm',w._test_f,x)+w._test_n
        logits=np.einsum('bm,m->b',np.tanh(abs(y)**2),wr)+br
        ce=float(np.mean(np.logaddexp(0,logits)-w.test_labels*logits))
        acc=float(np.mean((logits>=0)==w.test_labels))
        errors.append(abs(ce-r['service']['ce']))
        assert abs(ce-r['service']['ce'])<1e-12 and acc==r['service']['accuracy']
        assert np.linalg.norm(Wf)<=1/np.sqrt(3)+1e-12
        assert max(abs(wr).max(),abs(br))<=4+1e-12
        assert r['max_reverse_energy']<=1+1e-12
        assert r['gradient_or_loss_quant_clips']==0 and r['gain_quant_clips']==0
        assert all(np.isfinite([v['used_vs_current_cosine'],v['noiseless_return_vs_current_cosine']]).all() for v in r['gradient_diagnostics'])
        neg=float(np.mean([v['noiseless_return_vs_current_cosine']<0 for v in r['gradient_diagnostics']]))
        assert neg==r['noiseless_negative_fraction']
        key=(r['phase'],r['seed'])
        iw=(r['initial_W_real'],r['initial_W_imag'],r['initial_service'])
        if key in initial: assert initial[key]==iw
        else: initial[key]=iw
        co=r['costs']
        if r['method']=='CSI':
            assert co['training_labels']==2666 and co['rf_observations']==7998 and co['control_bits'] is None
            assert len(r['gradient_diagnostics'])==84
            csi_by_seed[key]=r
        elif r['method']=='polar_64':
            assert co['training_labels']==3712 and co['rf_observations']==8000 and co['control_bits']==89632
            assert len(r['gradient_diagnostics'])==116
            assert r['calibration']['channel_clips']==0 and r['calibration']['matrix_clips']==0
            saved=np.load(R/f"results/calibration_a{r['angle']}_s{r['seed']}.npz")
            assert np.array_equal(saved['cal_fhat'],w.cal_fhat) and np.array_equal(saved['cal_bhat'],w.cal_bhat)
            # Independent regularized covariance and polar construction.
            half=32767
            f=np.rint(np.clip(w.cal_fhat.real,-4,4)/4*half)/half*4+1j*np.rint(np.clip(w.cal_fhat.imag,-4,4)/4*half)/half*4
            fflat=f.transpose(1,0,2).reshape(7,-1)
            bflat=w.cal_bhat.transpose(1,0,2).reshape(7,-1)
            cc=fflat@fflat.conj().T/128-.01*np.eye(7)
            cc=(cc+cc.conj().T)/2+.02*np.eye(7)
            e,v=np.linalg.eigh(cc); cc=(v*np.maximum(e,.02))@v.conj().T
            cross=bflat@fflat.conj().T/128
            aa=cross@np.linalg.inv(cc)
            u,_,vh=np.linalg.svd(aa); m=u@vh
            mq=np.rint(np.clip(m.real,-32,32)/32*half)/half*32+1j*np.rint(np.clip(m.imag,-32,32)/32*half)/half*32
            u,_,vh=np.linalg.svd(mq); m=u@vh
            er=float(np.linalg.norm(m-B.unpack(r['actual_effective_M']))); polarerrors.append(er)
            assert er<1e-9
        else:
            assert co['training_labels']==3840 and co['rf_observations']==7680 and co['control_bits']==0
            assert len(r['gradient_diagnostics'])==120
            assert abs(r['restored_noise_variance_per_complex_coordinate']-.4032)<1e-12
            expected=np.eye(7) if r['method']=='S0' else g['M']
            assert np.array_equal(B.unpack(r['actual_effective_M']),expected)
    # Exact reuse of CSI across directions is justified by identical forward worlds.
    s=C['seeds'][0]; w0=B.World(s,gs[0])
    for angle,g in gs.items():
        w=B.World(s,g)
        for name in ['_f','_nf','_test_f','_test_n','inputs','labels']:
            assert np.array_equal(getattr(w0,name),getattr(w,name)),(angle,name)
    events=load('results/stop_events.json')
    groups={}
    for r in rows: groups.setdefault((r['method'],r['angle']),[]).append(r)
    for rr in groups.values(): assert [r['seed'] for r in rr]==C['seeds']
    for ev in events:
        rr=groups[ev['method'],ev['angle']]; ss=G.summary(rr)
        assert ss==ev['summary']
        base=groups['S0',ev['angle']]; csi=groups['CSI',0]
        gain=B.stats([a['service']['ce']-b['service']['ce'] for a,b in zip(base,rr)])
        gap=B.stats([a['service']['ce']-b['service']['ce'] for a,b in zip(rr,csi)])
        assert gain==ev['CE_gain_vs_S0'] and gap==ev['CE_gap_vs_CSI']
        ac=ss['accuracy']['mean']
        hit=(ev['angle']==60 and (ac<.85 or gain['mean']<.20)) if ev['line']=='G37.1' else (ev['angle']==30 and ac>=.75) if ev['line']=='G37.2' else ac<.90
        assert hit==ev['triggered']
        if hit:
            assert all((ev['method'],a) not in groups for a in ev['skipped_angles'])
    assert [('G37.1',60),('G37.1',0),('G37.1',120),('G37.2',30),('G37.3',60),('G37.3',0),('G37.3',120),('G37.3',30),('G37.3',90)]==[(e['line'],e['angle']) for e in events]
    # Independent continuous geometry argument: centroid obstruction plus rotation check.
    p=gs[0]['pos']; t=np.pi/3; rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    rotated=p@rot.T
    assert np.max(np.min(np.linalg.norm(rotated[:,None]-p[None,:],axis=2),axis=1))<1e-12
    for angle,g in gs.items():
        assert np.linalg.norm((p+g['dv']).sum(0)-p.sum(0))>3.49
        if angle in [0,60,120]:
            assert int(g['T'].sum())==4
            proj=g['T'].T@g['T']
            assert np.linalg.norm(g['st'].conj().T@g['T']-g['s0'].conj().T@proj)<1e-12
        else: assert g['T'].sum()==0
    out=dict(passed=True,preflight_rows=len(pre),formal_rows=len(rows),unique_rows=True,all_eight_seeds_present=True,
        max_test_CE_reconstruction_error=max(errors),test_accuracy_exact_match=True,max_independent_polar_error=max(polarerrors),
        max_test_tx_energy=max(powers),power_and_parameter_projections=True,no_quantization_clipping=True,shared_initialization_and_forward_world=True,
        finite_gradients=True,budgets_checked=True,stopping_replayed=True,CSI_reuse_exact=True,frozen_A_and_B_hashes_match=True,readonly_hashes_match=True,
        rotation_valid_translation_invalid=True,failed_runs=list(str(p.relative_to(R)) for p in R.rglob('*failure.json')))
    G.dump(R/'results/verification.json',out)
    primary=[]
    for (method,angle),rr in groups.items(): primary.append(dict(method=method,angle=angle,**G.summary(rr)))
    G.dump(R/'results/summary.json',primary)
    with (R/'figures/endpoints.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['method','angle','accuracy','accuracy_lo','accuracy_hi','ce','ce_lo','ce_hi','negative_fraction'])
        for z in primary: writer.writerow([z['method'],z['angle'],z['accuracy']['mean'],*z['accuracy']['ci95'],z['ce']['mean'],*z['ce']['ci95'],z['negative_fraction']['mean']])
    with (R/'figures/by_seed.csv').open('w') as f:
        writer=csv.writer(f); writer.writerow(['method','angle','seed','accuracy','ce','negative_fraction'])
        for z in rows: writer.writerow([z['method'],z['angle'],z['seed'],z['service']['accuracy'],z['service']['ce'],z['noiseless_negative_fraction']])
    print(json.dumps(out,indent=2))

if __name__=='__main__': check()
