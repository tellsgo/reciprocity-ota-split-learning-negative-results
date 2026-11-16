"""Launch and reap one local experiment process; keep exit evidence even on failure."""
import pathlib,sys,os,json,subprocess,time
R=pathlib.Path(__file__).resolve().parents[1];phase=sys.argv[1]
cmd=[sys.executable,'-B',str(R/'src/g50.py'),phase];start=time.time()
env=os.environ.copy();env.update(TMPDIR=str(R/'tmp'),XDG_CACHE_HOME=str(R/'cache'),MPLCONFIGDIR=str(R/'cache/matplotlib'),PYTHONDONTWRITEBYTECODE='1')
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
for p in ['tmp','cache/matplotlib']:(R/p).mkdir(parents=True,exist_ok=True)
with (R/f'logs/{phase}.log').open('x') as log:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    job=dict(command=cmd,pid=p.pid,supervisor_pid=os.getpid(),started=start)
    (R/f'logs/{phase}_job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job),flush=True)
    try:code=p.wait(timeout=1810)
    except subprocess.TimeoutExpired:p.kill();code=p.wait()
out=dict(**job,finished=time.time(),elapsed=time.time()-start,exit_code=code,reaped=True)
(R/f'logs/{phase}_exit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
sys.exit(0 if code==0 else 1)
