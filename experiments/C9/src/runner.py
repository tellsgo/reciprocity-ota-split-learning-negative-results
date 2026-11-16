"""Local supervised runner; all writes stay in this run, child is reaped."""
import os,sys,pathlib,json,time,subprocess,signal
R=pathlib.Path(__file__).resolve().parents[1];stage=sys.argv[1]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(R/'tmp'),MPLCONFIGDIR=str(R/'tmp'))
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
start=time.time();cmd=[sys.executable,'-B',str(R/'src/g43.py'),stage]
with (R/f'logs/{stage}.log').open('x') as log:
    proc=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (R/f'logs/{stage}_job.json').write_text(json.dumps(dict(pid=proc.pid,runner_pid=os.getpid(),start=start,command=cmd,cwd=str(R)),indent=2))
    timeout=False
    try:rc=proc.wait(timeout=1800)
    except subprocess.TimeoutExpired:
        timeout=True;os.killpg(proc.pid,signal.SIGTERM)
        try:rc=proc.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);rc=proc.wait()
(R/f'logs/{stage}_exit.json').write_text(json.dumps(dict(pid=proc.pid,exit_code=rc,timeout=timeout,child_reaped=True,seconds=time.time()-start,finished=time.time()),indent=2))
print(stage,'exit',rc,'seconds',time.time()-start,flush=True);sys.exit(rc)
