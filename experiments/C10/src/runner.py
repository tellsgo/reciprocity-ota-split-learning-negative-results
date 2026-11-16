import pathlib,subprocess,sys,json,time,os
R=pathlib.Path(__file__).resolve().parents[1];stage=sys.argv[1]
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
start=time.time()
with (R/f'logs/{stage}.log').open('x') as f:
 p=subprocess.Popen([sys.executable,'-B',str(R/'src/g51.py'),stage],cwd=str(R),stdout=f,stderr=subprocess.STDOUT,env=env)
 (R/f'logs/{stage}_job.json').write_text(json.dumps(dict(pid=p.pid,runner_pid=os.getpid(),start=start,command=[sys.executable,'-B','src/g51.py',stage]),indent=2))
 rc=p.wait()
(R/f'logs/{stage}_exit.json').write_text(json.dumps(dict(pid=p.pid,exit_code=rc,elapsed=time.time()-start,finished=time.time()),indent=2))
print(stage,rc,'elapsed',time.time()-start,flush=True)
sys.exit(rc)
