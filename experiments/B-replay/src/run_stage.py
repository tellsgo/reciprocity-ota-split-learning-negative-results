import os,sys,subprocess,time,pathlib,json,datetime,platform
R=pathlib.Path(__file__).resolve().parents[1];stage=sys.argv[1]
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:env[k]='1'
start=time.time();cmd=[sys.executable,str(R/'src/experiment.py'),stage]
with (R/'logs'/f'{stage}.log').open('x') as log:
    p=subprocess.Popen(cmd,cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
    dump(R/'logs'/f'{stage}_job.json',dict(pid=p.pid,supervisor_pid=os.getpid(),command=cmd,started=datetime.datetime.now().astimezone().isoformat(),python=sys.version,platform=platform.platform(),threads=1))
    try:code=p.wait(timeout=1220)
    except subprocess.TimeoutExpired:p.kill();code=p.wait()
dump(R/'logs'/f'{stage}_exit.json',dict(pid=p.pid,returncode=code,wall_seconds=time.time()-start,reaped=True,finished=datetime.datetime.now().astimezone().isoformat()))
print(json.dumps(dict(stage=stage,returncode=code,wall_seconds=time.time()-start)),flush=True)
sys.exit(code)
