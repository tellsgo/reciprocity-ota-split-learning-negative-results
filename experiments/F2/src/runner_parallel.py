import os,sys,json,time,subprocess,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1]
with (R/'logs/parallel.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B',str(R/'src/parallel_resume.py')],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'TMPDIR':str(R/'logs'),'PYTHONDONTWRITEBYTECODE':'1'})
 (R/'logs/parallel_job.json').write_text(json.dumps(dict(pid=p.pid,runner_pid=os.getpid(),started=time.time(),command=p.args)))
 remaining=max(1,14400-(time.time()-json.loads((R/'logs/formal_job.json').read_text())['started']))
 try:code=p.wait(timeout=remaining)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);code=p.wait()
 (R/'logs/parallel_exit.json').write_text(json.dumps(dict(exit_code=code,ended=time.time(),pid=p.pid)))
 sys.exit(code)
