import os,sys,json,time,subprocess,signal
from pathlib import Path
R=Path(__file__).resolve().parents[1]
stage=sys.argv[1]
with (R/f'logs/{stage}.log').open('xb') as log:
 p=subprocess.Popen([sys.executable,'-B',str(R/'src/experiment.py'),stage],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'TMPDIR':str(R/'logs'),'PYTHONDONTWRITEBYTECODE':'1'})
 (R/f'logs/{stage}_job.json').write_text(json.dumps(dict(pid=p.pid,runner_pid=os.getpid(),started=time.time(),command=p.args)))
 try:code=p.wait(timeout=14430)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);code=p.wait()
 (R/f'logs/{stage}_exit.json').write_text(json.dumps(dict(exit_code=code,ended=time.time(),pid=p.pid)))
 sys.exit(code)
