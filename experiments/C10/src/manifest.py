import pathlib,hashlib,json,datetime
r=pathlib.Path(__file__).resolve().parents[1]
entries=[]
for p in sorted(r.rglob('*')):
 if p.is_file() and p.name!='ARTIFACT_MANIFEST.json':
  entries.append(dict(path=str(p.relative_to(r)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
v=dict(time=datetime.datetime.now().astimezone().isoformat(),entries=entries,file_count=len(entries),total_bytes=sum(x['bytes'] for x in entries),self_excluded=True)
(r/'ARTIFACT_MANIFEST.json').write_text(json.dumps(v,indent=2))
print('MANIFEST',len(entries),'files',v['total_bytes'],'bytes')
