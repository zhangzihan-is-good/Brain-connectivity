"""Verify a minimal code-only standalone copy; this is not a release exporter."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default=sys.executable)
    parser.add_argument('--output',type=Path,required=True,help='New/empty local verification directory')
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new/empty verification output')
    stage=out/'standalone_core';stage.mkdir(parents=True,exist_ok=True)
    files=['run.py','run_public_tests.py','rgic_backend.py','rgic_shared.py','rgic_profiles_abide.py',
           'io_shared.py','shared_paths.py','requirements-demo.txt','examples/synthetic_demo.py',
           'tests/test_public_synthetic.py','tests/test_backend_profiles.py','_vendor/IC_PROVENANCE.json']
    files += [p.relative_to(ROOT).as_posix() for p in (ROOT/'_vendor'/'IC').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for name in files:
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env['FREEENERGY_WORKSPACE']=str(out/'absent_external_workspace')
    commands=[('info',['run.py','info']),('public_tests',['run_public_tests.py']),
              ('demo',['examples/synthetic_demo.py','--output',str(out/'demo')])]
    results=[]
    for name,parts in commands:
        log=out/(name+'.log')
        with log.open('w',encoding='utf-8') as stream:
            result=subprocess.run([args.python,'-B',str(stage/parts[0])]+parts[1:],cwd=str(stage),env=env,stdout=stream,stderr=subprocess.STDOUT)
        results.append(dict(name=name,exit_code=result.returncode,log=log.name))
    payloads=[p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file() and p.suffix.lower() in ('.npz','.npy','.hdf5','.h5','.nii','.pkl','.pt')]
    receipt=dict(status='PASS' if all(r['exit_code']==0 for r in results) and not payloads else 'FAIL',
                 checked_utc=datetime.now(timezone.utc).isoformat(),copied_files=len(files),copied_dataset_payloads=payloads,
                 participant_data_used=False,original_source_siblings_in_copy=False,
                 fresh_dependency_install_performed=False,scope='minimal standalone core, not four full experiment workflows',
                 commands=results,source_hashes={name:hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in files})
    (out/'result.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='source_hashes'},indent=2))
    if receipt['status']!='PASS':raise SystemExit(1)
if __name__=='__main__':main()
