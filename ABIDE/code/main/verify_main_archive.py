"""Compatibility integrity entry; optionally inspect explicit external inputs."""
import sys
sys.dont_write_bytecode = True
import argparse
import importlib.util
import json
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--data-workspace",type=Path)
    p.add_argument("--output-root",type=Path)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('_abide_integrity',BASE/'tools/integrity.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    out=checker.check()
    if a.dry_run:
        if a.data_workspace is None:p.error('--dry-run requires explicit --data-workspace')
        c=checker.read(BASE/"configs/main/input_contract.json");w=a.data_workspace.resolve()
        external=[]
        for item in c["requirements"]:
            f=(w/item["relative_to_workspace"]).resolve();f.relative_to(w)
            external.append(dict(relative_to_workspace=item['relative_to_workspace'],exists=f.is_file(),role=item["role"],payload_SHA_checked=False))
        out.update(external_files=external,external_file_count=len(external),output_created=False,scientific_computation_performed=False)
        if a.output_root:
            target=a.output_root.resolve();protected=[BASE.resolve(),w/'dataset',w/'第四部分',w/'第三部分最终版']
            out['output_outside_protected_roots']=not any(target==x or x in target.parents or target in x.parents for x in protected)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    return 0
if __name__=="__main__":raise SystemExit(main())
