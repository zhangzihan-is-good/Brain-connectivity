"""Local release guardrails; no upload and no automatic licensing decision.

This checks the current directory, not a hypothetical filtered release. A PASS
on mechanical rules would still require substantive permissions/data review.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def check(root):
    checks=[]
    def add(name,status,severity,detail,paths=()):
        checks.append(dict(check=name,status=status,severity=severity,detail=detail,paths=list(paths)))
    licenses=[p.name for p in root.iterdir() if p.is_file() and p.name.lower() in ('license','license.txt','license.md','copying')]
    add('project_license','PRESENT_REQUIRES_REVIEW' if licenses else 'BLOCKED','blocking',
        'An authorized project-wide license and exception scope must be established.',licenses)
    vendor=root/'_vendor'/'IC'
    vendor_license=[p.relative_to(root).as_posix() for p in vendor.glob('*') if p.is_file() and p.name.lower() in ('license','license.txt','license.md','copying')]
    add('pinned_ic_license','PRESENT_REQUIRES_REVIEW' if vendor_license else 'BLOCKED','blocking',
        'The fixed source snapshot needs an explicit license or documented applicable permission.',vendor_license)
    real=[p for p in ('tests/baselines/hnu_original.npz','tests/baselines/hcp_original_arrays.npz') if (root/p).is_file()]
    add('real_data_regression_payloads','BLOCKED' if real else 'PASS','blocking',
        'Real signals/derived connectome fixtures need separate distribution clearance or exclusion.',real)
    linked=[]
    tree=root/'ABIDE'/'code'/'main'/'source_tree'
    if tree.is_dir():
        for path in tree.rglob('selection.json'):
            try:value=json.loads(path.read_text(encoding='utf-8-sig'))
            except (OSError,ValueError):continue
            if isinstance(value,dict) and 'valid_ids' in value and 'valid_y_ASD' in value:
                linked.append(path.relative_to(root).as_posix())
    add('participant_linked_diagnoses','BLOCKED' if linked else 'PASS','blocking',
        'Participant-linked selection records are distinct from aggregate performance tables.',linked)
    internal=[p for p in ('ABIDE/_packaging_work','ABIDE/code/main/execution_provenance','verification','docs/internal') if (root/p).exists()]
    add('internal_release_filter','REVIEW_REQUIRED' if internal else 'PASS','review',
        'Use an explicit reviewed release allowlist; gitignore does not sanitize an archive.',internal)
    docs=['README.md']
    missing=[p for p in docs if not (root/p).is_file()]
    add('reader_documentation','PASS' if not missing else 'BLOCKED','blocking',
        'One reader-oriented README consolidating installation, reproduction, data and source notices.',missing)
    public=['run_public_tests.py','examples/synthetic_demo.py','tests/test_public_synthetic.py','requirements-demo.txt']
    missing_public=[p for p in public if not (root/p).is_file()]
    add('dataset_free_test_entry','PASS' if not missing_public else 'BLOCKED','blocking',
        'This checks entry existence; execution is recorded separately.',missing_public)
    readmes=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name.lower().startswith('readme')]
    add('single_readme','PASS' if readmes==['README.md'] else 'BLOCKED','blocking',
        'The repository must have exactly its consolidated root README.',readmes)
    add('third_party_permission_chain','AUTHOR_REVIEW_REQUIRED','review',
        'Resolve MADE source/license preservation, GCMI implementation provenance and model ownership; filename checks cannot establish these permissions.')
    add('release_identifier_and_citation','AUTHOR_REVIEW_REQUIRED','review',
        'Populate and version the experimental-code release and final approved manuscript/software citation.')
    blocked=any(c['status']=='BLOCKED' for c in checks)
    return dict(status='BLOCKED' if blocked else 'MANUAL_REVIEW_REQUIRED',
                audited_utc=datetime.now(timezone.utc).isoformat(),
                scope='local current tree; no files published or deleted',
                legal_clearance_automated=False,checks=checks)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--strict',action='store_true',help='Return nonzero until blocking/manual review items are cleared')
    args=parser.parse_args();result=check(args.root.resolve())
    serialized=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(serialized,encoding='utf-8')
    print(json.dumps(dict(status=result['status'],checks=len(result['checks']),
                          blockers=[c['check'] for c in result['checks'] if c['status']=='BLOCKED'],
                          participant_record_files=sum(len(c['paths']) for c in result['checks'] if c['check']=='participant_linked_diagnoses')),indent=2))
    if args.strict and result['status']!='CLEARED':raise SystemExit(2)
if __name__=='__main__':main()
