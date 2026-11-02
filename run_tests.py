"""Run dataset-free verification of the code-only repository."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',default=sys.executable,help='Python 3.12 / NumPy 2.3.5 interpreter')
    args=parser.parse_args()
    commands=[('pinned_source_hashes',ROOT/'tests'/'verify_vendor.py'),
              ('public_core',ROOT/'run_public_tests.py'),
              ('upstream_scientific_and_io_tests',ROOT/'tests'/'run_vendor_tests.py')]
    failed=[]
    for name,entry in commands:
        print('Running '+name,flush=True)
        code=subprocess.call([args.python,'-B',str(entry)],cwd=str(ROOT))
        if code:failed.append(name)
    print('PASS: all code-only checks' if not failed else 'FAIL: '+', '.join(failed),flush=True)
    if failed:raise SystemExit(1)
if __name__=='__main__':main()
