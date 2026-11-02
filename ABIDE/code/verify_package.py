"""Compatibility entry for the active code/configuration integrity check."""
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parents[1]
def main():
    spec=importlib.util.spec_from_file_location('_abide_integrity',BASE/'tools/integrity.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    print(json.dumps(module.check(),ensure_ascii=False,indent=2))
    return 0
if __name__=='__main__':sys.exit(main())
