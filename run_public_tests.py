"""Run dataset-free tests; no author workspace, GPU, or original tree needed."""
import json
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def main():
    suite=unittest.TestSuite()
    loader=unittest.TestLoader()
    suite.addTests(loader.discover(str(ROOT/'tests'),pattern='test_public_*.py'))
    suite.addTests(loader.discover(str(ROOT/'tests'),pattern='test_backend_profiles.py'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
                          skipped=len(result.skipped),participant_data_used=False,
                          original_tree_required=False),indent=2))
    if not result.wasSuccessful():raise SystemExit(1)
if __name__=='__main__':main()
