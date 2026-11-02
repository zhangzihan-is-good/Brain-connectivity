"""Common launcher; each experiment retains its own arguments and runtime."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser(description=__doc__,epilog='Forward experiment arguments after --, for example: simulation -- --workers 4 --output NEW_OUTPUT')
    parser.add_argument('--python',dest='interpreter',default=sys.executable,help='Interpreter for the selected experiment')
    parser.add_argument('task',choices=('simulation','hcp','hnu','abide','test','info'))
    args,remaining=parser.parse_known_args()
    if remaining[:1]==['--']:remaining=remaining[1:]
    if args.task=='info':
        if remaining:parser.error('info does not accept additional arguments')
        import json
        metadata=json.loads((ROOT/'_vendor'/'IC_PROVENANCE.json').read_text(encoding='utf-8'))
        info={key:metadata[key] for key in ('repository','commit','version')}
        info.update(source=str(ROOT/'_vendor'/'IC'),upstream_files=len(metadata['files']),
                    interface=str(ROOT/'rgic_shared.py'))
        print(json.dumps(info,ensure_ascii=False,indent=2));return
    if args.task=='test':entry=ROOT/'run_tests.py'
    else:entry=ROOT/{'simulation':'Simulation','hcp':'HCP','hnu':'HNU','abide':'ABIDE'}[args.task]/'run.py'
    raise SystemExit(subprocess.call([args.interpreter,'-B',str(entry)]+remaining,cwd=str(entry.parent)))
if __name__=='__main__':main()
