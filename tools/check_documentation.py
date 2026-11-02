"""Check reader-facing Markdown paths/fences without evaluating shell text."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    docs=[ROOT/'README.md']
    readmes=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and p.name.lower().startswith('readme')]
    missing=[];fences=[];personal=[]
    links=re.compile(r'\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)')
    for path in docs:
        value=path.read_text(encoding='utf-8-sig')
        if sum(line.startswith('```') for line in value.splitlines())%2:fences.append(path.relative_to(ROOT).as_posix())
        if re.search(r'[A-Za-z]:[\\/](?:Users|大电脑|miniconda)',value):personal.append(path.relative_to(ROOT).as_posix())
        for match in links.finditer(value):
            target=(match.group(1) or match.group(2)).split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):continue
            if not (path.parent/target).exists():missing.append(dict(file=path.relative_to(ROOT).as_posix(),target=target))
    result=dict(status='PASS' if not missing and not fences and not personal and readmes==['README.md'] else 'FAIL',markdown_files=len(docs),
                readme_files=readmes,
                missing_links=missing,unbalanced_code_fences=fences,author_machine_paths_in_reader_docs=personal)
    print(json.dumps(result,ensure_ascii=False))
    if result['status']!='PASS':raise SystemExit(1)
