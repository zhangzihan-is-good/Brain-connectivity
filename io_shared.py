"""Shared file utilities with explicit serialization/formatting choices."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def read_json(path,encoding='utf-8'):
    return json.loads(Path(path).read_text(encoding=encoding))

def sha256_file(path, block_size=1024*1024):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(block_size),b''):
            h.update(block)
    return h.hexdigest()

def clean_json(value):
    if isinstance(value,dict): return {str(k):clean_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [clean_json(v) for v in value]
    if isinstance(value,np.ndarray): return clean_json(value.tolist())
    if isinstance(value,np.generic): return clean_json(value.item())
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def write_json(path,value,*,indent=2,ensure_ascii=False,allow_nan=False,clean=True,
               atomic=True,sort_keys=False,newline=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=clean_json(value) if clean else value
    serialized=json.dumps(payload,ensure_ascii=ensure_ascii,indent=indent,allow_nan=allow_nan,sort_keys=sort_keys)
    if newline: serialized+='\n'
    target=path.with_suffix('.tmp.json') if atomic else path
    target.write_text(serialized,encoding='utf8')
    if atomic: os.replace(target,path)

def write_csv(path,rows,fields=None,*,clean=False):
    if fields is None: fields=list(rows[0])
    with Path(path).open('w',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(clean_json(rows) if clean else rows)
