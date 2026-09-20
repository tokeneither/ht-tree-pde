#!/usr/bin/env python3
"""Export non-ignored project files, without Git history or local build caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import subprocess
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/ht-tree-pde-github.zip')
    args = parser.parse_args()
    output = args.output.resolve()
    # Git handles its metadata directory and ignore rules; include new source
    # files as well as already tracked project files.
    result = subprocess.run(
        ['git','ls-files','--cached','--others','--exclude-standard','-z'],
        cwd=ROOT,check=True,capture_output=True)
    names = sorted(set(result.stdout.decode().split('\0'))-{''})
    files = []
    for name in names:
        if name == 'REPOSITORY_CONTENTS.json':
            continue
        path = ROOT/name
        if not path.is_file() or path.is_symlink() or path.resolve()==output:
            continue
        if any(p in {'.git','.cache','.venv','__pycache__','build','dist'}
               or p.endswith('.egg-info') for p in Path(name).parts):
            continue
        files.append((name,path.read_bytes(),path.stat().st_mode))
    if not files:
        raise RuntimeError('No project files found; run git init -b main first.')
    manifest = dict(format_version=1,root='ht-tree-pde',files=len(files),
        file_sha256={name:hashlib.sha256(data).hexdigest() for name,data,_ in files},
        note='Hashes describe this export, excluding the manifest itself. No Git history included.')
    files.append(('REPOSITORY_CONTENTS.json',(json.dumps(manifest,indent=2)+'\n').encode(),0o644))
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        for name,data,mode in files:
            info = ZipInfo('ht-tree-pde/'+name,date_time=(1980,1,1,0,0,0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | (mode & 0o777)) << 16
            archive.writestr(info,data)
    print(f'Exported {len(files)-1} project files plus a contents manifest')
    print(f'{output} ({output.stat().st_size/1_000_000:.2f} MB)')


if __name__=='__main__':
    main()
