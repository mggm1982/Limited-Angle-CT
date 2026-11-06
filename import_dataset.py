"""Safely merge the two supplied dataset ZIPs without executing bundled scripts."""
import json
import shutil
import stat
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
import numpy as np

DATASET_ROOT = 'limited_angle_ct_1000'


def _inside(root, relative):
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or '\\' in relative or ':' in relative:
        raise ValueError(f'Unsafe dataset path: {relative}')
    dest = root.joinpath(*p.parts)
    if not dest.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Dataset path escapes root: {relative}')
    return dest


def validate_dataset(root, expected_samples=1000, check_arrays=True):
    root = Path(root)
    m = json.loads((root/'manifest.json').read_text())
    if not m.get('complete'):
        raise ValueError('Incomplete dataset manifest')
    records = m['records']
    if len(records) != expected_samples*2:
        raise ValueError(f'Expected {expected_samples*2} paired cases, found {len(records)}')
    families, seen, samples = {}, set(), set()
    for r in records:
        key = (r['id'], r['angle'])
        if key in seen or r['angle'] not in (90,120) or r['split'] not in ('train','val','test'):
            raise ValueError(f'Duplicate or invalid record: {key}')
        seen.add(key); samples.add(r['id'])
        if r['family'] in families and families[r['family']] != r['split']:
            raise ValueError('A phantom family crosses dataset splits')
        families[r['family']] = r['split']
        for rel in [r['path']+'/fbp.npy', r['shared']+'/target.npy', r['shared']+'/phantom.npy']:
            file = _inside(root, rel)
            if not file.is_file():
                raise FileNotFoundError(f'Missing {file}. Extract BOTH training ZIP parts.')
            if check_arrays:
                a = np.load(file, allow_pickle=False, mmap_mode='r')
                n = m['config']['image_size']
                if a.shape != (n,n) or a.dtype != np.float32 or not np.isfinite(a).all():
                    raise ValueError(f'Invalid image array: {file}')
                if a.min()<0 or a.max()>1:
                    raise ValueError(f'Array outside fixed normalized range: {file}')
    if len(samples)!=expected_samples or any((i,a) not in seen for i in samples for a in (90,120)):
        raise ValueError('Each phantom must have both angle conditions')
    counts = {str(a):dict(Counter(r['split'] for r in records if r['angle']==a)) for a in (90,120)}
    if expected_samples==1000 and any(c!={'train':700,'val':150,'test':150} for c in counts.values()):
        raise ValueError(f'Unexpected 1,000-image split: {counts}')
    return {'unique_images':len(samples), 'paired_cases':len(records), 'splits':counts,
            'backend':m.get('reconstruction_backend', 'see geometry.json'),
            'available_reconstructions':m.get('available_reconstructions',['fbp'])}


def _crc(file):
    value=0
    with file.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            value=zlib.crc32(block,value)
    return value & 0xffffffff


def import_dataset(part1, part2, output, expected_samples=1000):
    paths=[Path(part1).resolve(),Path(part2).resolve()]
    output=Path(output).resolve()
    if paths[0]==paths[1]:
        raise ValueError('Part 1 and Part 2 must be different ZIP files')
    for file in paths:
        if not file.is_file():raise FileNotFoundError(f'Dataset ZIP not found: {file}')
    manifests=[]; member_names=set()
    # Reject traversal/symlink entries before making any dataset writes.
    for file in paths:
        with zipfile.ZipFile(file) as z:
            manifests.append(z.read(DATASET_ROOT+'/manifest.json'))
            for info in z.infolist():
                name=PurePosixPath(info.filename)
                _inside(output.parent,info.filename)
                if not name.parts or name.parts[0]!=DATASET_ROOT:
                    raise ValueError(f'Unexpected archive root: {info.filename}')
                if stat.S_ISLNK(info.external_attr>>16):
                    raise ValueError('Dataset ZIP cannot contain symlinks')
                member_names.add(info.filename)
    if manifests[0]!=manifests[1]:
        raise ValueError('The ZIP parts contain different dataset manifests')
    manifest=json.loads(manifests[0])
    for r in manifest['records']:
        for rel in [r['path']+'/fbp.npy',r['shared']+'/target.npy',r['shared']+'/phantom.npy']:
            if DATASET_ROOT+'/'+rel not in member_names:
                raise ValueError('ZIP parts do not contain all declared pairs; provide both matching parts')
    if output.exists():
        if (output/'manifest.json').read_bytes()!=manifests[0]:
            raise FileExistsError('Existing dataset is different. Choose a new output folder.')
        report=validate_dataset(output,expected_samples)
        print('Existing extracted dataset verified:',json.dumps(report),flush=True)
        return report
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.import_ct_',dir=output.parent))
    try:
        for file in paths:
            print(f'Extracting {file.name}',flush=True)
            with zipfile.ZipFile(file) as z:
                for info in z.infolist():
                    dest=_inside(stage,info.filename)
                    if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():
                        if dest.stat().st_size!=info.file_size or _crc(dest)!=info.CRC:
                            raise ValueError(f'ZIP parts disagree about {info.filename}')
                        continue
                    with z.open(info) as src,dest.open('wb') as target:
                        shutil.copyfileobj(src,target)
        report=validate_dataset(stage/DATASET_ROOT,expected_samples)
        (stage/DATASET_ROOT/'local_import.json').write_text(json.dumps({
            'archives':[str(p) for p in paths], 'validation':report},indent=2))
        (stage/DATASET_ROOT).rename(output)
        print('Dataset ready:',json.dumps(report),flush=True)
        return report
    finally:
        shutil.rmtree(stage,ignore_errors=True)
