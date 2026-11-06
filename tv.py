"""TV hyperparameters are selected without consulting test labels."""
import json
from pathlib import Path
import hashlib
import numpy as np
from .data import manifest
from .astra_ops import FanBeam
from .generate import save_array


def tune(args):
    root, out = Path(args.data), Path(args.output)
    if (out/'selection.json').exists():
        raise FileExistsError('Existing TV selection; use another output folder')
    out.mkdir(parents=True, exist_ok=True)
    m = manifest(root)
    op, scores = FanBeam(m['config']), []
    scale = m['config']['attenuation_scale_per_mm']
    rows = [r for r in m['records'] if r['split']=='val' and r['angle']==args.angle]
    if not rows or any(w<0 for w in args.weights):
        raise ValueError('Need validation records and nonnegative TV weights')
    for w in args.weights:
        errors = []
        for r in rows:
            sino = np.load(root/r['path']/'sinogram.npy', allow_pickle=False)
            raw = op.reconstruct(sino, np.arange(len(sino)), 'tv_sart', tv_weight=w)
            target = np.load(root/r['shared']/'target.npy', allow_pickle=False)
            errors.append(float(np.mean(np.abs(np.clip(raw/scale,0,1)-target))))
        scores.append({'weight':w,'val_l1':float(np.mean(errors))})
        print(scores[-1], flush=True)
    selection = {'angle':args.angle, 'weight':min(scores,key=lambda r:r['val_l1'])['weight'],
                 'scores':scores, 'selected_on':'val', 'validation_samples':len(rows),
                 'geometry_sha256':hashlib.sha256((root/'geometry.json').read_bytes()).hexdigest()}
    (out/'selection.json').write_text(json.dumps(selection, indent=2))


def apply(args):
    root = Path(args.data)
    selection = json.loads(Path(args.selection).read_text())
    digest = hashlib.sha256((root/'geometry.json').read_bytes()).hexdigest()
    if selection['selected_on']!='val' or selection['geometry_sha256'] != digest:
        raise ValueError('Selection was not made on matching validation geometry')
    if list(root.glob('evaluation_lock*')):
        raise ValueError('Dataset marked evaluated; do not change a baseline after final evaluation')
    m = manifest(root)
    op = FanBeam(m['config'])
    scale = m['config']['attenuation_scale_per_mm']
    # Only TV arrays change. Image training inputs and manifest identity remain stable.
    for r in m['records']:
        if r['angle'] != selection['angle']:
            continue
        folder = root/r['path']
        sino = np.load(folder/'sinogram.npy', allow_pickle=False)
        raw = op.reconstruct(sino, np.arange(len(sino)), 'tv_sart', selection['weight'])
        save_array(folder/'tv_sart_raw.npy', raw)
        save_array(folder/'tv_sart.npy', np.clip(raw/scale,0,1))
    (root/f'tv_selection_{selection["angle"]}.json').write_text(json.dumps(selection, indent=2))
