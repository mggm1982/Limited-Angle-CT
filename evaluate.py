import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import manifest
from .models import ResidualUNet
from .metrics import image_metrics, directional_error, directional_rmse
from .train import manifest_hash
from .device import resolve_device


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_generator(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    g = ResidualUNet(ckpt['settings']['base']).to(device)
    g.load_state_dict(ckpt['generator'])
    g.eval()
    return g, ckpt


def evaluate(args):
    args.device = str(resolve_device(args.device))
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'summary.csv').exists():
        raise FileExistsError('Evaluation output exists; choose a new directory')
    m = manifest(root)
    cfg = m['config']
    records = [r for r in m['records'] if r['angle'] == args.angle and r['split'] == args.split]
    if not records:
        raise ValueError('No matching evaluation records')
    baseline_names = getattr(args, 'baselines', None) or m.get('available_reconstructions', ['fbp','sart','tv_sart'])
    if not baseline_names or any(n not in ('fbp','sart','tv_sart') for n in baseline_names):
        raise ValueError('Invalid baseline selection')
    for r in records:
        for name in baseline_names:
            file = root/r['path']/f'{name}.npy'
            if not file.is_file():
                raise FileNotFoundError(f'Missing selected baseline: {file}. For FBP-only data use --baselines fbp.')
    raw_files = [(root/r['path']/'fbp_raw.npy').is_file() for r in records]
    if any(raw_files) and not all(raw_files):
        raise ValueError('Raw FBP is present for only some evaluation records')
    include_raw = 'fbp' in baseline_names and all(raw_files)
    print('Evaluating baselines:', ', '.join(baseline_names),
          '(raw FBP also available)' if include_raw else '(raw FBP unavailable)', flush=True)
    models = {}
    for label_path in args.checkpoint or []:
        label, path = label_path.split('=', 1)
        if label in models or label in ('fbp','fbp_unclipped','sart','tv_sart'):
            raise ValueError('Duplicate/reserved method name')
        g, ckpt = load_generator(path, args.device)
        if ckpt['settings']['angle'] != args.angle or ckpt['manifest_sha256'] != manifest_hash(root):
            raise ValueError('Checkpoint angle/dataset does not match evaluation')
        models[label] = (g, ckpt)
    op = None
    if args.projection_residual:
        from .astra_ops import FanBeam
        op = FanBeam(cfg)
    lock = None
    if args.split == 'test':
        lock = root/f'evaluation_lock_{args.angle}.json'
        with lock.open('x') as f:
            json.dump({'status':'running', 'output':str(output), 'arguments':vars(args)}, f, indent=2)
    bins = cfg['orientation_bins']
    rows, sums = [], defaultdict(lambda: [np.zeros(bins), np.zeros(bins, dtype=np.int64)])
    scale = cfg['attenuation_scale_per_mm']
    for index, record in enumerate(records):
        folder, shared = root/record['path'], root/record['shared']
        target = np.load(shared/'target.npy', allow_pickle=False)
        ideal = np.load(shared/'phantom.npy', allow_pickle=False)
        predictions = {name: np.load(folder/f'{name}.npy', allow_pickle=False)
                       for name in baseline_names}
        if include_raw:
            predictions['fbp_unclipped'] = np.load(folder/'fbp_raw.npy', allow_pickle=False)/scale
        with torch.no_grad():
            for label, (g, ckpt) in models.items():
                x = np.load(folder/f'{ckpt["settings"]["input"]}.npy', allow_pickle=False)
                raw = g(torch.from_numpy(x)[None,None].to(args.device)).cpu().numpy()[0,0]
                predictions[label] = np.clip(raw, 0, 1)
                dest = output/'predictions'/label
                dest.mkdir(parents=True, exist_ok=True)
                np.save(dest/f'{record["id"]}.npy', predictions[label].astype(np.float32))
                np.save(dest/f'{record["id"]}_unclipped.npy', raw.astype(np.float32))
        for method, p in predictions.items():
            row = {'id':record['id'], 'family':record['family'], 'angle':args.angle,
                   'method':method, 'split':args.split}
            for name, ref in [('target',target), ('ideal',ideal)]:
                row.update({f'{name}_{k}':v for k,v in image_metrics(p,ref).items()})
                # Bin normals relative to central-ray measured-normal center.
                sse, count = directional_error(p, ref, bins, cfg['edge_threshold_fraction'], args.angle/2)
                sums[(method,name)][0] += sse
                sums[(method,name)][1] += count
            if op:
                measured = np.load(folder/'sinogram.npy', allow_pickle=False)
                reprojected = op.forward(p*scale, np.arange(len(measured)))
                row['measured_projection_relative_l2'] = float(np.linalg.norm(reprojected-measured) /
                                                              max(np.linalg.norm(measured),1e-12))
            rows.append(row)
        if index < args.panels:
            images = {'Ideal':ideal, 'Full FBP target':target, **predictions}
            fig, axes = plt.subplots(2, len(images), figsize=(3*len(images), 6), squeeze=False)
            for ax, (name, p) in zip(axes[0], images.items()):
                ax.imshow(p, cmap='gray', vmin=0, vmax=1)
                ax.set_title(name)
                ax.axis('off')
            for ax, (name, p) in zip(axes[1], images.items()):
                ax.imshow(np.abs(p-ideal), cmap='magma', vmin=0, vmax=.3)
                ax.set_title('|error| vs ideal [0,.3]')
                ax.axis('off')
            fig.suptitle(f'{args.angle} degrees; {record["id"]}; shared display ranges')
            fig.tight_layout()
            fig.savefig(output/f'comparison_{record["id"]}.png', dpi=130)
            plt.close(fig)
    write_csv(output/'per_sample.csv', rows)
    metric_keys = [k for k in rows[0] if k.startswith(('target_','ideal_','measured_'))]
    summary = []
    for method in sorted({r['method'] for r in rows}):
        selected = [r for r in rows if r['method'] == method]
        item = {'angle':args.angle, 'split':args.split, 'method':method,
                'samples':len(selected), 'families':len({r['family'] for r in selected})}
        for key in metric_keys:
            values = np.array([r[key] for r in selected])
            item[key+'_mean'] = float(values.mean())
            item[key+'_std'] = float(values.std())
        summary.append(item)
    write_csv(output/'summary.csv', summary)
    # Family means reduce misleading precision from correlated material variants.
    family_rows = []
    for method in sorted({r['method'] for r in rows}):
        for family in sorted({r['family'] for r in rows}):
            selected = [r for r in rows if r['method']==method and r['family']==family]
            if selected:
                family_rows.append({'angle':args.angle, 'method':method, 'family':family,
                                    **{k:float(np.mean([r[k] for r in selected])) for k in metric_keys}})
    write_csv(output/'per_family.csv', family_rows)
    direction_rows = []
    centers = (np.arange(bins)+.5)*180/bins
    for ref in ('target','ideal'):
        fig, ax = plt.subplots(subplot_kw={'projection':'polar'}, figsize=(7,6))
        for (method, reference), (sse, count) in sums.items():
            if reference != ref:
                continue
            rmse = directional_rmse(sse, count)
            theta = np.deg2rad(centers)
            ax.plot(theta, rmse, label=method)
            for i in range(bins):
                direction_rows.append({'angle':args.angle, 'reference':ref, 'method':method,
                    'relative_edge_normal_deg':float(centers[i]), 'sse':float(sse[i]),
                    'edge_pixels':int(count[i]), 'rmse':float(rmse[i]) if count[i] else ''})
        ax.set_thetamin(0)
        ax.set_thetamax(180)
        ax.set_title(f'{args.angle} degrees: pixel RMSE at reference edges\nnormal relative to scan center; fan wedge approximate')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02,1))
        fig.savefig(output/f'directional_{ref}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
    write_csv(output/'directional.csv', direction_rows)
    (output/'evaluation.json').write_text(json.dumps({'arguments':vars(args),
           'manifest_sha256':manifest_hash(root), 'checkpoint_paths':args.checkpoint,
           'checkpoint_sha256': {spec.split('=',1)[0]: hashlib.sha256(Path(spec.split('=',1)[1]).read_bytes()).hexdigest()
                                 for spec in (args.checkpoint or [])},
           'tv_selection': json.loads((root/f'tv_selection_{args.angle}.json').read_text())
                           if (root/f'tv_selection_{args.angle}.json').exists() else {'weight':cfg.get('tv_weight'), 'selected_on':'fixed_config'},
           'baselines_evaluated': baseline_names,
           'raw_fbp_evaluated': include_raw,
           'dataset_backend': m.get('reconstruction_backend', 'see geometry.json'),
           'dataset_available_reconstructions': m.get('available_reconstructions'),
           'metric_range':1.0, 'prediction_policy':'network outputs clipped; unclipped arrays saved',
           'directional_definition':'reference Sobel gradient normal in physical x/y; modulo 180; '
           'subtract scan center angle; no exact fan-beam missing-wedge classification'}, indent=2))
    if lock:
        lock.write_text(json.dumps({'status':'completed', 'output':str(output),
                                   'arguments':vars(args)}, indent=2))
    print(f'Evaluated {len(records)} samples. Results: {output}')


def infer(args):
    args.device = str(resolve_device(args.device))
    g, ckpt = load_generator(args.checkpoint, args.device)
    x = np.load(args.input, allow_pickle=False).astype(np.float32)
    n = ckpt['config']['image_size']
    if x.shape != (n,n) or not np.isfinite(x).all() or x.min()<0 or x.max()>1:
        raise ValueError(f'Expected a finite normalized [{n},{n}] image in [0,1]')
    with torch.no_grad():
        p = g(torch.from_numpy(x)[None,None].to(args.device)).cpu().numpy()[0,0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.clip(p,0,1).astype(np.float32))
    out.with_suffix('.json').write_text(json.dumps({'angle':ckpt['settings']['angle'],
        'input_kind':ckpt['settings']['input'], 'attenuation_scale_per_mm':
        ckpt['config']['attenuation_scale_per_mm'], 'units':'normalized attenuation',
        'caution':'Learned missing structure may be factually incorrect.'}, indent=2))
