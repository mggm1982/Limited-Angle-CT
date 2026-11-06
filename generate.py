import json
from pathlib import Path
from collections import Counter
import numpy as np
from .config import read_config, geometry_metadata, indices_for_range
from .phantoms import phantom, split_plan
from .astra_ops import FanBeam, poisson_log, pad_sinogram


def save_array(path, x):
    np.save(path, np.asarray(x, dtype=np.float32), allow_pickle=False)


def generate(config_path, output, plan_only=False):
    c, root = read_config(config_path), Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory; existing dataset is never overwritten')
    plan = split_plan(c)
    metadata = geometry_metadata(c)
    metadata['split_samples_per_condition'] = dict(Counter(r['split'] for r in plan))
    metadata['split_families'] = {s: len({r['family'] for r in plan if r['split'] == s})
                                   for s in ('train', 'val', 'test')}
    metadata['poisson_zero_count_floor'] = .5
    (root/'geometry.json').write_text(json.dumps(metadata, indent=2))
    (root/'plan.json').write_text(json.dumps(plan, indent=2))
    if plan_only:
        print(json.dumps(metadata, indent=2))
        return
    op, records = FanBeam(c), []
    scale = c['attenuation_scale_per_mm']
    norm = lambda x: np.clip(x / scale, 0, 1).astype(np.float32)
    for i, item in enumerate(plan):
        common = root/'shared'/item['id']
        common.mkdir(parents=True, exist_ok=True)
        raster, shapes = phantom(c['seed'] + 10007 * item['family'], item['variant'], c['image_size'])
        clean = op.forward(raster * scale)
        rng = np.random.default_rng(np.random.SeedSequence([c['seed'], item['family'], item['variant']]))
        noisy = poisson_log(clean, c['photons'], rng)
        target_raw = op.reconstruct(clean)
        for name, array in [('phantom', raster), ('sinogram_clean', clean),
                            ('sinogram_noisy', noisy), ('target', norm(target_raw)),
                            ('target_raw', target_raw)]:
            save_array(common/f'{name}.npy', array)
        (common/'shapes.json').write_text(json.dumps(shapes, indent=2))
        for angle in c['angular_ranges']:
            indices = indices_for_range(c, angle)
            restricted = noisy[indices]
            padded, mask = pad_sinogram(restricted, indices, c['full_views'])
            folder = root/f'angle_{angle}'/item['id']
            folder.mkdir(parents=True, exist_ok=True)
            # Full-grid quadrature weights are preserved, avoiding range-dependent FBP rescaling.
            fbp_raw = op.reconstruct(padded)
            sart_raw = op.reconstruct(restricted, indices, 'sart')
            tv_raw = op.reconstruct(restricted, indices, 'tv_sart')
            for name, array in [('sinogram', restricted), ('sinogram_padded', padded),
                                ('measured_mask', mask), ('fbp_raw', fbp_raw),
                                ('sart_raw', sart_raw), ('tv_sart_raw', tv_raw),
                                ('fbp', norm(fbp_raw)), ('sart', norm(sart_raw)),
                                ('tv_sart', norm(tv_raw)),
                                ('artifact_mask', np.abs(norm(fbp_raw)-norm(target_raw)))]:
                save_array(folder/f'{name}.npy', array)
            entry = {**item, 'angle': angle, 'path': folder.relative_to(root).as_posix(),
                     'shared': common.relative_to(root).as_posix()}
            records.append(entry)
            (folder/'metadata.json').write_text(json.dumps({**entry, 'view_indices': indices.tolist(),
                    'range_deg': [0, angle], 'n_views': len(indices), 'angular_density_views_per_deg':
                    c['full_views']/360, 'geometry_file': '../../geometry.json'}, indent=2))
        print(f'[{i+1}/{len(plan)}] {item["id"]}', flush=True)
    # Publish manifest only when dataset generation finishes; partial data cannot train accidentally.
    manifest = {'config': c, 'records': records, 'complete': True}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
