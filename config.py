import json
from pathlib import Path
import numpy as np


def read_config(path):
    c = json.loads(Path(path).read_text())
    if c['image_size'] < 32 or c['image_size'] % 32:
        raise ValueError('image_size must be a multiple of 32 and >=32')
    for key in ('full_views', 'detectors', 'phantom_instances', 'samples_per_condition',
                'sart_sweeps', 'tv_inner_iterations', 'orientation_bins'):
        if int(c[key]) != c[key] or c[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('pixel_size_mm', 'source_origin_mm', 'origin_detector_mm',
                'detector_spacing_mm', 'photons', 'attenuation_scale_per_mm'):
        if c[key] <= 0:
            raise ValueError(f'{key} must be positive')
    if c['phantom_instances'] < 3 or c['samples_per_condition'] < c['phantom_instances']:
        raise ValueError('Need >=3 phantom families and >=1 sample per family')
    if c['tv_weight'] < 0 or not 0 < c['edge_threshold_fraction'] < 1:
        raise ValueError('Invalid TV weight or edge threshold')
    for angle in c['angular_ranges']:
        indices_for_range(c, angle)
    # Phantoms lie inside radius 0.40*N pixels. Check untruncated fan coverage.
    sdd = c['source_origin_mm'] + c['origin_detector_mm']
    half_fan = np.arctan(c['detectors'] * c['detector_spacing_mm'] / (2 * sdd))
    if c['image_size'] * c['pixel_size_mm'] * .40 >= c['source_origin_mm'] * np.sin(half_fan):
        raise ValueError('Detector does not cover phantom support; increase detector width')
    return c


def indices_for_range(c, angle):
    n = c['full_views'] * float(angle) / 360
    if not 0 < angle < 180 or not np.isclose(n, round(n)):
        raise ValueError('Limited range must be <180 degrees and align with full grid')
    return np.arange(round(n), dtype=np.int64)


def geometry_metadata(c):
    half = np.rad2deg(np.arctan(c['detectors'] * c['detector_spacing_mm'] /
                               (2 * (c['source_origin_mm'] + c['origin_detector_mm']))))
    return {**c, 'geometry': 'fanflat', 'angle_step_deg': 360 / c['full_views'],
            'half_fan_angle_deg': float(half), 'minimum_short_scan_deg': float(180 + 2 * half),
            'angle_interval': '[0, range), endpoint excluded',
            'sinogram_axis_order': ['view', 'detector'],
            'image_axes': 'columns increase +x; rows increase -y (ASTRA)',
            'normalization': 'clip(mu / attenuation_scale_per_mm, 0, 1); raw files retain mu',
            'target': 'noise-free 360-degree FBP; ideal phantom also retained',
            'fbp_baseline': 'FBP_CUDA on zero-filled 360-degree grid, ShortScan=False',
            'tv_algorithm': '50 alternating full SART sweeps and Chambolle TV proximal denoising',
            'tv_weight_units': 'normalized attenuation per sweep; not an exact TV minimizer'}
