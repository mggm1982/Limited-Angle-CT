"""Analytic 2D multi-material families; all variants share a family ID."""
import numpy as np


def phantom(seed, variant, size=256):
    # Geometry is fixed per family. Only material intensities change across variants.
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    image = np.zeros((size, size), np.float32)
    support = x*x + y*y < .78**2
    image[support] = .10
    shapes = []
    for k in range(18):
        theta = rng.uniform(0, np.pi)
        cx, cy = rng.uniform(-.42, .42, 2)
        u = (x-cx)*np.cos(theta) + (y-cy)*np.sin(theta)
        v = -(x-cx)*np.sin(theta) + (y-cy)*np.cos(theta)
        a, b = rng.uniform(.07, .25), rng.uniform(.015, .065)
        if k % 3 == 0:
            mask = (np.abs(u) < a) & (np.abs(v) < b)
            kind = 'rectangle'
        elif k % 3 == 1:
            mask = (u/a)**2 + (v/b)**2 < 1
            kind = 'ellipse'
        else:
            radius = (u/a)**2 + (v/(a*.8))**2
            mask = (radius < 1) & (radius > .55)
            kind = 'layered_ring'
        material = float(rng.choice([.25, .45, .65, .85]))
        image[mask & support] = material
        if kind == 'layered_ring':
            # Distinct material in an inner annulus stresses adjacent curved boundaries.
            inner = (radius <= .55) & (radius > .25) & support
            image[inner] = .65 if material < .5 else .25
        shapes.append({'kind': kind, 'angle_deg': float(np.rad2deg(theta)),
                       'center': [float(cx), float(cy)], 'material': material})
    jitter = np.random.default_rng(np.random.SeedSequence([seed, variant, 1821]))
    # Piecewise constant materials remain piecewise constant, with identical boundaries.
    for material in np.unique(image[image > 0]):
        image[image == material] = np.clip(material * jitter.uniform(.92, 1.08), 0, 1)
    return image.astype(np.float32), shapes


def split_plan(c):
    """Allocate exact 70/15/15 image totals while never splitting a family.

    250 families -> 175/38/37; 2500 images -> 1750/375/375.
    Validation/test families have 9-11 variants where necessary.
    """
    n, total = c['phantom_instances'], c['samples_per_condition']
    n_train = min(n - 2, max(1, round(n * .70)))
    n_val = min(n - n_train - 1, max(1, round(n * .15)))
    groups = [n_train, n_val, n - n_train - n_val]
    t_train = min(total - groups[1] - groups[2], max(groups[0], round(total * .70)))
    t_val = min(total - t_train - groups[2], max(groups[1], round(total * .15)))
    counts = [t_train, t_val, total - t_train - t_val]
    families = np.random.default_rng(c['seed']).permutation(n)
    out, offset = [], 0
    for split, ng, ns in zip(('train', 'val', 'test'), groups, counts):
        for j, family in enumerate(families[offset:offset+ng]):
            variants = ns // ng + (j < ns % ng)
            for v in range(variants):
                out.append({'id': f'p{family:04d}_v{v:02d}', 'family': int(family),
                            'variant': v, 'split': split})
        offset += ng
    return out
