"""Metrics in fixed normalized attenuation units; never per-image min/max."""
import numpy as np
from scipy.ndimage import sobel


def image_metrics(prediction, target):
    from skimage.metrics import structural_similarity
    p, t = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    mse = float(np.mean((p-t)**2))
    return {'rmse': float(np.sqrt(mse)), 'mae': float(np.mean(np.abs(p-t))),
            'psnr': float(-10*np.log10(max(mse, 1e-12))),
            'ssim': float(structural_similarity(t, p, data_range=1.0))}


def directional_error(prediction, reference, bins=18, edge_fraction=.05, center_deg=0):
    """Pixel-error RMSE binned by REFERENCE edge-normal orientation.

    Physical +x points right; +y points up, hence gy=-d/drow. Gradient normals
    are modulo 180 degrees. center_deg sets the source-angle center as an
    approximate central-ray measured-normal reference (not an exact fan wedge).
    Return SSE and counts so aggregation is pixel weighted across samples.
    """
    ref = np.asarray(reference, dtype=np.float64)
    gx, gy = sobel(ref, axis=1)/8, -sobel(ref, axis=0)/8
    magnitude = np.hypot(gx, gy)
    valid = magnitude > max(float(magnitude.max())*edge_fraction, 1e-12)
    # ASTRA fanflat theta=0 central ray is vertical; ray normal is +x.
    orientation = (np.rad2deg(np.arctan2(gy, gx))-center_deg) % 180
    index = np.minimum((orientation / (180/bins)).astype(int), bins-1)
    squared = (np.asarray(prediction, np.float64)-ref)**2
    count = np.bincount(index[valid].ravel(), minlength=bins)
    sse = np.bincount(index[valid].ravel(), weights=squared[valid].ravel(), minlength=bins)
    return sse, count


def directional_rmse(sse, count):
    return np.sqrt(np.divide(sse, count, out=np.full_like(sse, np.nan, dtype=float), where=count>0))
