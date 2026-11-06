"""Physical fan-beam geometry. ASTRA is imported only for simulation/evaluation."""
import numpy as np


class FanBeam:
    def __init__(self, config):
        try:
            import astra
        except ImportError as e:
            raise RuntimeError('Install astra-toolbox with CUDA support for fan-beam simulation') from e
        if not astra.use_cuda():
            raise RuntimeError('This fan-beam FBP pipeline requires a CUDA-enabled ASTRA installation')
        self.a, self.c = astra, config
        n, dx = config['image_size'], config['pixel_size_mm']
        self.vg = astra.create_vol_geom(n, n, -n*dx/2, n*dx/2, -n*dx/2, n*dx/2)

    def pg(self, indices=None):
        c = self.c
        if indices is None:
            indices = np.arange(c['full_views'])
        angles = np.asarray(indices, dtype=np.float64) * (2*np.pi / c['full_views'])
        return self.a.create_proj_geom('fanflat', c['detector_spacing_mm'], c['detectors'],
                                       angles, c['source_origin_mm'], c['origin_detector_mm'])

    def forward(self, image, indices=None):
        a = self.a
        pid = sid = None
        try:
            pid = a.create_projector('cuda', self.pg(indices), self.vg)
            sid, sino = a.create_sino(np.ascontiguousarray(image, dtype=np.float32), pid)
            return np.asarray(sino, dtype=np.float32)
        finally:
            if sid is not None:
                a.data2d.delete(sid)
            if pid is not None:
                a.projector.delete(pid)

    def reconstruct(self, sino, indices=None, kind='fbp', tv_weight=None):
        a, c = self.a, self.c
        if kind not in ('fbp', 'sart', 'tv_sart'):
            raise ValueError(kind)
        ids, aid = [], None
        try:
            sid = a.data2d.create('-sino', self.pg(indices), np.ascontiguousarray(sino, dtype=np.float32))
            ids.append(sid)
            rid = a.data2d.create('-vol', self.vg, 0)
            ids.append(rid)
            cfg = a.astra_dict('FBP_CUDA' if kind == 'fbp' else 'SART_CUDA')
            cfg.update(ProjectionDataId=sid, ReconstructionDataId=rid)
            cfg['option'] = ({'FilterType': 'ram-lak', 'ShortScan': False} if kind == 'fbp'
                             else {'MinConstraint': 0, 'ProjectionOrder': 'sequential'})
            aid = a.algorithm.create(cfg)
            if kind == 'fbp':
                a.algorithm.run(aid)
            elif kind == 'sart':
                a.algorithm.run(aid, c['sart_sweeps'] * len(sino))
            else:
                from skimage.restoration import denoise_tv_chambolle
                weight = c['tv_weight'] if tv_weight is None else tv_weight
                for _ in range(c['sart_sweeps']):
                    # One ASTRA iteration updates a single view. A sweep updates all views.
                    a.algorithm.run(aid, len(sino))
                    x = a.data2d.get(rid) / c['attenuation_scale_per_mm']
                    if weight > 0:
                        x = denoise_tv_chambolle(x, weight=weight,
                                                max_num_iter=c['tv_inner_iterations'], channel_axis=None)
                    a.data2d.store(rid, np.maximum(x, 0) * c['attenuation_scale_per_mm'])
            return np.asarray(a.data2d.get(rid), dtype=np.float32)
        finally:
            if aid is not None:
                a.algorithm.delete(aid)
            if ids:
                a.data2d.delete(ids)


def poisson_log(clean, photons, rng):
    expected = photons * np.exp(-np.maximum(clean, 0).astype(np.float64))
    counts = rng.poisson(expected)
    # Jeffreys-like 0.5 count floor prevents log(0); record this convention in metadata.
    return (-np.log(np.maximum(counts, .5) / photons)).astype(np.float32)


def pad_sinogram(sino, indices, full_views):
    padded = np.zeros((full_views, sino.shape[1]), np.float32)
    mask = np.zeros_like(padded)
    padded[indices] = sino
    mask[indices] = 1
    return padded, mask
