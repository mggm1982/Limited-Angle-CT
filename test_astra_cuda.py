"""Opt-in real physics sanity test: RUN_ASTRA_CUDA_TESTS=1 python -m unittest discover -s tests."""
import os
import unittest
from pathlib import Path
import numpy as np
from limited_ct.config import read_config
from limited_ct.astra_ops import FanBeam


@unittest.skipUnless(os.environ.get('RUN_ASTRA_CUDA_TESTS')=='1','requires real ASTRA CUDA GPU')
class AstraTests(unittest.TestCase):
    def test_disk_full_scan_scale_and_limited_reconstruction(self):
        c=read_config(Path(__file__).resolve().parents[1]/'configs/project18.json')
        c['sart_sweeps']=1
        op=FanBeam(c)
        y,x=np.mgrid[:256,:256]
        disk=(((x-127.5)**2+(y-127.5)**2)<60**2).astype(np.float32)*.01
        sino=op.forward(disk)
        self.assertEqual(sino.shape,(720,512))
        self.assertAlmostEqual(float(sino[0].max()),1.2,delta=.05)
        fbp=op.reconstruct(sino)
        self.assertLess(float(np.mean(abs(fbp[100:150,100:150]-.01))),.001)
        for kind in ('sart','tv_sart'):
            recon=op.reconstruct(sino[:180],np.arange(180),kind)
            self.assertTrue(np.isfinite(recon).all())
            self.assertGreater(recon.max(),0)
