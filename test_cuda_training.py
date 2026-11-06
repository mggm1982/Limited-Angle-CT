"""Opt-in real CUDA neural training check, independent of ASTRA."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from test_local_workflow import make_parts
from limited_ct.import_dataset import import_dataset


@unittest.skipUnless(os.environ.get('RUN_LOCAL_CUDA_TESTS')=='1','requires a real local CUDA GPU')
class CudaTrainingTests(unittest.TestCase):
    def test_gan_amp_checkpoint_and_resume(self):
        import torch
        from limited_ct.train import train
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=make_parts(root)
            data=root/'data';import_dataset(a,b,data,expected_samples=3)
            args=SimpleNamespace(data=str(data),angle=90,model='gan',output=str(root/'run'),input='fbp',
                epochs=1,batch_size=1,base=8,lr=2e-4,lambda_l1=100.,lambda_adv=1.,lambda_ssim=0.,
                artifact_weight=0.,jitter=0.,seed=18,device='cuda:0',resume=None,tensorboard=False,
                amp=True,num_workers=0)
            train(args)
            ckpt=torch.load(root/'run/last.pt',map_location='cpu',weights_only=False)
            self.assertTrue(ckpt['grad_scaler'])
            self.assertTrue(ckpt['settings']['amp'])
            args.resume=str(root/'run/last.pt');args.epochs=2
            train(args)
            ckpt=torch.load(root/'run/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['epoch'],1)
