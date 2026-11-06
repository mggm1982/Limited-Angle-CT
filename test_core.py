"""Focused scientific-contract and neural-pipeline tests. No invented CT results."""
import json
import tempfile
import unittest
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np
from limited_ct.config import read_config, indices_for_range
from limited_ct.phantoms import phantom, split_plan
from limited_ct.astra_ops import poisson_log, pad_sinogram
from limited_ct.metrics import directional_error, directional_rmse

ROOT = Path(__file__).resolve().parents[1]


class CoreTests(unittest.TestCase):
    def test_phantom_level_split_exact_counts(self):
        c = read_config(ROOT/'configs/project18.json')
        plan = split_plan(c)
        self.assertEqual(Counter(r['split'] for r in plan), {'train':1750,'val':375,'test':375})
        sets = [{r['family'] for r in plan if r['split']==s} for s in ('train','val','test')]
        self.assertEqual([len(s) for s in sets], [175,38,37])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len({r['id'] for r in plan}),2500)

    def test_angles_and_padding_do_not_resample(self):
        c = read_config(ROOT/'configs/project18.json')
        for angle, n in [(90,180),(120,240)]:
            indices = indices_for_range(c,angle)
            self.assertEqual(len(indices),n)
            self.assertAlmostEqual(indices[-1]*.5,angle-.5)
            raw = np.arange(n*4,dtype=np.float32).reshape(n,4)
            padded, mask = pad_sinogram(raw,indices,720)
            np.testing.assert_array_equal(padded[indices],raw)
            self.assertEqual(mask.sum(),n*4)
            self.assertTrue(np.all(mask[n:]==0))
            self.assertTrue(np.all(padded[n:]==0))

    def test_poisson_log_finite_and_reproducible(self):
        clean = np.full((10000,), 1., np.float32)
        x = poisson_log(clean,1e6,np.random.default_rng(18))
        y = poisson_log(clean,1e6,np.random.default_rng(18))
        np.testing.assert_array_equal(x,y)
        self.assertLess(abs(x.mean()-1),.001)
        self.assertTrue(np.isfinite(poisson_log(clean*1000,1,np.random.default_rng(1))).all())

    def test_phantoms_are_reproducible_and_have_materials(self):
        x, shapes = phantom(18,0)
        y, _ = phantom(18,0)
        np.testing.assert_array_equal(x,y)
        self.assertEqual(x.shape,(256,256))
        self.assertEqual(x.dtype,np.float32)
        self.assertGreater(len(np.unique(x)),4)
        self.assertGreaterEqual(x.min(),0)
        self.assertLessEqual(x.max(),1)
        self.assertEqual({s['kind'] for s in shapes},{'rectangle','ellipse','layered_ring'})

    def test_directional_metric_reference_normals(self):
        vertical_edge = np.zeros((64,64)); vertical_edge[:,32:]=1
        p = vertical_edge+.1
        sse,count = directional_error(p,vertical_edge,bins=18)
        self.assertGreater(count[0],0)
        self.assertEqual(np.count_nonzero(count),1)
        self.assertAlmostEqual(directional_rmse(sse,count)[0],.1)
        horizontal_edge = vertical_edge.T.copy()
        _, count2 = directional_error(horizontal_edge+.1,horizontal_edge,bins=18)
        self.assertGreater(count2[9],0)
        zeros = np.zeros((64,64))
        sse,count = directional_error(zeros,zeros)
        self.assertEqual(count.sum(),0)
        self.assertTrue(np.isnan(directional_rmse(sse,count)).all())


class NeuralTests(unittest.TestCase):
    def test_shapes_gradients_and_discriminator_receptive_field(self):
        import torch
        from limited_ct.models import ResidualUNet, PatchGAN
        torch.set_num_threads(2)
        g,d = ResidualUNet(8),PatchGAN(8)
        x = torch.rand(2,1,64,64)
        prediction = g(x)
        self.assertEqual(prediction.shape,x.shape)
        torch.testing.assert_close(prediction,x)  # zero residual initialization
        loss = (prediction-torch.zeros_like(x)).abs().mean()
        loss.backward()
        self.assertGreater(g.out.weight.grad.abs().sum().item(),0)
        rf,jump = 1,1
        for layer in d.net:
            if isinstance(layer,torch.nn.Conv2d):
                rf += (layer.kernel_size[0]-1)*jump
                jump *= layer.stride[0]
        self.assertEqual(rf,70)
        self.assertEqual(d(x,prediction).shape,(2,1,6,6))

    def test_cpu_training_checkpoint_resume_and_evaluation(self):
        # Array fixture tests plumbing ONLY; these arrays are not simulated CT.
        import torch
        from limited_ct.train import train
        from limited_ct.evaluate import evaluate, infer
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'; data.mkdir()
            records=[]
            rng=np.random.default_rng(18)
            for i,split in enumerate(('train','val','test')):
                folder=data/f'sample{i}'; folder.mkdir()
                target=rng.uniform(.1,.8,(64,64)).astype(np.float32)
                for name,array in [('target',target),('phantom',target),('fbp',target*.8),
                                   ('sart',target*.9),('tv_sart',target*.95),
                                   ('fbp_raw',target*.8*.02),('artifact_mask',target*.2)]:
                    np.save(folder/f'{name}.npy',array)
                records.append({'id':f'sample{i}','family':i,'angle':90,'split':split,
                                'path':f'sample{i}','shared':f'sample{i}'})
            cfg={'image_size':64,'orientation_bins':18,'edge_threshold_fraction':.05,
                 'attenuation_scale_per_mm':.02}
            (data/'manifest.json').write_text(json.dumps({'complete':True,'config':cfg,'records':records}))
            args=SimpleNamespace(data=str(data),angle=90,model='gan',output=str(root/'run'),input='fbp',
                epochs=1,batch_size=1,base=8,lr=2e-4,lambda_l1=100.,lambda_adv=1.,lambda_ssim=0.,
                artifact_weight=0.,jitter=0.,seed=18,device='cpu',resume=None,tensorboard=False)
            train(args)
            self.assertTrue((root/'run/best.pt').exists())
            args.resume=str(root/'run/last.pt'); args.epochs=2
            train(args)
            ckpt=torch.load(root/'run/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ckpt['epoch'],1)
            self.assertIsNotNone(ckpt['discriminator'])
            ev=SimpleNamespace(data=str(data),angle=90,split='test',checkpoint=[f'gan={root}/run/best.pt'],
                output=str(root/'evaluation'),device='cpu',panels=1,projection_residual=False)
            evaluate(ev)
            self.assertTrue((root/'evaluation/directional_ideal.png').exists())
            self.assertTrue((data/'evaluation_lock_90.json').exists())
            infer(SimpleNamespace(checkpoint=str(root/'run/best.pt'),input=str(data/'sample2/fbp.npy'),
                                  output=str(root/'prediction.npy'),device='cpu'))
            self.assertEqual(np.load(root/'prediction.npy').shape,(64,64))
            args.model='unet'; args.output=str(root/'unet'); args.resume=None; args.epochs=1
            train(args)
            plain=torch.load(root/'unet/best.pt',map_location='cpu',weights_only=False)
            self.assertIsNone(plain['discriminator'])
            ev.output=str(root/'second_evaluation')
            with self.assertRaises(FileExistsError):
                evaluate(ev)


if __name__=='__main__':
    unittest.main()
