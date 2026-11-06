import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from limited_ct.import_dataset import import_dataset,validate_dataset
from limited_ct.local_run import load_settings,planned_runs,PROJECT


def make_parts(root):
    records=[];entries={}
    def array(name,a):
        b=io.BytesIO();np.save(b,np.asarray(a,np.float32),allow_pickle=False);entries[name]=b.getvalue()
    for i,split in enumerate(('train','val','test')):
        identity=f'p{i:04d}_v00';shared=f'shared/{identity}'
        rng=np.random.default_rng(i);ideal=rng.uniform(.1,.8,(64,64)).astype(np.float32)
        array(f'{shared}/phantom.npy',ideal);array(f'{shared}/target.npy',ideal)
        for angle in (90,120):
            folder=f'angle_{angle}/{identity}'
            array(f'{folder}/fbp.npy',ideal*.8)
            records.append({'id':identity,'family':i,'variant':0,'split':split,
                            'angle':angle,'path':folder,'shared':shared})
    cfg={'image_size':64,'orientation_bins':18,'edge_threshold_fraction':.05,
         'attenuation_scale_per_mm':.02}
    manifest={'complete':True,'config':cfg,'records':records,
              'available_reconstructions':['fbp'],'reconstruction_backend':'cpu_fixture'}
    for part in (1,2):
        with zipfile.ZipFile(root/f'part{part}.zip','w') as z:
            z.writestr('limited_angle_ct_1000/manifest.json',json.dumps(manifest))
            z.writestr('limited_angle_ct_1000/README.md','Common header')
            for name,blob in entries.items():
                if ('p0002' in name)==(part==2):z.writestr('limited_angle_ct_1000/'+name,blob)
    return root/'part1.zip',root/'part2.zip'


class LocalWorkflowTests(unittest.TestCase):
    def test_archive_merge_lazy_mask_and_fbp_only_evaluation(self):
        from limited_ct.data import Pairs
        from limited_ct.evaluate import evaluate
        import torch
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=make_parts(root);out=root/'dataset'
            report=import_dataset(a,b,out,expected_samples=3)
            self.assertEqual(report['paired_cases'],6)
            # A repeat verifies and preserves the existing dataset.
            self.assertEqual(import_dataset(a,b,out,expected_samples=3),report)
            dataset=Pairs(out,90,'train','fbp');item=dataset[0]
            self.assertEqual(item['input'].shape,(1,64,64))
            torch.testing.assert_close(item['weight'],abs(item['input']-item['target']))
            self.assertFalse((out/'angle_90/p0000_v00/artifact_mask.npy').exists())
            args=SimpleNamespace(data=str(out),angle=90,split='val',checkpoint=[],
                output=str(root/'evaluation'),device='cpu',panels=0,projection_residual=False)
            evaluate(args)
            with (root/'evaluation/summary.csv').open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([r['method'] for r in rows],['fbp'])
            provenance=json.loads((root/'evaluation/evaluation.json').read_text())
            self.assertFalse(provenance['raw_fbp_evaluated'])
            self.assertEqual(provenance['dataset_backend'],'cpu_fixture')

    def test_mismatched_parts_and_path_traversal_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=make_parts(root)
            with self.assertRaises(ValueError):import_dataset(a,a,root/'same',3)
            with zipfile.ZipFile(b,'a') as z:
                z.writestr('limited_angle_ct_1000/../../escaped.txt','bad')
            with self.assertRaises(ValueError):import_dataset(a,b,root/'unsafe',3)
            self.assertFalse((root/'escaped.txt').exists())
            self.assertFalse((root/'unsafe').exists())

    def test_incomplete_and_conflicting_parts_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=make_parts(root)
            with zipfile.ZipFile(root/'partial.zip','w') as z:
                with zipfile.ZipFile(b) as orig:
                    z.writestr('limited_angle_ct_1000/manifest.json',orig.read('limited_angle_ct_1000/manifest.json'))
            with self.assertRaises(ValueError):import_dataset(a,root/'partial.zip',root/'incomplete',3)
            with zipfile.ZipFile(root/'conflict.zip','w') as z:
                with zipfile.ZipFile(b) as orig:
                    for info in orig.infolist():
                        value=orig.read(info.filename)
                        if info.filename.endswith('/README.md'):value=b'Conflicting header'
                        z.writestr(info.filename,value)
            with self.assertRaises(ValueError):import_dataset(a,root/'conflict.zip',root/'conflict',3)
            self.assertFalse((root/'conflict').exists())

    def test_no_cuda_fallback_and_invalid_gpu_index(self):
        from limited_ct.device import resolve_device
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA is unavailable'):resolve_device('cuda:0')
        with patch('torch.cuda.is_available',return_value=True),patch('torch.cuda.device_count',return_value=1):
            with self.assertRaisesRegex(ValueError,'invalid'):resolve_device('cuda:2',probe=False)

    def test_config_plans_four_cuda_runs_without_generation(self):
        config=load_settings(PROJECT/'configs/local_cuda_1000.json')
        runs=list(planned_runs(config))
        self.assertEqual(len(runs),4)
        self.assertEqual({r['angle'] for r in runs},{90,120})
        for r in runs:
            self.assertIn('--amp',r['command'])
            self.assertIn('cuda:0',r['command'])
            self.assertIn('fbp',r['command'])
            self.assertNotIn('generate',r['command'])
        config=load_settings(PROJECT/'configs/local_cuda_1000_full_study.json')
        self.assertEqual(len(list(planned_runs(config))),8)


if __name__=='__main__':unittest.main()
