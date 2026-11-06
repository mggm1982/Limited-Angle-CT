import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


def manifest(root):
    m = json.loads((Path(root)/'manifest.json').read_text())
    if not m.get('complete'):
        raise ValueError('Dataset generation is incomplete')
    seen = {}
    for r in m['records']:
        if r['family'] in seen and seen[r['family']] != r['split']:
            raise ValueError('Phantom-family leakage across splits')
        seen[r['family']] = r['split']
    return m


class Pairs(Dataset):
    def __init__(self, root, angle, split, input_kind='fbp', intensity_jitter=0):
        if input_kind not in ('fbp', 'sart'):
            raise ValueError('input_kind must be fbp or sart')
        self.root, self.input_kind = Path(root), input_kind
        self.records = [r for r in manifest(root)['records'] if r['angle'] == angle and r['split'] == split]
        if not self.records:
            raise ValueError(f'No {split} records for {angle} degrees')
        self.jitter = intensity_jitter if split == 'train' else 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        x = np.load(self.root/r['path']/f'{self.input_kind}.npy', allow_pickle=False)
        y = np.load(self.root/r['shared']/'target.npy', allow_pickle=False)
        mask_path = self.root/r['path']/'artifact_mask.npy'
        if mask_path.is_file():
            w = np.load(mask_path, allow_pickle=False)
        else:
            # Compact dataset ZIPs omit this exactly recoverable training label.
            fbp = x if self.input_kind == 'fbp' else np.load(self.root/r['path']/'fbp.npy', allow_pickle=False)
            w = np.abs(fbp-y).astype(np.float32)
        if self.jitter:
            factor = float(torch.empty(()).uniform_(1-self.jitter, 1+self.jitter))
            x, y = np.clip(x*factor, 0, 1), np.clip(y*factor, 0, 1)
        return {'input': torch.from_numpy(x.copy()).unsqueeze(0),
                'target': torch.from_numpy(y.copy()).unsqueeze(0),
                'weight': torch.from_numpy(w.copy()).unsqueeze(0), 'id': r['id']}
