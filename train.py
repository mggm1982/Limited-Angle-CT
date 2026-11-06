import csv
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import Pairs, manifest
from .models import ResidualUNet, PatchGAN, ssim_loss
from .metrics import image_metrics
from .device import resolve_device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def manifest_hash(root):
    return hashlib.sha256((Path(root)/'manifest.json').read_bytes()).hexdigest()


def train(args):
    seed_all(args.seed)
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0:
        raise ValueError('epochs, batch size and learning rate must be positive')
    if min(args.lambda_l1, args.lambda_adv, args.lambda_ssim, args.artifact_weight) < 0:
        raise ValueError('Loss weights cannot be negative')
    args.amp = getattr(args, 'amp', False)
    args.num_workers = getattr(args, 'num_workers', 0)
    if args.num_workers < 0:
        raise ValueError('num_workers cannot be negative')
    device = resolve_device(args.device)
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'best.pt').exists() and not args.resume:
        raise FileExistsError('Existing run; use --resume or a new output directory')
    digest = manifest_hash(args.data)
    dataset_manifest = manifest(args.data)
    config = dataset_manifest['config']
    train_set = Pairs(args.data, args.angle, 'train', args.input, args.jitter)
    val_set = Pairs(args.data, args.angle, 'val', args.input)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                       pin_memory=device.type == 'cuda')
    dl = DataLoader(train_set, shuffle=True, **loader_args)
    vl = DataLoader(val_set, shuffle=False, **loader_args)
    print(f'{args.angle} degrees, {args.model}: {len(train_set)} train / {len(val_set)} val; '
          f'device={device}; AMP={use_amp}; data backend='
          f'{dataset_manifest.get("reconstruction_backend", "see geometry.json")}', flush=True)
    g = ResidualUNet(args.base).to(device)
    d = PatchGAN(args.base).to(device) if args.model == 'gan' else None
    go = torch.optim.Adam(g.parameters(), lr=args.lr, betas=(.5, .999))
    do = torch.optim.Adam(d.parameters(), lr=args.lr, betas=(.5, .999)) if d else None
    bce = nn.BCEWithLogitsLoss()
    best, start = float('inf'), 0
    settings = vars(args).copy()
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ('model', 'base', 'angle', 'input', 'lambda_l1', 'lambda_adv',
                    'lambda_ssim', 'artifact_weight', 'jitter', 'batch_size', 'lr', 'seed'):
            if ckpt['settings'][key] != settings[key]:
                raise ValueError(f'Resume setting mismatch: {key}')
        if ckpt['manifest_sha256'] != digest:
            raise ValueError('Cannot resume on a different dataset manifest')
        g.load_state_dict(ckpt['generator'])
        go.load_state_dict(ckpt['generator_optimizer'])
        if d:
            d.load_state_dict(ckpt['discriminator'])
            do.load_state_dict(ckpt['discriminator_optimizer'])
        if ckpt['settings'].get('amp', False) != args.amp:
            raise ValueError('Resume AMP setting mismatch. Keep the original precision or use a new run.')
        if ckpt.get('grad_scaler'):
            scaler.load_state_dict(ckpt['grad_scaler'])
        best, start = ckpt['best_val_l1'], ckpt['epoch'] + 1
        torch.set_rng_state(ckpt['torch_rng'].cpu())
        np.random.set_state(ckpt['numpy_rng'])
        random.setstate(ckpt['python_rng'])
        if torch.cuda.is_available() and ckpt.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
    (out/'settings.json').write_text(json.dumps(settings, indent=2))
    writer = None
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out/'tensorboard')
    for epoch in range(start, args.epochs):
        g.train()
        if d:
            d.train()
        losses = np.zeros(4, dtype=float)
        for batch in dl:
            x, y, w = (batch[k].to(device, non_blocking=device.type == 'cuda')
                       for k in ('input', 'target', 'weight'))
            go.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                fake = g(x)
            loss_d = torch.zeros((), device=device)
            if d:
                d.requires_grad_(True)
                do.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    real_logits, fake_logits = d(x, y), d(x, fake.detach())
                    loss_d = .5 * (bce(real_logits, torch.ones_like(real_logits)) +
                                    bce(fake_logits, torch.zeros_like(fake_logits)))
                if not torch.isfinite(loss_d):
                    raise RuntimeError('Nonfinite discriminator loss; stop and inspect inputs/precision')
                scaler.scale(loss_d).backward()
                scaler.step(do)
                d.requires_grad_(False)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                l1 = ((fake-y).abs()*(1+args.artifact_weight*w)).mean()
                adv = torch.zeros((), device=device)
                if d:
                    logits = d(x, fake)
                    adv = bce(logits, torch.ones_like(logits))
                ssim = ssim_loss(fake, y) if args.lambda_ssim else torch.zeros((), device=device)
                loss_g = args.lambda_l1*l1 + (args.lambda_adv*adv if d else 0) + args.lambda_ssim*ssim
            if not torch.isfinite(loss_g):
                raise RuntimeError('Nonfinite generator loss; stop and inspect inputs/precision')
            scaler.scale(loss_g).backward()
            scaler.step(go)
            # One scaler update after both GAN optimizers have stepped.
            scaler.update()
            losses += np.array([loss_g.item(), l1.item(), adv.item(), loss_d.item()])*len(x)
        losses /= len(train_set)
        g.eval()
        val_l1, val_ssim = 0., 0.
        with torch.no_grad():
            for batch in vl:
                y = batch['target'].numpy()[:, 0]
                p = g(batch['input'].to(device)).cpu().numpy()[:, 0]
                # Select on raw L1 (no clamp hiding overshoot); log clipped SSIM for display.
                val_l1 += float(np.abs(p-y).sum())
                val_ssim += sum(image_metrics(np.clip(a, 0, 1), b)['ssim'] for a,b in zip(p,y))
        val_l1 /= len(val_set)*config['image_size']**2
        val_ssim /= len(val_set)
        improved = val_l1 < best
        best = min(best, val_l1)
        state = {'generator': g.state_dict(), 'discriminator': d.state_dict() if d else None,
                 'generator_optimizer': go.state_dict(), 'discriminator_optimizer': do.state_dict() if do else None,
                 'grad_scaler': scaler.state_dict(),
                 'dataset_metadata': {k:dataset_manifest.get(k) for k in
                                      ('dataset_kind','reconstruction_backend','available_reconstructions')},
                 'epoch': epoch, 'best_val_l1': best, 'settings': settings,
                 'manifest_sha256': digest, 'config': config,
                 'torch_rng': torch.get_rng_state(), 'numpy_rng': np.random.get_state(),
                 'python_rng': random.getstate(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
        torch.save(state, out/'last.pt')
        if improved:
            torch.save(state, out/'best.pt')
        values = [epoch+1, *losses, val_l1, val_ssim]
        with (out/'history.csv').open('a', newline='') as f:
            csvwriter = csv.writer(f)
            if f.tell() == 0:
                csvwriter.writerow(['epoch','g_loss','train_l1','g_adv','d_loss','val_l1','val_ssim'])
            csvwriter.writerow(values)
        if writer:
            for tag, value in zip(['g_loss','train_l1','g_adv','d_loss','val_l1','val_ssim'], values[1:]):
                writer.add_scalar(tag, value, epoch+1)
        print(f'epoch={epoch+1} val_l1={val_l1:.6f} val_ssim={val_ssim:.4f} best={best:.6f}', flush=True)
    if writer:
        writer.close()
