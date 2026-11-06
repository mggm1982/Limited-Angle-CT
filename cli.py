import argparse


def main():
    parser = argparse.ArgumentParser(description='Project 18: limited-angle fan-beam CT')
    sub = parser.add_subparsers(dest='command', required=True)
    gen = sub.add_parser('generate', help='Generate synthetic pairs using ASTRA CUDA')
    gen.add_argument('--config', default='configs/project18.json')
    gen.add_argument('--output', default='data/project18')
    gen.add_argument('--plan-only', action='store_true', help='Validate geometry/splits without ASTRA')
    tr = sub.add_parser('train', help='Train one model for one angular range')
    tr.add_argument('--data', default='data/project18')
    tr.add_argument('--angle', type=int, choices=[90,120], required=True)
    tr.add_argument('--model', choices=['unet','gan'], required=True)
    tr.add_argument('--output', required=True)
    tr.add_argument('--input', choices=['fbp','sart'], default='fbp')
    tr.add_argument('--epochs', type=int, default=180)
    tr.add_argument('--batch-size', type=int, default=8)
    tr.add_argument('--base', type=int, default=64)
    tr.add_argument('--lr', type=float, default=2e-4)
    tr.add_argument('--lambda-l1', type=float, default=100)
    tr.add_argument('--lambda-adv', type=float, default=1)
    tr.add_argument('--lambda-ssim', type=float, default=0)
    tr.add_argument('--artifact-weight', type=float, default=0)
    tr.add_argument('--jitter', type=float, default=0, help='Paired intensity jitter only; no rotations/flips')
    tr.add_argument('--seed', type=int, default=18)
    tr.add_argument('--device', default='cuda:0')
    tr.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                    help='CUDA automatic mixed precision and gradient scaling')
    tr.add_argument('--num-workers', type=int, default=0)
    tr.add_argument('--resume')
    tr.add_argument('--tensorboard', action='store_true')
    ev = sub.add_parser('evaluate', help='Compare baselines/checkpoints; validate before final test')
    ev.add_argument('--data', default='data/project18')
    ev.add_argument('--angle', type=int, choices=[90,120], required=True)
    ev.add_argument('--split', choices=['val','test'], default='val')
    ev.add_argument('--checkpoint', action='append', help='label=path.pt; repeat for model comparisons')
    ev.add_argument('--output', required=True)
    ev.add_argument('--device', default='cuda')
    ev.add_argument('--baselines', nargs='+', choices=['fbp','sart','tv_sart'], help='Default: dataset-declared available baselines')
    ev.add_argument('--panels', type=int, default=6)
    ev.add_argument('--projection-residual', action='store_true', help='ASTRA CUDA measured-data check')
    inf = sub.add_parser('infer')
    inf.add_argument('--checkpoint', required=True)
    inf.add_argument('--input', required=True, help='Normalized 2D FBP/SART .npy matching checkpoint')
    inf.add_argument('--output', required=True)
    inf.add_argument('--device', default='cuda')
    tv = sub.add_parser('tune-tv', help='Select TV strength using validation data only')
    tv.add_argument('--data', default='data/project18')
    tv.add_argument('--angle', type=int, choices=[90,120], required=True)
    tv.add_argument('--weights', type=float, nargs='+', default=[.0005,.001,.002,.005,.01])
    tv.add_argument('--output', required=True)
    ap = sub.add_parser('apply-tv', help='Apply validation-selected TV strength to all splits')
    ap.add_argument('--data', default='data/project18')
    ap.add_argument('--selection', required=True, help='selection.json from tune-tv')
    imp = sub.add_parser('import-dataset', help='Merge and validate the two existing training ZIPs')
    imp.add_argument('--part1', required=True)
    imp.add_argument('--part2', required=True)
    imp.add_argument('--output', default='data/limited_angle_ct_1000')
    args = parser.parse_args()
    if args.command == 'import-dataset':
        from .import_dataset import import_dataset
        import_dataset(args.part1, args.part2, args.output)
    elif args.command == 'generate':
        from .generate import generate
        generate(args.config, args.output, args.plan_only)
    elif args.command == 'train':
        from .train import train
        if not 0 <= args.jitter < 1:
            parser.error('--jitter must be in [0,1)')
        train(args)
    elif args.command in ('evaluate','infer'):
        from .evaluate import evaluate, infer
        (evaluate if args.command == 'evaluate' else infer)(args)
    elif args.command in ('tune-tv','apply-tv'):
        from .tv import tune, apply
        (tune if args.command == 'tune-tv' else apply)(args)


if __name__ == '__main__':
    main()
