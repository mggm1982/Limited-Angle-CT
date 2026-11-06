"""Cross-platform local CUDA runner for the two downloaded FBP ZIP parts."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

PROJECT=Path(__file__).resolve().parents[1]


def load_settings(path, overrides=None):
    c=json.loads(Path(path).read_text())
    for key,value in (overrides or {}).items():
        if value is not None:c[key]=value
    for key in ('part1','part2','dataset_dir','run_dir','results_dir'):
        p=Path(c[key]).expanduser()
        c[key]=str(p if p.is_absolute() else PROJECT/p)
    if not str(c['device']).startswith('cuda'):
        raise ValueError('The local CUDA configuration requires device cuda:N')
    if c['epochs']<=0 or c['batch_size']<=0 or c['num_workers']<0:
        raise ValueError('Invalid training size or worker setting')
    if c['lr']<=0 or c['lambda_l1']<0:
        raise ValueError('Learning rate must be positive and L1 weight nonnegative')
    if c['base']<8 or c['base']%8:
        raise ValueError('base must be a positive multiple of 8')
    if not c['angles'] or any(a not in (90,120) for a in c['angles']) or len(set(c['angles']))!=len(c['angles']):
        raise ValueError('angles must be a nonempty selection of 90 and 120')
    if not c['models'] or any(m not in ('unet','gan') for m in c['models']) or len(set(c['models']))!=len(c['models']):
        raise ValueError('models must be a nonempty selection of unet and gan')
    weights=c['gan_adversarial_weights']
    if not weights or len(set(weights))!=len(weights) or any(w<0 for w in weights):
        raise ValueError('GAN adversarial weights must be nonnegative and distinct')
    return c


def planned_runs(c):
    for angle in c['angles']:
        for model in c['models']:
            weights=c['gan_adversarial_weights'] if model=='gan' else [0.0]
            for weight in weights:
                label='unet' if model=='unet' else f'gan_adv{weight:g}'
                out=Path(c['run_dir'])/f'{angle}_{label}'
                command=[sys.executable,'-m','limited_ct.cli','train','--data',c['dataset_dir'],
                    '--angle',str(angle),'--model',model,'--input','fbp','--output',str(out),
                    '--device',c['device'],'--epochs',str(c['epochs']),'--batch-size',str(c['batch_size']),
                    '--base',str(c['base']),'--num-workers',str(c['num_workers']),
                    '--lr',str(c['lr']),'--seed',str(c['seed']),
                    '--lambda-l1',str(c['lambda_l1']),'--lambda-adv',str(weight)]
                if c['amp']:command.append('--amp')
                if c['tensorboard']:command.append('--tensorboard')
                if c['resume'] and (out/'last.pt').is_file():
                    command+=['--resume',str(out/'last.pt')]
                yield {'angle':angle,'label':label,'output':out,'command':command}


def run_command(command):
    print('\n'+shlex.join(command),flush=True)
    subprocess.run(command,cwd=PROJECT,check=True)


def main():
    parser=argparse.ArgumentParser(description='Train existing Project 18 ZIP data on a local NVIDIA CUDA GPU')
    parser.add_argument('--config',default=str(PROJECT/'configs/local_cuda_1000.json'))
    parser.add_argument('--part1');parser.add_argument('--part2')
    parser.add_argument('--data-dir',dest='dataset_dir')
    parser.add_argument('--device')
    parser.add_argument('--epochs',type=int);parser.add_argument('--batch-size',type=int)
    parser.add_argument('--base',type=int);parser.add_argument('--num-workers',type=int)
    parser.add_argument('--run-dir');parser.add_argument('--results-dir')
    parser.add_argument('--amp',action=argparse.BooleanOptionalAction,default=None)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only',action='store_true',help='Extract/validate both ZIP parts without importing PyTorch')
    modes.add_argument('--check-only',action='store_true',help='Check CUDA and data paths without training or extraction')
    modes.add_argument('--dry-run',action='store_true',help='Print training commands without touching datasets or using CUDA')
    modes.add_argument('--final-test',action='store_true',help='Evaluate existing best checkpoints once, without training')
    args=parser.parse_args()
    keys=('part1','part2','dataset_dir','device','epochs','batch_size','base','num_workers','run_dir','results_dir','amp')
    c=load_settings(args.config,{k:getattr(args,k) for k in keys})
    runs=list(planned_runs(c))
    if args.dry_run:
        print('Existing paired data:',c['dataset_dir'])
        print('ZIP 1:',c['part1']);print('ZIP 2:',c['part2'])
        for r in runs:print(shlex.join(r['command']))
        print('Dry run only: no extraction, CUDA execution, or model training performed.')
        return
    if not args.prepare_only:
        try:
            from .device import resolve_device
        except ModuleNotFoundError as e:
            raise RuntimeError('Install CUDA-enabled PyTorch and requirements-local.txt in this Python environment.') from e
        c['device']=str(resolve_device(c['device']))
    from .import_dataset import import_dataset,validate_dataset
    dataset=Path(c['dataset_dir'])
    if args.check_only:
        if dataset.exists():
            print(json.dumps(validate_dataset(dataset,check_arrays=False),indent=2))
        else:
            for key in ('part1','part2'):
                if not Path(c[key]).is_file():raise FileNotFoundError(c[key])
            print('CUDA ready and both ZIP paths exist. Run normally to extract and train.')
        return
    if dataset.exists():
        report=validate_dataset(dataset)
    else:
        report=import_dataset(c['part1'],c['part2'],dataset)
    print('Using existing FBP data:',json.dumps(report),flush=True)
    if args.prepare_only:return
    Path(c['run_dir']).mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    (Path(c['run_dir'])/f'launch_{stamp}.json').write_text(json.dumps(c,indent=2))
    runs=list(planned_runs(c))
    if args.final_test:
        for r in runs:
            if not (r['output']/'best.pt').is_file():raise FileNotFoundError(r['output']/'best.pt')
    else:
        for r in runs:run_command(r['command'])
    if args.final_test or c['validate_after_training']:
        split='test' if args.final_test else 'val'
        for angle in c['angles']:
            out=Path(c['results_dir'])/f'{angle}_{split}_{stamp}'
            command=[sys.executable,'-m','limited_ct.cli','evaluate','--data',c['dataset_dir'],
                '--angle',str(angle),'--split',split,'--device',c['device'],'--baselines','fbp',
                '--output',str(out)]
            for r in runs:
                if r['angle']==angle:command+=['--checkpoint',f'{r["label"]}={r["output"]}/best.pt']
            run_command(command)
    print('Finished. Research weights and logs:',c['run_dir'],flush=True)


if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as e:
        raise SystemExit(str(e)) from e
