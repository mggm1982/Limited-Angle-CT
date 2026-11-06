"""Explicit device selection with no silent CUDA-to-CPU fallback."""
import torch


def resolve_device(name, probe=True):
    device=torch.device(name)
    if device.type=='cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Install a CUDA-enabled PyTorch build for your '
                'NVIDIA driver and activate that Python environment. Check: '
                'python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"')
        index=0 if device.index is None else device.index
        if not 0<=index<torch.cuda.device_count():
            raise ValueError(f'cuda:{index} is invalid; {torch.cuda.device_count()} CUDA device(s) visible')
        device=torch.device('cuda',index)
        torch.cuda.set_device(device)
        if probe:
            try:
                (torch.ones(1,device=device)+1).sum().item()
            except RuntimeError as e:
                raise RuntimeError(f'CUDA device {device} cannot execute tensors. Check NVIDIA driver '
                    'and PyTorch GPU architecture compatibility.') from e
        props=torch.cuda.get_device_properties(device)
        print(f'Using {device}: {props.name}, {props.total_memory/1024**3:.1f} GiB VRAM',flush=True)
    elif device.type!='cpu':
        raise ValueError('Supported devices: cpu or cuda:N')
    return device
