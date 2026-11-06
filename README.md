# Project 18: local CUDA training on the two 1,000-image dataset parts

**Run `python run_local_cuda.py` from this project folder.**

This version trains on the existing downloads:

- `Limited_Angle_CT_1000_Training_Part1.zip`
- `Limited_Angle_CT_1000_Training_Part2.zip`

The runner merges and validates both parts, trains the 90°/120° residual U-Net
and GAN on your local NVIDIA GPU, and evaluates validation images against FBP.
It does not regenerate projections or call ASTRA during this workflow.
Actual trained weights are produced on your computer when you run training.

## 1. Prepare a CUDA-enabled Python environment

Python 3.11 or 3.12 is a practical choice. Create and activate an environment:

```bash
python -m venv .venv
# Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
```

Install **CUDA-enabled PyTorch** using the command selected for your OS,
GPU and driver at https://pytorch.org/get-started/locally/ . Choose a compatible
CUDA build rather than the CPU build. This project uses PyTorch >=2.3.
Then install the other local-training dependencies:

```bash
python -m pip install -r requirements-local.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No usable CUDA GPU')"
```

The availability check must print `True`. The launcher also executes a tensor
on the selected device and reports its name and VRAM. An installed CUDA toolkit
alone does not guarantee that the active Python environment can use the GPU.
The launcher fails with an explanation if CUDA is unavailable; it never silently
switches a local CUDA job to CPU. ASTRA is optional for existing-image training.

## 2. Put both ZIPs in `datasets/`

```text
limited_angle_ct/
  run_local_cuda.py
  configs/local_cuda_1000.json
  datasets/
    Limited_Angle_CT_1000_Training_Part1.zip
    Limited_Angle_CT_1000_Training_Part2.zip
```

The data archives are separate downloads and are not duplicated in the code ZIP.
Do not unzip one part into a separate dataset and train on it alone. Both
archives supply one combined dataset. The importer checks all declared pairs,
array shapes/types/ranges, duplicate records, and phantom-family split isolation.
It merges matching common headers, rejects conflicting/traversal entries, and
never executes scripts from the dataset ZIPs.

## 3. Start local GPU training

```bash
python run_local_cuda.py
```

The sequence is:

1. Validate `cuda:0` and print the GPU identity.
2. Combine both ZIPs into `data/limited_angle_ct_1000/` and validate all pairs.
3. Compute missing artifact masks in memory when each training pair loads.
4. Train U-Net and GAN separately for 90° and 120° (four core runs).
5. Select each best checkpoint using validation L1 and export FBP/model
   comparisons, global metrics and directional-error curves on validation data.

`bash run_full_pipeline.sh` on Linux and `run_full_pipeline.bat` on Windows
now call this same existing-data CUDA runner. They no longer start simulation.

If your ZIPs are in Downloads, pass their actual paths:

```bash
python run_local_cuda.py --part1 "/path/to/Limited_Angle_CT_1000_Training_Part1.zip" \
  --part2 "/path/to/Limited_Angle_CT_1000_Training_Part2.zip"
```

For Windows, put the command on one line and use quoted Windows paths.

If you already extracted **both** parts into one complete folder:

```bash
python run_local_cuda.py --data-dir "/path/to/limited_angle_ct_1000"
```

When a complete extracted folder exists, the runner reuses it and does not
require the ZIP files. It refuses an incomplete folder instead of treating
one half as a complete dataset. Remove an incomplete extraction or choose a
fresh `--data-dir` before importing again.

## Local configuration

Edit `configs/local_cuda_1000.json`. Relative paths resolve from the project
folder, not your terminal's current directory.

| Setting | Default | Meaning |
|---|---|---|
| `device` | `cuda:0` | First visible local NVIDIA GPU |
| `amp` | `true` | CUDA mixed precision with gradient scaling |
| `batch_size` | `4` | Lower to 2 or 1 for a smaller GPU |
| `epochs` | `180` | Training budget for each model |
| `base` | `64` | Generator channels start at 64 |
| `num_workers` | `0` | Portable Linux/Windows data loading |
| `angles` | `[90,120]` | Independent acquisition-specific models |
| `models` | `["unet","gan"]` | Four core model/angle runs |
| `gan_adversarial_weights` | `[1.0]` | Default GAN uses 100*L1 + 1*adversarial |
| `resume` | `true` | Resume matching `last.pt` checkpoints |

Select another visible GPU with `--device cuda:1` or change the JSON. A run uses
one selected GPU; this is not a distributed/multi-GPU training implementation.
If `CUDA_VISIBLE_DEVICES` is set, the index refers to the visible device order.
Batch size 4 with AMP is a starting configuration, not a memory-fit guarantee.

Useful overrides:

```bash
python run_local_cuda.py --device cuda:0 --batch-size 2 --epochs 180
python run_local_cuda.py --no-amp
```

A resume must keep architecture, loss, precision, and batch size compatible with
the original checkpoint. To change these settings after starting a run, choose
a new `--run-dir`. Increasing the epoch budget is supported. A full-checkpoint
resume restores G/D optimizers, random states and the AMP scaler. Only load
checkpoints you trust.

For the full adversarial-weight study, use the supplied eight-run configuration:

```bash
python run_local_cuda.py --config configs/local_cuda_1000_full_study.json
```

It adds GAN weights 0.1 and 10 at both angles. Training/TV timings from the
original project brief are not measured runtime guarantees for your machine.

## Check preparation without launching a long run

```bash
# Print the configured commands, without requiring a GPU or extracting data:
python run_local_cuda.py --dry-run

# Combine and validate the existing archives without importing PyTorch:
python run_local_cuda.py --prepare-only

# Check local CUDA and configured data paths without training:
python run_local_cuda.py --check-only
```

A small training check should use separate output folders so its batch/architecture
settings do not conflict with the full run:

```bash
python run_local_cuda.py --epochs 1 --batch-size 1 --base 16 \
  --run-dir runs/local_smoke --results-dir results/local_smoke
```

The last command still uses all training/validation pairs for one epoch; it
requires your CUDA GPU. It does not establish research performance.

## Outputs and final test

```text
runs/local_cuda_1000/
  launch_*.json
  90_unet/{best.pt,last.pt,settings.json,history.csv,tensorboard/}
  90_gan_adv1/...
  120_unet/...
  120_gan_adv1/...
results/local_cuda_1000/
  90_val_TIMESTAMP/...
  120_val_TIMESTAMP/...
```

After fixing settings using validation results, run the held-out evaluation once:

```bash
python run_local_cuda.py --final-test
```

This loads existing best checkpoints without retraining. Each condition has a
final-test lock. A failed test run leaves a running lock for diagnosis; do not
remove a completed lock to tune on test data. Standard training runs never
inspect test references for checkpoint selection.

## Dataset provenance and available baselines

The combined dataset contains 1,000 ideal phantom images and 2,000 angle-specific
pairs. Each angle has **700 training / 150 validation / 150 test cases**. The
250 phantom families do not cross splits.

The already-generated images use **ASTRA CPU `line_fanflat` projections and
reference CPU fan-beam FBP**. Moving neural training to CUDA does not change how
those inputs/targets were generated. The importer preserves the manifest and
its CPU backend metadata; checkpoints and evaluation reports retain provenance.

The two training ZIPs have no SART, TV-SART, raw signed FBP, or sinogram files.
Evaluation now respects `available_reconstructions=["fbp"]` and compares FBP
with learned outputs, without inventing missing classical baselines. To include
extra methods later, generate their files and request them explicitly with
`limited_ct.cli evaluate --baselines fbp sart tv_sart`. Missing requested files
produce a clear error. Original full datasets without that availability field
retain the original full-baseline behavior.

The optional measured-sinogram archive is not needed for this neural workflow.
Reprojection evaluation and further classical reconstructions require separately
prepared measurements and the relevant projector backend.

## Original simulation workflow

CUDA simulation remains available explicitly:

```bash
python -m pip install -r requirements.txt
python -m limited_ct.cli generate --config configs/project18.json --output data/new_cuda_simulation
```

`configs/project18.json` still describes the original 2,500-image-per-condition
simulation design. It is a simulation configuration, while
`configs/local_cuda_1000.json` controls training on the existing 1,000-image set.
See `docs/ORIGINAL_PIPELINE_README.md` for the original research workflow. The
3D gVXR code stays under `legacy/` and does not drive this local training runner.

## Verification

```bash
python -m unittest discover -s tests -v
# Optional real CUDA AMP training test (Linux):
RUN_LOCAL_CUDA_TESTS=1 python -m unittest discover -s tests -p test_cuda_training.py -v
```

For Windows, set `RUN_LOCAL_CUDA_TESTS=1` in the environment before the test command.
GPU execution cannot be verified without a real local CUDA device. The optional
ASTRA physics test is separate and is not required to train existing image pairs.

Primary CUDA/AMP guidance:
https://pytorch.org/get-started/locally/
https://docs.pytorch.org/docs/stable/notes/amp_examples.html
