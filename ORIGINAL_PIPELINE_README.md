# Project 18: Limited-Angle CT Reconstruction Using Deep Learning

This is the Project 18 adaptation of the supplied gVXR/dual-domain reconstruction repository. The active code uses **2D synthetic phantoms, ASTRA fan-beam geometry, fixed-density 90°/120° acquisition, residual U-Net, and conditional GAN correction**. The original 3D cone-beam/sparse-view code and documentation are preserved under `legacy/` for reference; its commands, GUI, dependencies, and claimed results do not describe this new workflow.

Read [docs/PROJECT_DESCRIPTION_AND_JUSTIFICATION.md](docs/PROJECT_DESCRIPTION_AND_JUSTIFICATION.md) for the proposal and scientific justification, and [docs/IMPLEMENTATION_NOTES.md](docs/IMPLEMENTATION_NOTES.md) for requirements mapping and limitations.

## What is implemented

- Analytic, reproducible multi-material phantom families with rotated elongated ellipses, rectangles, and layered rings.
- 720-view, 360° ASTRA `fanflat` projection: SOD 600 mm, ODD 400 mm, 512 detectors. Added explicit assumptions: 1 mm pixels, 1 mm detector pitch, maximum attenuation 0.02 mm⁻¹, incident count 100,000 per ray.
- Restricted acquisitions `[0°,90°)` and `[0°,120°)` with **180 and 240 measured views**, preserving 0.5° spacing. These are limited-angle acquisitions, not uniformly sparse views over 360°.
- Beer–Lambert Poisson counting noise; shared measured noisy rays across the two angle conditions.
- Noise-free full-range FBP target and ideal phantom reference; limited-angle FBP, 50-sweep SART, and alternating SART/TV baselines.
- Residual U-Net with five down/up levels (64–128–256–512–512), bilinear upsampling, skip connections, and global input residual.
- Conditional 70×70 PatchGAN, BCE logits loss, default `100*L1 + 1*adversarial`; optional SSIM and artifact weighting.
- Independent training by angular range; validation-L1 checkpoint selection; resumable G/D and optimizer states; CSV and optional TensorBoard history.
- PSNR/SSIM/RMSE/MAE against both full-FBP target and ideal raster; per-family summaries; edge-normal-binned RMSE and polar plots; shared-scale image/error panels; optional measured-data reprojection residual.
- Validation-only TV weight tuning, GAN ratio ablations, inference, demo notebook, focused tests, and a CUDA physics test.

**Execution status:** Code and CPU neural plumbing have been checked. ASTRA CUDA physics and full experimental training have not been run in the delivery environment. No pretrained weights or measured research performance are included. The included phantom previews are real raster examples, not CT reconstruction results.

## Installation

Python 3.10+ and a CUDA-capable NVIDIA GPU are required for this implementation's fan-beam FBP simulation. CPU can train/infer on an already generated dataset, but generation does not silently substitute parallel-beam Radon geometry.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a PyTorch build appropriate to your CUDA/driver using pytorch.org instructions.
python -m pip install -r requirements.txt
python -m limited_ct.cli --help
```

For Windows, activate `.venv\Scripts\activate`. ASTRA must report a usable CUDA GPU; wheel/driver compatibility depends on the machine. Verify both frameworks:

```bash
python -c "import torch, astra; print('PyTorch CUDA:', torch.cuda.is_available()); print('ASTRA CUDA:', astra.use_cuda())"
```

For CPU-only code checks using the existing raster examples:

```bash
python -m pip install -r requirements-cpu.txt
python -m unittest discover -s tests -v
```

## Run the project

Run commands from this folder. Defaults are centralized in `configs/project18.json`.

### 1. Validate geometry and exact split counts without a GPU

```bash
python -m limited_ct.cli generate --config configs/project18.json --output data/plan --plan-only
```

The split is by phantom family **before simulation**. All material variants and both angular conditions use the same split. There are 175 train, 38 validation, and 37 test families. Balancing the variants within each split yields exactly **1,750/375/375 samples per condition**. Most families have 10 variants; some validation/test families have 9 or 11. This avoids the impossible requirement of making 375 images from an integer number of families with exactly 10 images each.

### 2. Generate 5,000 pairs across both conditions

```bash
python -m limited_ct.cli generate --config configs/project18.json --output data/project18
```

There are 2,500 unique phantom rasters with paired angle conditions: 5,000 total pairs, of which **3,500 are training pairs**, 750 validation, and 750 test. Full-range data are shared on disk across conditions. Allow roughly 12–20 GB of data space plus checkpoints (estimate; depends on config). Generating SART and TV baselines is substantial work beyond network training.

Generation refuses to overwrite a completed dataset. The manifest is written only on successful completion. Partial generation does not support resume: use a fresh output folder or delete the incomplete folder yourself before retrying.

### 3. Tune the TV baseline on validation data

Do this for both conditions before final evaluation:

```bash
python -m limited_ct.cli tune-tv --angle 90 --output runs/tv_90
python -m limited_ct.cli apply-tv --selection runs/tv_90/selection.json
python -m limited_ct.cli tune-tv --angle 120 --output runs/tv_120
python -m limited_ct.cli apply-tv --selection runs/tv_120/selection.json
```

Default search: `0.0005 0.001 0.002 0.005 0.01` in normalized attenuation units. This is an alternating SART/TV proximal baseline, not a claim to solve an exact constrained-TV objective. All baseline settings and the selected weight must be reported.

### 4. Train independent U-Net and GAN models

```bash
python -m limited_ct.cli train --angle 90 --model unet --output runs/90_unet --tensorboard
python -m limited_ct.cli train --angle 90 --model gan --output runs/90_gan --tensorboard
python -m limited_ct.cli train --angle 120 --model unet --output runs/120_unet --tensorboard
python -m limited_ct.cli train --angle 120 --model gan --output runs/120_gan --tensorboard
```

Defaults: 180 epochs, batch 8, Adam `lr=2e-4`, betas `(0.5,0.999)`, base channels 64. Start with a smaller config and `--base 16 --batch-size 2 --epochs 2` to verify your installation. A 16 GB GPU is a target, not a verified memory/runtime guarantee; lower batch size if needed. For SART-input experiments add `--input sart` to every relevant training command and use separate output folders.

```bash
python -m limited_ct.cli train --angle 90 --model gan --output runs/90_gan \
  --resume runs/90_gan/last.pt --epochs 200 --tensorboard
```

Resume checks architecture, angle, loss settings, optimization settings, and dataset manifest. `last.pt` and `best.pt` contain G/D weights, optimizer states, settings, epoch, dataset identity, and RNG states. Resume from `last.pt` for the current history. Never load untrusted `.pt` files: these checkpoints use Python object serialization.

Optional losses: `--lambda-ssim 5`, `--artifact-weight 2`. Default image augmentation is disabled because transforms can alter edge orientation relative to fixed fan geometry. `--jitter 0.03` enables small paired intensity changes; no rotations or flips are applied. Diverse orientations are generated physically before projection.

### 5. Run the adversarial-weight ablation and validate

Keep L1 at 100; compare GAN adversarial weights `0.1`, `1`, and `10`. The plain U-Net is the no-adversarial reference. Repeat for both angles.

```bash
python -m limited_ct.cli train --angle 90 --model gan --lambda-adv 0.1 --output runs/90_gan_adv0.1
python -m limited_ct.cli train --angle 90 --model gan --lambda-adv 10 --output runs/90_gan_adv10
python -m limited_ct.cli evaluate --angle 90 --split val --output results/90_validation \
  --checkpoint unet=runs/90_unet/best.pt \
  --checkpoint gan_adv1=runs/90_gan/best.pt \
  --checkpoint gan_adv0.1=runs/90_gan_adv0.1/best.pt \
  --checkpoint gan_adv10=runs/90_gan_adv10/best.pt
```

`bash run_full_pipeline.sh` runs generation, TV selection, all eight training runs, and validation. It stops before final test so choices can be fixed using validation results. `run_full_pipeline.bat` runs generation and the four core models; follow the commands above for TV tuning and ablations on Windows.

### 6. Perform the final held-out evaluation once per angle

```bash
bash run_final_test.sh
```

This evaluates every predeclared model and all classical baselines on the same held-out phantoms, and checks measured projection residuals with ASTRA. You can use the individual `evaluate` command with `--split test`; omit `--projection-residual` to evaluate existing image arrays without ASTRA. Each test condition creates a lock in the dataset folder to prevent casual reruns and TV changes. A failed run leaves a `running` lock: diagnose the failure and remove only that failed lock before retrying. Do not remove completed locks to tune on test data. Separate test directories must be used for the two conditions.

`summary.csv` is the TV-SART/U-Net/GAN comparison table. `per_sample.csv` and `per_family.csv` preserve detail. `directional.csv` retains SSE/counts so empty orientation bins are not treated as zero error. `directional_target.png`, `directional_ideal.png`, comparison panels, predictions, and evaluation provenance are also saved. Report correlated variants at family level; run additional independently seeded experiments if claiming statistical significance.

### 7. Infer on a matching normalized FBP or SART image

```bash
python -m limited_ct.cli infer --checkpoint runs/90_gan/best.pt \
  --input data/project18/angle_90/PHANTOM_ID/fbp.npy --output corrected.npy
```

Replace `PHANTOM_ID` with an actual ID from the manifest. Inputs must match the checkpoint's acquisition and reconstruction settings. Outputs are normalized attenuation. Multiply by `attenuation_scale_per_mm` to recover mm⁻¹. These are synthetic attenuation units, not calibrated Hounsfield units.

## Data layout

```text
data/project18/
  geometry.json                  # units, geometry, angular convention, normalization
  plan.json                      # family-level allocation before simulation
  manifest.json                  # complete paired sample index
  shared/p0000_v00/
    phantom.npy                  # ideal normalized raster, float32
    sinogram_clean.npy           # 720x512, dimensionless line integrals
    sinogram_noisy.npy            # shared Poisson log data
    target.npy                   # normalized full-range noise-free FBP
    target_raw.npy               # full-range FBP in mm^-1
    shapes.json
  angle_90/p0000_v00/             # angle_120 has the same fields
    sinogram.npy                 # 180x512 (240x512 for 120 degrees)
    sinogram_padded.npy           # 720x512, zeros outside measured rows
    measured_mask.npy            # 720x512, measured=1, missing=0
    fbp.npy, sart.npy, tv_sart.npy
    fbp_raw.npy, sart_raw.npy, tv_sart_raw.npy
    artifact_mask.npy            # abs(normalized FBP - target), training-only label use
    metadata.json
```

Every `.npy` is float32; JSON contains metadata. Image normalization uses one fixed physical attenuation scale with clipping to `[0,1]`; it never rescales each sample independently or learns bounds from held-out data. Raw signed FBP is retained, and `fbp_unclipped` is evaluated alongside the clipped FBP network input to disclose the effect of clipping. Network outputs are saved both clipped and unclipped; published summary metrics use clipped network predictions.

## Scientific interpretation

The study asks whether learned priors improve plausible missing structure and how much orientation-dependent error remains. A GAN may increase visual plausibility while reducing pixel accuracy or inventing boundaries. It does **not** guarantee measured-data consistency. Enable reprojection residuals to measure that limitation; enforcing projection consistency during learning is future work. Zero-filled sinograms and masks are available for a future hybrid model; a trained sinogram-extrapolation branch is not implemented.

Edge bins use the **reference** gradient normal, physical +x to the right and +y upward, modulo 180°, relative to the measured source-angle center. Fan-beam ray normals vary across detector channels, so the central-ray missing-wedge interpretation is approximate, not an exact visible/invisible-boundary classification. Both target- and ideal-reference orientation analyses are exported.

Performance numbers in the supplied brief are hypotheses, not measured outcomes of this code. Lower-angle acquisition is expected to be harder, but no PSNR/SSIM threshold is promised. This synthetic project does not establish clinical suitability.

## Checks and references

```bash
python -m unittest discover -s tests -v
RUN_ASTRA_CUDA_TESTS=1 python -m unittest discover -s tests -v
```

The opt-in GPU test checks analytic disk line-integral scale, full-scan FBP amplitude, and finite SART/TV output. Run it before generating the research dataset. The CPU neural test uses an explicitly artificial array fixture to check software plumbing; its printed metrics are not CT experimental results.

Primary references: [ASTRA FBP_CUDA](https://astra-toolbox.com/docs/algs/FBP_CUDA.html), [ASTRA SART_CUDA](https://astra-toolbox.com/docs/algs/SART_CUDA.html), [ASTRA projector geometry](https://astra-toolbox.com/docs/geom2d.html), [U-Net](https://arxiv.org/abs/1505.04597), [pix2pix](https://arxiv.org/abs/1611.07004).
