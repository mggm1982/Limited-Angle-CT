# Local CUDA existing-dataset workflow validation

Checked on 31 October 2026.

- The modified project was based on the newly uploaded project ZIP.
- Imported the actual `Limited_Angle_CT_1000_Training_Part1.zip` and Part2 ZIP into
  a fresh staging/output folder using the new `--prepare-only` entry point.
- Verified 1,000 unique image IDs and 2,000 complete paired records. Each angle
  has 700 train / 150 validation / 150 test cases. No family crosses splits.
- Verified every required image file, shape, float32 dtype, finite values and
  normalized range. The original CPU-generation manifest remained unchanged.
- Loaded real 90°/120° training pairs using the new lazy-mask data loader, without
  precreating artifact-mask files.
- CPU test suite: 14 cases, 12 passed, 2 GPU-dependent cases skipped.
- Tests cover extraction/merge, repeat import, traversal/conflicting headers,
  incomplete archives, lazy masks, FBP-only evaluation/provenance, CUDA failure
  messages, configuration plans, generator gradients, CPU GAN/U-Net training,
  checkpoint resume, metrics, inference, and test locks.
- Four-run and eight-run configurations produce the expected CUDA commands and
  never schedule dataset generation.
- Active Python code and notebook cells parse. Root shell scripts pass syntax checks.
- The default CUDA availability check exits with an actionable error in this
  environment instead of switching to CPU.

Not verified here: CUDA tensor execution, GPU AMP optimization or resume, memory
fit on the user's GPU, full 180-epoch training, Windows execution, or measured
held-out reconstruction quality. The code supplies a real CUDA AMP test for the
user's machine. No fixture/test checkpoint is packaged as a trained research model.
