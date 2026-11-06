# Validation report

Verified on 23 October 2026 in the delivery environment.

- Python 3 with PyTorch 2.14.1, ASTRA 2.5.0, NumPy, SciPy, scikit-image, and matplotlib installed.
- PyTorch CUDA availability: false. ASTRA CUDA availability: false.
- `python -m unittest discover -s tests -v`: 8 test cases discovered; 7 passed, 1 CUDA physics case skipped.
- Verified exact 1,750/375/375 per-condition split and disjoint phantom families.
- Verified fixed-density 180/240-view subsets and zero-padding/mask semantics.
- Verified finite/reproducible Poisson log data, multi-material rasters, and reference-based directional bins.
- Verified generator shape, residual initialization, gradient propagation, and the 70×70 PatchGAN receptive field.
- Verified CPU GAN training, resumed training, U-Net training, checkpoint load, inference, metric tables, image/polar figures, and final-test lock behavior with an explicitly artificial array fixture. Fixture scores are not CT research results.
- Active Python modules compile. Both shell scripts pass `bash -n`.
- Every demo-notebook code cell parses; its first three CPU cells execute using included phantoms.
- Ideal phantom preview rendered and visually inspected.
- Generation was attempted in the no-CUDA environment and stopped with the expected explicit CUDA requirement. It did not silently generate a different projection geometry.

Not verified: real ASTRA CUDA forward projection/FBP scale, GPU SART/TV reconstruction, full paired dataset generation, GPU memory use, runtime, 180-epoch training, and held-out scientific performance. Run the opt-in GPU disk test before generating the research dataset. No research weights or reconstructed test results are bundled.
