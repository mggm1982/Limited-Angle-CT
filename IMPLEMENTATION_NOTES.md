# Implementation notes and requirements traceability

## Why the original code needed a new active pipeline

The uploaded repository projects 3D STL models using gVXR, reconstructs cone-beam FDK volumes, and learns from sparse views. A 3D cone-beam projector, FDK target, or uniformly spaced reduced view count cannot be relabeled as limited-angle 2D fan-beam CT. The original files have therefore been moved intact under `legacy/`, and the root entry points now run Project 18. New code is in `limited_ct/`.

| Requirement | Active implementation |
|---|---|
| Directional multi-material phantoms | `phantoms.py`: 18 oriented inclusions per family, circular material background, varied attenuation values |
| Full 720/360° fan-beam scan | `astra_ops.py`: physical mm geometry and ASTRA CUDA projector |
| 90° and 120° fixed-density acquisition | `config.py`, `generate.py`: contiguous first 180/240 rows of the full grid |
| Poisson Beer–Lambert model | `poisson_log`: counts ~ Poisson(I0 exp(-p)); log transform with 0.5-count floor |
| Full-range FBP target and ideal raster | Shared per-sample arrays in `shared/`; clean FBP chosen explicitly as target |
| Limited-angle FBP/SART network inputs | Zero-filled full-grid FBP; restricted-ray SART, saved independently |
| TV-regularized iterative comparison | Alternating SART/Chambolle TV after each full sweep; validation-only weight selection |
| Five-level residual U-Net | `models.py`: default 64/128/256/512/512 encoders plus a bottom block |
| Conditional GAN and 70×70 discriminator | Four feature convolutions plus logits head; receptive field verified in tests |
| Pixel/adversarial loss ablation | CLI weights; shell runner uses adv=0.1,1,10, with L1=100 |
| Separate models for both ranges | `--angle` stored in checkpoint and checked at evaluation |
| Grouped 70/15/15 split | Exact image counts and disjoint phantom families checked in tests |
| Directional metrics | Reference Sobel normals, fixed threshold and bins; aggregate SSE/counts |
| Image metrics and comparison panels | Against full-FBP and ideal; shared panel ranges and explicit error scale |
| Test once; select on validation | Training reads train/val only; final test uses a per-condition lock |
| Trained weights, research results | Produced when the user runs GPU generation/training; not supplied as fabricated output |
| Optional hybrid sinogram model | Padded sinograms/masks supplied; learned extrapolation architecture is future work |

## Resolved ambiguities and scientific choices

1. **Angular density versus count.** “720 views/360° worth of density” means two views/degree, not 720 views in each restricted range. Intervals exclude the endpoint, avoiding duplicate boundary samples. Source rotation 0° corresponds to ASTRA's source on the negative y axis; edge-normal angles use physical +x/+y.
2. **Fan angle and Parker weights.** With the stated defaults and added 1 mm detector pitch, the short-scan completeness threshold is approximately 208.7°. Neither 90° nor 120° satisfies it. `ShortScan=True` is never used. The limited FBP baseline is an unweighted, incomplete angular integral computed with full-grid quadrature on a zero-filled 360° sinogram, not a mathematically complete short-scan reconstruction. Zeros are used for that linear integral, never as measured rays for SART or reprojection evaluation. A direct restricted-grid FBP call can change ASTRA's normalization with projection count; the fixed-grid baseline avoids that ambiguity. Always report this convention when comparing papers.
3. **SART iteration convention.** The implementation interprets “50 iterations” as 50 full sweeps over measured data: 9,000 single-view ASTRA updates at 90°, and 12,000 at 120°. This choice is explicit and configurable. Calling ASTRA for only 50 updates would not visit all rays even once.
4. **PatchGAN specification.** Kernel-4 convolutions with feature strides `2,2,2,1` followed by a stride-1 output convolution produce a 70×70 receptive field. Four stride-2 feature convolutions plus an output head would produce a 142×142 field. The code prioritizes the requested 70×70 pix2pix discriminator and records this correction.
5. **250 phantom families and exact 375-image splits.** Families are never split. There are 175/38/37 train/validation/test families; sample allocations are 1,750/375/375. Material variants per family vary slightly to meet exact image counts. Thus sample fractions are exactly 70/15/15; family fractions are approximately 70/15/15. The default is 5,000 total pairs, not 5,000 training-only pairs.
6. **Normalization.** One fixed physical scale of 0.02 mm⁻¹ is used. Normalized network arrays are clipped to [0,1], raw arrays retain signed attenuation artifacts. No image-specific normalization and no test-derived parameters are used. Sinograms remain dimensionless line integrals in physical projection units, including their zero-padding representation; normalize separately if implementing a future sinogram network.
7. **Noise-free target.** Full-range noise-free FBP is used so the supervised label does not contain the same stochastic noise as the input. Full clean and noisy sinograms and ideal rasters are stored, permitting a future noisy-target experiment. Do not compare scores to a noisy-target study without stating this difference.
8. **Geometrically meaningful augmentation.** Rotations are disabled. Arbitrary image flips can also change the relation between structures and the fixed acquisition wedge, so they are disabled as well. Directional diversity is simulated by rotating phantom structures before projection. Optional paired intensity jitter is off by default and may be used as a regularization ablation.
9. **Directional evaluation.** Bins use gradients of each reference, not the prediction; a network cannot change the bin membership by erasing an edge. The metric is pixel intensity RMSE evaluated at reference edge locations grouped by edge-normal orientation, not RMSE of gradient vectors. Sobel gradients below 5% of each reference's maximum are excluded. Empty bins are missing, not zero. Error is pooled by pixel counts; per-image PSNR/SSIM summaries are arithmetic means with population standard deviation. Exact-match PSNR is capped at 120 dB for finite CSV output.
10. **Fan-beam wedge interpretation.** Relative angle is the physical gradient-normal angle minus half the source scan range, modulo 180°. The central-ray approximation is useful for a directional plot, but fan rays span a detector-dependent range of normals and visibility is position-dependent. The code makes no exact visible/invisible classification. A future geometry-aware wavefront analysis could improve this interpretation.
11. **TV baseline identity.** TV-SART alternates one sequential, nonnegative SART sweep with Chambolle TV denoising in normalized attenuation units. It is a clearly specified heuristic regularizer, not ASD-POCS or an exact optimizer of data-error-plus-TV. Weight is chosen using validation L1. Extra hyperparameters such as sweep count and inner TV iterations remain fixed across angles by default.
12. **Data consistency.** The image network uses no differentiable projection loss. Reprojection residuals measure disagreement on measured rays, including expected noise; they are diagnostics rather than a proof of consistency. Improvements in image metrics cannot establish factual correctness of missing structures.

## Known operational limitations

- CUDA generation is required; the code does not provide a CPU fan-beam FBP fallback. No CPU parallel-beam approximation is substituted silently.
- Generation is synchronous, writes substantial uncompressed data, and has no partial resume. It fails clearly if ASTRA CUDA is unavailable. Network training supports resume.
- Wall time and 16 GB memory fit have not been benchmarked. The full experiment includes eight training runs, data generation, and multiple TV-validation passes, so the brief's 10–16-hour estimate cannot be assumed for the entire study.
- The active workflow is CLI/notebook-based. The legacy GUI and legacy paper do not drive or document Project 18.
- Synthetic monoenergetic 2D results do not cover real detector calibration, scatter, beam hardening, motion, anatomical diversity, or clinical use.
- For robust research claims, expand the phantom-family diversity and train with several random seeds. Ten material variants of one family do not equal ten independent geometries.

## Primary sources checked during adaptation

- ASTRA FBP_CUDA: supported fanflat geometry, `ShortScan` completeness requirement, filter options. https://astra-toolbox.com/docs/algs/FBP_CUDA.html
- ASTRA SART_CUDA: one single projection per iteration, sequential projection ordering and constraints. https://astra-toolbox.com/docs/algs/SART_CUDA.html
- U-Net: https://arxiv.org/abs/1505.04597
- pix2pix: https://arxiv.org/abs/1611.07004
- Bubba et al., *Learning the invisible: a hybrid deep learning-shearlet framework for limited angle computed tomography*: https://doi.org/10.1088/1361-6420/ab10ca
