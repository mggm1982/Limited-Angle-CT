# Project description and justification

## Title

**Limited-Angle CT Reconstruction Using Deep Learning**

## Project description

This project develops and evaluates a deep-learning pipeline for reconstructing two-dimensional CT images when projections are available over only 90° or 120° of rotation. Instead of reducing the sampling density across a complete scan, the acquisition omits an entire angular region. This removes orientation-dependent information and produces directional streaking, boundary smearing, and elongated structures. The investigation focuses on reconstructing plausible structure while explicitly measuring which errors remain.

The dataset is entirely synthetic. A library of 250 phantom families contains elongated ellipses, rotated rectangles, and layered rings with several attenuation levels. Approximately ten material variants per family produce 2,500 images at 256×256 resolution. ASTRA simulates 720 projections over a full 360° fan-beam trajectory with a 600 mm source-to-object distance, 400 mm object-to-detector distance, and 512 detector elements. Restricting this grid to 90° and 120° yields 180 and 240 measured views, respectively. Keeping 0.5° spacing isolates the effect of missing angular coverage from changes in sampling density.

Photon counts follow the Beer–Lambert attenuation model and Poisson statistics. The restricted measurements produce FBP and SART inputs. Full-range noise-free FBP provides the supervised target, and the original phantom is retained as an independent ideal reference. The two angular conditions share phantom identities, noise realizations on overlapping rays, and train/validation/test assignments for a controlled comparison.

A residual U-Net first learns artifact correction using pixel supervision. A conditional GAN then adds a PatchGAN discriminator to encourage plausible local structure. The core comparison is between identical generator architectures with and without adversarial training. Additional experiments vary adversarial strength and compare learning against TV-regularized SART. Separate models are trained for 90° and 120°, with checkpoints selected using validation L1 rather than discriminator performance.

Evaluation reports PSNR, SSIM, RMSE, MAE, side-by-side reconstructions, absolute-error maps, and error binned by reference edge-normal orientation. Scores are computed against both the full-range FBP target and the ideal phantom so that errors inherited from the reconstruction target are visible. An optional measured-ray reprojection residual tests whether a visually improved reconstruction still agrees with observed data.

## Why this project is worth doing

**Practical motivation.** Restricted mechanical access can prevent complete scanner rotation in industrial inspection and some C-arm configurations. A controlled limited-angle simulation therefore addresses a meaningful acquisition constraint rather than an arbitrary image-corruption problem. This project studies the constraint in 2D; it does not claim to reproduce every aspect of a real C-arm or 3D scanner.

**A scientifically distinct inverse problem.** Missing angular coverage removes information associated with particular edge orientations. It therefore cannot be studied adequately by randomly dropping projections from a full scan. The proposed acquisition preserves sampling density and deliberately varies object orientation, making the experiment suitable for analyzing directional failure modes.

**A clear test of learned priors.** U-Net regression measures how much improvement comes from paired pixel supervision. Adversarial training tests whether local realism provides a useful additional prior, and the loss-weight ablation reveals its tradeoff with fidelity. TV-SART offers an explicit non-learned smoothness prior. Comparing these approaches distinguishes better-looking images from more faithful reconstruction.

**Reliable supervision and controlled comparisons.** Synthetic data provide exact object boundaries, known material values, matched scans, and reproducible geometry. Family-level splitting prevents nearly identical phantom variants from leaking into the test set. Sharing the underlying objects between angle conditions reduces confounding when comparing 90° with 120°.

**More informative evaluation than a single global score.** Overall PSNR can obscure small but important lost boundaries. Orientation-binned error and ideal-reference comparisons identify the structures that remain poorly recovered and expose hallucinated edges or persistent smearing. Projection residuals add an independent measurement-space check.

**Feasibility and educational value.** Two-dimensional phantoms avoid the memory and data-generation demands of a full volumetric system while retaining the central limited-angle challenge. The work combines forward modeling, inverse methods, deep learning, adversarial optimization, reproducible experimentation, and critical interpretation in a coherent advanced project.

## Aim and research questions

The aim is to determine how effectively residual U-Net and conditional-GAN correction recover limited-angle fan-beam CT structure, and to characterize the directional errors that remain.

1. How do reconstruction quality and residual edge errors differ between 90° and 120°?
2. Does adversarial supervision improve structural appearance at the cost of pixel fidelity or measured-data agreement?
3. Which reference edge orientations remain difficult after correction?
4. How do learned methods compare with a validation-tuned TV-SART baseline?
5. How sensitive are the conclusions to the L1/adversarial loss ratio?

## Planned experiments

| Experiment | Controlled settings | Main evidence |
|---|---|---|
| U-Net vs GAN | Same generator, input, split, seed, epoch budget | Fidelity, SSIM, error maps, projection residual |
| 90° vs 120° | Same phantom families and angular spacing | Matched-family metrics and directional curves |
| Adversarial-weight ablation | L1=100; adversarial=0.1, 1, 10 | Realism/fidelity tradeoff |
| TV-SART comparison | Same noisy restricted rays; validation-only TV selection | Common comparison table and panels |
| Directional analysis | Reference-based edge normals and fixed bins | Orientation-binned RMSE with bin counts |

The default four main runs and four ablation runs are a minimum functional study. Repeating with multiple training seeds and reporting family-level uncertainty would strengthen the research claims.

## Expected contribution and limits

The intended contribution is a reproducible experimental pipeline and an honest analysis of orientation-dependent recovery, not a claim that missing measurements can always be reconstructed uniquely. At 120°, more measured information should generally help; the 90° case should expose stronger dependence on learned priors. These are hypotheses to test. The numerical PSNR/SSIM ranges supplied in the project brief must not be presented as achieved results without completing the experiments.

A GAN can generate convincing but incorrect structures. Pixel loss, a visually convincing panel, or high SSIM alone cannot eliminate that risk. This implementation measures several aspects of quality but does not mathematically enforce projection consistency or certify uniqueness. Domain shift to experimental measurements, scatter, beam hardening, detector effects, and three-dimensional geometry remain outside the present synthetic monoenergetic study.

## Ten-week work plan

| Week | Work and reviewable output |
|---|---|
| 1 | Literature review; define geometry, hypotheses, and metrics |
| 2 | Directional phantom design; inspect material and orientation coverage |
| 3 | Validate projector scale and full-FBP quality; generate pilot data |
| 4 | Produce grouped dataset; validate FBP/SART/TV baselines |
| 5 | Train and validate U-Net at both angles |
| 6 | Train and stabilize conditional GAN at both angles |
| 7 | Run adversarial-weight ablations and TV validation tuning |
| 8 | Freeze settings; evaluate test sets and direction-dependent errors |
| 9 | Inspect residual artifacts and hallucination cases; prepare figures |
| 10 | Write report, finalize demo notebook, archive code and checkpoints |

## Deliverables and current status

The code package supplies simulation, training, inference, evaluation, experiment scripts, example ideal phantoms, and a demo notebook. It also supplies this description, implementation notes, and a report template. Trained weights, full paired datasets, scientific result figures, and the completed results section require running ASTRA CUDA and the training experiments; they are not fabricated in the package.

## References

- ASTRA Toolbox: fan-beam projection geometry, FBP_CUDA, and SART_CUDA documentation. https://astra-toolbox.com/
- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. https://arxiv.org/abs/1505.04597
- Isola, Zhu, Zhou, Efros. *Image-to-Image Translation with Conditional Adversarial Networks*. https://arxiv.org/abs/1611.07004

- Bubba et al. *Learning the invisible: a hybrid deep learning-shearlet framework for limited angle computed tomography*. https://doi.org/10.1088/1361-6420/ab10ca
