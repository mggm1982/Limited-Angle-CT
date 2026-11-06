# Limited-Angle CT Reconstruction Using Deep Learning

**Status: report template. Experimental result cells are intentionally unfilled.**

## Abstract

Describe the limited-angle problem, paired synthetic fan-beam dataset, residual U-Net and conditional GAN, and classical comparison. Insert measured findings only after completing the experiments. Mention directional residuals and hallucination risk explicitly.

## Background and literature review (5 marks)

Explain incomplete angular data, orientation-dependent boundary visibility, FBP limitations, TV regularization, U-Net regression, and adversarial priors. Cite primary ASTRA, U-Net, pix2pix, and limited-angle reconstruction references. Distinguish limited-angle data from sparse views sampled around a full trajectory.

## Data preparation (6 marks)

Report actual geometry, physical units, detector pitch, pixel size, material values, angular grid, photon count, noise-floor convention, and target construction. Include phantom previews, sample sinograms, measured masks, and full-FBP scale sanity checks. State 250 phantom families, actual variant allocation, exact sample counts and grouped split. Explain shared noise and objects across the two angle conditions.

## Methodology and implementation (10 marks)

Describe all baselines and the fixed-grid incomplete FBP convention. Specify SART sweeps, TV proximal solver/weight selection, U-Net widths and residual output, actual 70×70 PatchGAN strides, optimization, loss weights, checkpoint criterion, and optional terms actually used. Include training histories and disclose deviations from the planned protocol.

## Results and evaluation (8 marks)

Populate this table from final held-out `summary.csv` files; include both target-reference and ideal-reference metrics in the final report.

| Angle | Method | PSNR vs target | SSIM vs target | RMSE vs target | RMSE vs ideal |
|---|---|---|---|---|---|
| 90° | FBP | Pending | Pending | Pending | Pending |
| 90° | SART | Pending | Pending | Pending | Pending |
| 90° | TV-SART | Pending | Pending | Pending | Pending |
| 90° | U-Net | Pending | Pending | Pending | Pending |
| 90° | GAN, adv 0.1 | Pending | Pending | Pending | Pending |
| 90° | GAN, adv 1 | Pending | Pending | Pending | Pending |
| 90° | GAN, adv 10 | Pending | Pending | Pending | Pending |
| 120° | Same seven methods | Pending | Pending | Pending | Pending |

Show matched-phantom panels at both angles, reference edge-normal polar curves and bin counts, residual error maps, and GAN loss-ratio ablations. Include the unclipped FBP diagnostic and optional projection residual. Report per-family variation; avoid treating material variants as independent geometry samples. Select illustrative failure cases using a declared rule rather than showing only favorable examples.

## Critical discussion (3 marks)

Discuss persistent direction-dependent errors, missing structures, incorrect invented edges, plausibility versus pixel fidelity, and measured-data disagreement. Do not infer uniqueness or clinical safety from image quality. Discuss synthetic-to-real limitations, fan-beam orientation approximations, TV tuning fairness, and computational cost. Numerical expectations from the assignment are hypotheses, not evidence.

## Report and presentation quality (3 marks)

Use consistent units, readable figure scales, labeled angle conventions, a reproducible command appendix, and transparent checkpoint/data provenance. Add actual hardware, runtime, peak GPU memory, software versions, and random seeds. The notebook should load the reported checkpoint and reproduce at least one final comparison.

## Conclusion

State only conclusions supported by your measured experiments. Explain what the method improves and what remains unresolved, especially in the 90° condition.
