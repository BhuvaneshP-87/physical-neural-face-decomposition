# Physical-Neural Decomposition of Facial Appearance

## Abstract

This project investigates a hybrid inverse-rendering pipeline for facial appearance decomposition. Given a single face image, the system estimates interpretable physical components including geometry, albedo, and illumination, then augments the physical renderer with a lightweight neural residual model to explain image evidence not captured by a Lambertian model. The project is framed as a research prototype for digital-human inverse rendering: it emphasizes differentiable optimization, physical parameter interpretability, relighting behavior, ablation analysis, and failure-case diagnosis.

## Research Question

Can a physical differentiable renderer recover interpretable facial appearance parameters while a neural residual branch improves reconstruction quality without fully replacing the physical model?

The project studies the tradeoff between:

- **Interpretability:** geometry, albedo, lighting, masks, and spherical harmonics coefficients remain inspectable.
- **Reconstruction quality:** residual learning captures missing non-Lambertian details, image noise, and local texture effects.
- **Editability:** recovered lighting and geometry should support relighting and limited novel-view visualization.

## Method Overview

The pipeline follows four phases:

1. **Preprocessing:** load an image, detect/crop the face, remove background with GrabCut when available, create a soft support mask, and standardize resolution.
2. **Physical inverse rendering:** optimize depth, albedo, and illumination under a differentiable Lambertian renderer.
3. **Neural residual modeling:** condition a compact U-Net on physical renderings and estimated scene components to predict residual appearance.
4. **Analysis and editing:** export relighting, limited-view sweeps, loss curves, component maps, metrics, and ablation summaries.

## Physical Scene Parameterization

### Geometry

The current geometry representation is a dense canonical depth map. This is intentionally simple and differentiable. It is not a full anatomical head mesh. The depth map is used to derive approximate normals through finite-difference filters and to support small view perturbations.

### Albedo

Albedo is represented as an RGB image in canonical face space. During optimization it is directly updated and clamped to valid image range. Smoothness regularization discourages high-frequency lighting leakage into albedo.

### Illumination

Illumination is represented with RGB spherical-harmonics coefficients. This gives a compact differentiable lighting model that supports relighting presets and coefficient analysis. The diagnostic exporter saves both plots and raw JSON coefficients.

### Renderer

The default renderer is a differentiable PyTorch Lambertian renderer. The repository also includes a validated Mitsuba/Dr.Jit inverse-rendering experiment that exports a canonical face-like mesh scene, loads it in Mitsuba, traverses differentiable scene parameters, deliberately perturbs albedo and illumination, and optimizes those variables back toward a target render with Dr.Jit gradients. Geometry keys are traversed and reported, but the stable notebook experiment optimizes albedo and illumination first because unconstrained single-view vertex optimization is underdetermined without a morphable-face prior.

This creates two complementary tracks:

- **PyTorch track:** fast experimentation, diagnostics, residual modeling, and relighting.
- **Mitsuba/Dr.Jit track:** direct differentiable renderer integration, scene traversal, renderer-side gradients, constrained albedo/illumination optimization, and convergence diagnostics.

## Neural Residual Model

The residual branch predicts an RGB correction over the physical reconstruction. It receives the physical render plus estimated albedo, normals, and depth as conditioning inputs. The model is deliberately compact so that it behaves as a residual appearance term rather than a full black-box renderer.

The residual branch is useful for:

- fine-scale texture and image detail,
- non-Lambertian effects,
- preprocessing artifacts,
- limitations in the simple depth representation,
- reconstruction errors caused by missing anatomy or occlusion.

## Losses

The optimization objective combines:

- **L1 reconstruction loss:** aligns the predicted image to the target.
- **Perceptual loss:** encourages image-space structural similarity.
- **Lighting regularization:** discourages unstable illumination coefficients.
- **Albedo smoothness:** discourages albedo from absorbing shading/noise.
- **Depth smoothness:** discourages spiky geometry.
- **Residual penalty:** keeps neural corrections small enough that the physical model remains meaningful.

## Diagnostics and Analysis Outputs

Each phase can export a diagnostics folder containing:

- `optimization_history.csv`: per-iteration scalar losses.
- `loss_curves.svg`: total, reconstruction, perceptual, smoothness, and residual loss curves.
- `lighting_coefficients.json`: raw RGB spherical-harmonics coefficients.
- `lighting_coefficients.svg`: lighting coefficient plot.
- `component_statistics.json`: tensor statistics for depth, albedo, lighting, prediction, residual, and mask.
- `decomposition_grid.png`: target, physical reconstruction, final prediction, albedo, shading, depth, normals, and error heatmap.
- `residual_map.png`: neural residual magnitude, when Phase 2 is enabled.
- `optimization_snapshots.gif`: reconstruction progress over optimization.
- `diagnostic_summary.md`: human-readable phase summary.

The root output directory also contains `ablation_summary.md`, comparing physical-only and physical+neural phases.

## Experimental Protocol

### Baseline

Run Phase 1 with the residual model disabled. This establishes how far the physical renderer can go using only depth, albedo, and lighting.

### Mitsuba/Dr.Jit Gradient Experiment

Run the optional Mitsuba experiment to demonstrate direct differentiable rendering through Mitsuba:

```bash
pip install -e ".[render]"
python -m src.pipeline --output-dir outputs/mitsuba_inverse_run --mitsuba-inverse-demo
```

The experiment records:

- available Mitsuba scene parameter keys,
- optimized parameter keys,
- loss trace over Dr.Jit optimization,
- target, perturbed-initial, and optimized Mitsuba renders,
- initial-vs-final held-seed MSE diagnostics,
- exported scene XML and mesh assets,
- residual heatmaps when visualized in the unified notebook.

If the optional backend is not installed, the experiment writes a skipped summary rather than failing. This keeps the repository reproducible in lightweight environments while preserving a concrete Mitsuba integration path.

### Physical + Neural

Run Phase 2 initialized from Phase 1. This evaluates whether a residual model improves reconstruction while preserving physically interpretable components.

### Relighting

Use recovered depth/albedo with edited lighting presets:

- frontal lighting,
- side lighting,
- warm sunset lighting,
- colored illumination.

Relighting is a qualitative check on whether albedo and lighting are meaningfully separated.

### Limited Novel View

Run small yaw/pitch sweeps from the recovered depth field. This is a diagnostic visualization only. It should not be interpreted as full 3D head reconstruction.

## Quantitative Metrics

The project reports:

- **PSNR:** pixel reconstruction fidelity.
- **SSIM:** structural similarity.
- **LPIPS proxy / optional LPIPS:** perceptual distance when available.

Metrics alone are insufficient for inverse rendering because a high-quality reconstruction can still have poor decomposition. Therefore the project also reports parameter visualizations, loss curves, lighting coefficients, residual maps, and qualitative relighting.

## Expected Ablations

| Experiment | Purpose |
| --- | --- |
| Physical-only | Measures interpretability and baseline reconstruction. |
| Physical + neural residual | Measures reconstruction improvement from learned appearance. |
| No albedo smoothness | Tests whether albedo absorbs shading/noise. |
| No depth smoothness | Tests geometry stability and spike artifacts. |
| Different lighting presets | Tests editability and relighting stability. |

## Failure Modes

The current implementation has known limitations:

- Single-view depth is ambiguous.
- The default geometry is not a full anatomical face model.
- Hair, ears, neck, and shoulders are not recovered from the image.
- The Lambertian model cannot represent specular skin, subsurface scattering, glasses, makeup, or cast shadows.
- Background removal can fail for complex scenes.
- The neural residual can hide physical-model errors if its weight is too high.
- The Mitsuba experiment is now a stable renderer-gradient proof, but the full real-photo pipeline still uses the PyTorch renderer as its primary optimization backend.

## Relation to Digital Human Inverse Rendering

This project aligns with digital-human inverse rendering through its focus on physical parameter recovery, relighting, and differentiable optimization. To reach a stronger research level, the most important next steps are:

1. Expand the stable Mitsuba/Dr.Jit proof into the primary real-photo optimization backend.
2. Fit a real face prior such as FLAME, DECA, or EMOCA when assets are available.
3. Evaluate on multiple real face images with consistent metrics and qualitative panels.
4. Add stronger priors for identity, symmetry, skin reflectance, and illumination.
5. Compare against established baselines and report ablations.

## Conclusion

The project is best understood as a research scaffold for physical-neural facial inverse rendering, not as a production image-to-3D generator. Its strongest contribution is the interpretable decomposition workflow, analysis tooling, and now a stable Mitsuba/Dr.Jit gradient experiment with measurable convergence. The neural residual branch improves image reconstruction while preserving a physical renderer at the center of the system. The next research milestone is to expand the Mitsuba optimization loop to real photos and add a real morphable face geometry prior.
