# Physical-Neural Face Decomposition

This repository is a research-style scaffold for inverse rendering a single face image into:

- Geometry
- Albedo
- Illumination
- Neural residual appearance

The project is organized around four phases:

1. `Phase 1`: physical inverse rendering with a differentiable Lambertian renderer
2. `Phase 2`: a neural residual model that explains appearance the physical model misses
3. `Phase 3`: relighting under edited illumination presets
4. `Phase 4`: limited novel-view synthesis from recovered geometry

The code is intentionally modular so you can swap in a higher-fidelity renderer such as Mitsuba 3 when it is available, while still having a pure PyTorch fallback that runs for demos and experiments.

The main research focus is not generic image-to-3D generation. It is **inverse rendering for digital-human appearance**: recovering interpretable shape, reflectance, and lighting parameters, then measuring how much a neural residual improves reconstruction while preserving physical editability.

It also includes:

- automatic face-image preprocessing with OpenCV face detection and a center-crop fallback
- optional FLAME and DECA prior hooks for stronger geometry initialization
- saved experiment configs, metrics logs, and checkpoints for training runs
- a concrete Mitsuba scene exporter that writes textured meshes, lighting environment maps, and XML bundles
- research diagnostics including loss curves, lighting coefficient plots, decomposition grids, residual maps, tensor statistics, and ablation summaries

## Repository Layout

```text
project/
├── data/
├── notebooks/
├── outputs/
├── reports/
└── src/
    ├── evaluation/
    ├── models/
    ├── optimization/
    ├── renderer/
    └── training/
```

## Quick Start

1. Create an environment with Python 3.11+.
2. Install the package in editable mode:

```bash
pip install -e .
```

3. Run the synthetic demo or open the unified research notebook:

```bash
python -m src.pipeline --synthetic --output-dir outputs/demo
```

The synthetic demo uses a torch-only face-like generator so the pipeline can be exercised without a face dataset. Real images can be passed in once `torch`, `opencv-python`, and the optional renderer stack are installed.

Example with a real face image and Mitsuba export:

```bash
python -m src.pipeline --input path/to/face.jpg --output-dir outputs/run01 --export-mitsuba
```

Research-style run with automatic diagnostics:

```bash
python -m src.pipeline --input path/to/face.jpg --output-dir outputs/research_run --iterations 300
```

Mitsuba/Dr.Jit differentiable-rendering experiment:

```bash
pip install -e ".[render]"
python -m src.pipeline --output-dir outputs/mitsuba_inverse_run --mitsuba-inverse-demo --mitsuba-iterations 32 --mitsuba-spp 16
```

This experiment performs a deterministic renderer-gradient optimization over Mitsuba scene parameters. It saves target, perturbed-initial, and optimized EXR renders plus MSE convergence diagnostics in `mitsuba_inverse_summary.json`.

If Mitsuba/Dr.Jit are not installed, the command writes `outputs/mitsuba_inverse_run/mitsuba_inverse/mitsuba_inverse_summary.json` with `status: skipped` instead of failing.

Fast smoke test:

```bash
python -m src.pipeline --synthetic --output-dir outputs/smoke --iterations 5
```

You can also point the pipeline at a saved experiment config:

```bash
python -m src.pipeline --config configs/experiment.json --output-dir outputs/run02
```

## Core Modules

- `src/renderer/geometry.py`: depth-to-normal conversion, canonical face grids, and limited view transforms
- `src/renderer/lighting.py`: spherical-harmonics lighting and relighting presets
- `src/renderer/torch_renderer.py`: differentiable PyTorch fallback renderer
- `src/renderer/mitsuba_adapter.py`: optional Mitsuba 3 adapter with lazy imports
- `src/data/preprocessing.py`: real-image loading, face detection, canonical crops, and soft masks
- `src/models/face_priors.py`: optional FLAME/DECA adapter hooks plus synthetic fallback priors
- `src/models/residual_net.py`: lightweight neural residual appearance network
- `src/optimization/inverse_rendering.py`: joint optimization over geometry, albedo, and lighting
- `src/experiments/checkpointing.py`: config saving, metric logging, and checkpoint management
- `src/optimization/losses.py`: photometric, perceptual, and regularization losses
- `src/training/trainer.py`: residual model training loop
- `src/evaluation/metrics.py`: PSNR, SSIM, and LPIPS-style helpers
- `src/evaluation/diagnostics.py`: research plots, component statistics, residual maps, optimization traces, and ablation summaries
- `src/optimization/mitsuba_inverse.py`: optional Mitsuba/Dr.Jit inverse-rendering experiment with parameter traversal and gradient-based optimization
- `src/renderer/mitsuba_scene.py`: mesh export and Mitsuba XML scene bundle generation
- `reports/technical_report.md`: methodology, protocol, diagnostics, limitations, and next research steps
- `notebooks/research_inverse_rendering_demo.ipynb`: unified end-to-end notebook for input preview, decomposition, graphs, relighting, novel views, ablation, and Mitsuba/Dr.Jit evidence

## Suggested Workflow

1. Open `notebooks/research_inverse_rendering_demo.ipynb` for the full visual workflow.
2. Start with `Phase 1` and verify that the physical renderer reconstructs the input face reasonably well.
3. Enable the residual model for `Phase 2` and compare reconstruction quality.
4. Use the recovered parameters to render relighting presets and limited novel views.
5. Run the Mitsuba/Dr.Jit section to inspect differentiable renderer traversal, optimized scene keys, and convergence diagnostics.

## Research Artifacts

Each CLI run writes experiment artifacts under `outputs/<run_name>/`.

For `--phase both`, inspect:

- `phase1/diagnostics/decomposition_grid.png`: target, physical reconstruction, prediction, albedo, shading, depth, normals, and error heatmap
- `phase1/diagnostics/loss_curves.svg`: physical-only optimization curves
- `phase1/diagnostics/lighting_coefficients.json`: recovered RGB spherical-harmonics lighting
- `phase1/diagnostics/component_statistics.json`: summary statistics for depth, albedo, lighting, prediction, and mask
- `phase2/diagnostics/residual_map.png`: magnitude of the learned residual correction
- `phase2/diagnostics/lighting_coefficients.svg`: recovered illumination graph
- `phase2/diagnostics/optimization_snapshots.gif`: reconstruction progress
- `ablation_summary.md`: physical-only versus physical+neural metric comparison
- `mitsuba_inverse/mitsuba_inverse_summary.json`: Mitsuba/Dr.Jit availability, traversed parameters, optimized keys, loss trace, and initial-vs-final MSE reduction

The diagnostics are designed to answer research questions such as:

- Does the physical renderer explain the target image without hiding errors in albedo?
- Does the residual branch improve reconstruction quality?
- Are recovered lighting coefficients stable and interpretable?
- Where does the physical model fail: geometry, shading, texture, mask, or non-Lambertian appearance?
- Do relighting edits preserve plausible facial identity?

## Data Conventions

- `data/`: raw or processed face images
- `outputs/`: rendered reconstructions, relighting frames, GIFs, and checkpoints
- `reports/`: ablation summaries and a final technical report

## Notes

- Mitsuba 3 and Dr.Jit are optional dependencies, but the repository includes a validated differentiable-rendering experiment using Mitsuba scene traversal and Dr.Jit gradients.
- The current fallback renderer is intentionally simple and differentiable; it is meant to be a strong baseline and a development harness.
- The unified notebook and synthetic dataset are designed so the codebase is usable before real face data is wired in.
- Training runs persist `config.json`, `manifest.json`, `logs/metrics.jsonl`, and epoch checkpoints under the configured run directory.
- The procedural clay bust export is a visualization utility, not the core research contribution. The core project is physical-neural inverse rendering and analysis.
