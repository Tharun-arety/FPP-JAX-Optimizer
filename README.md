# FPP-JAX-Optimizer

FPP-JAX-Optimizer is a differentiable optimization toolkit for Fiber Patch Placement (FPP) on Type IV COPV domes. The current package couples dome geometry, smooth patch masks, kinematic screening, laminate thickness buildup, analytical membrane loads, classical laminate theory, and a Tsai-Wu failure objective in one JAX optimization loop.

![Optimization evolution](assets/optimization_evolution.gif)

The animation above is the current default demonstration output. It shows the patch footprints evolving on the dome, the same motion on the unwrapped dome, and the objective convergence through the run.

## What It Optimizes

The optimizer does not just paint patches onto a shell. It jointly updates:

- patch center location in `theta` and `phi`
- patch footprint length and width
- patch fiber angle
- patch ply count
- the helical-to-FPP transition boundary
- manufacturability penalties from shear, areal distortion, and thickness gradients
- laminate response under internal pressure through Tsai-Wu failure index

## Current Mechanics Workflow

The current structural path is no longer the old heuristic stress proxy. The workflow is:

1. Build the oblate ellipsoidal dome grid in `(\theta, \phi)`.
2. Seed a differentiable set of FPP patches across the dome.
3. Convert raw optimizer parameters into smooth patch masks and laminate thickness fields.
4. Evaluate kinematic feasibility with Jacobian-based shear and areal distortion penalties.
5. Compute analytical membrane stress resultants for the pressurized dome.
6. Assemble the laminate `A`-matrix from the helical baseline plies and the optimizer-controlled patches.
7. Solve for midplane strains and evaluate a Tsai-Wu failure field.
8. Combine structural, kinematic, thickness, and mass terms into one loss and optimize it with Adam.
9. Record layout snapshots through the run and export them as a convergence GIF and static figures.

## Default Demonstration Run

These numbers come from the current default 90-step asset-generation run.

| Metric | Helical-only baseline | Optimized hybrid demo |
| --- | ---: | ---: |
| Peak Tsai-Wu index | 2,826,724 | 11,641 |
| Total mass (kg) | 0.1095 | 0.2345 |
| Patch mass added (kg) | 0.0000 | 0.0519 |
| Cost saving vs all-FPP laminate | - | 31.6% |
| Transition height (mm) | - | 171.9 |
| Max shear strain | 0.2099 | 0.5456 |
| Max areal distortion | 0.2661 | 0.4889 |
| Max thickness gradient (mm/m) | 9.77 | 13.17 |

The current default run is useful as a visualization and optimization-trace demonstration. It sharply reduces the structural hotspot, but it is not yet a fully balanced final laminate: manufacturability penalties rise again late in the run, so the present repo should be treated as a research prototype rather than a finished design pipeline.

## Visual Outputs

The asset pipeline currently writes:

- `assets/optimization_evolution.gif`
  Full layout-evolution animation with dome view, unwrapped patch map, and convergence panel.
- `assets/optimized_dome_overview.png`
  Seed-vs-optimized patch placement on the dome plus the unwrapped movement map.
- `assets/field_comparison.png`
  Baseline-vs-optimized Tsai-Wu, wrinkle-risk, and thickness-gradient fields.
- `assets/tsai_wu_comparison.png`
  Baseline-vs-optimized Tsai-Wu comparison only.
- `assets/optimization_convergence.png`
  Static convergence plots for loss and normalized metrics.

![Patch placement overview](assets/optimized_dome_overview.png)

## Minimal API

```python
from fpp_jax_optimizer import optimize_patch_layout, summarize_result

result = optimize_patch_layout()
summary = summarize_result(result)

print(summary["optimized_peak_stress_index"])
print(summary["optimized_mass_kg"])
print(len(result["frames"]))
```

The returned `result` bundle includes:

- `baseline`
  Helical-only baseline response.
- `optimized`
  Best layout found in the run.
- `history`
  Scalar convergence trace.
- `frames`
  Serialized layout snapshots used to render the GIF.
- `layout_serialized`
  Final patch centers, angles, sizes, plies, and transition boundary.

## Export Workflow

Typical usage after optimization is:

```python
from pathlib import Path

from fpp_jax_optimizer import (
    optimize_patch_layout,
    write_nastran_bdf,
    write_summary_json,
)

result = optimize_patch_layout()

write_summary_json(result, Path("outputs") / "optimization_summary.json")
write_nastran_bdf(result, Path("outputs") / "fpp_type_iv_dome.bdf")
```

The package currently exports:

- JSON optimization summaries
- Nastran-compatible `PCOMP` / `CQUAD4` shell decks
- README-ready visualization assets through `tools/generate_readme_assets.py`

## Repository Structure

```text
FPP-JAX-Optimizer/
  assets/
    field_comparison.png
    optimization_convergence.png
    optimization_evolution.gif
    optimized_dome_overview.png
    tsai_wu_comparison.png
  examples/
    type_iv_dome_workflow.ipynb
  src/
    fpp_jax_optimizer/
      config.py
      core/
        geometry.py
        kinematics.py
      io/
        export.py
      model/
        fem.py
        loss.py
      topology/
        mapping.py
  tests/
    conftest.py
    test_gradients.py
    test_kinematics.py
    test_mechanics.py
    test_optimization_frames.py
  tools/
    generate_readme_assets.py
  pyproject.toml
  requirements.txt
  setup.py
  README.md
```

## Quick Start

Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest
```

Regenerate the current README assets:

```bash
python tools/generate_readme_assets.py
```

Open the notebook workflow:

```bash
jupyter lab examples/type_iv_dome_workflow.ipynb
```

## Current Dependencies

The package currently expects:

- `jax`
- `jaxlib`
- `optax`
- `numpy`
- `scipy`
- `matplotlib`
- `plotly`
- `Pillow`

## Scope

This repository is a technical prototype for coupled FPP layout exploration. It is useful for studying how patch position, orientation, thickness buildup, and laminate mechanics interact on a dome, but it is not a certification-grade shell solver or manufacturing release tool. The code is structured so that higher-fidelity mechanics, better constraints, and improved export paths can be added without rewriting the whole package.
