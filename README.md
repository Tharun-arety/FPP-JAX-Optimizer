# FPP-JAX-Optimizer

FPP-JAX-Optimizer is a differentiable optimization toolkit for Fiber Patch Placement (FPP) on Type IV COPV domes. It couples patch layout, kinematic feasibility, thickness buildup, and a smooth pressure-response proxy in a single JAX optimization loop, then exports the optimized reinforcement layout to downstream analysis formats.

## Features

- Oblate ellipsoidal dome parameterization in spherical coordinates
- Smooth differentiable patch masks for center, size, orientation, and ply-intensity optimization
- Jacobian-based shear and distortion penalties for manufacturability screening
- Thickness accumulation and gradient penalties for laminate smoothness
- Differentiable structural proxy for internal-pressure-driven optimization
- Export to Nastran-compatible shell decks and JSON summaries

## Repository Structure

```text
FPP-JAX-Optimizer/
  src/
    fpp_jax_optimizer/
      __init__.py
      config.py
      core/
        geometry.py
        kinematics.py
      model/
        fem.py
        loss.py
      topology/
        mapping.py
      io/
        export.py
  tests/
    conftest.py
    test_kinematics.py
    test_gradients.py
  examples/
    type_iv_dome_workflow.ipynb
  outputs/
    .gitkeep
  requirements.txt
  setup.py
  README.md
```

## Optimization Workflow

1. Build the Type IV dome grid in `(\theta, \phi)`.
2. Define smooth FPP patches with center, size, orientation, and ply-intensity parameters.
3. Evaluate coupled structural, kinematic, and thickness objectives.
4. Run gradient-based optimization in JAX to move, rotate, resize, and taper the patches.
5. Export the optimized layout to `.bdf`, JSON, and optional visualization artifacts.

## Demonstration Geometry

The included workflow uses a representative Type IV dome configuration:

- oblate ellipsoidal dome
- `R_major / R_minor = 1.414`
- `50 mm` polar opening
- baseline helical spillover layer plus optimized FPP reinforcement

## Quick Start

Install the package and its dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the tests:

```bash
pytest
```

Open the example notebook:

```bash
jupyter lab examples/type_iv_dome_workflow.ipynb
```

## Generated Artifacts

Typical outputs are written to `outputs/`:

- `fpp_type_iv_dome.bdf`
- `fpp_layout.html`
- `optimization_summary.json`

## Scope

This repository is a technical prototype, not a certification-grade structural solver. The structural path uses a smooth pressure/stress proxy rather than a full shell finite-element implementation, but the package layout keeps the geometry, kinematics, optimization, and export layers modular for higher-fidelity extensions.
