from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..config import DomeConfig, MaterialConfig, OptimizationConfig
from ..core.geometry import DomeState, build_type_iv_dome
from ..core.kinematics import evaluate_kinematics
from ..model.fem import evaluate_structural_response
from ..topology.mapping import decode_layout, evaluate_thickness_state, initial_raw_layout


def _scalarize(aux: dict[str, object], path: str) -> float:
    value = aux
    for part in path.split("."):
        value = value[part]  # type: ignore[index]
    return float(value)


def _serialize_layout(layout: dict[str, jnp.ndarray]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in layout.items():
        if isinstance(value, jnp.ndarray):
            serialized[key] = np.asarray(value).tolist()
        else:
            serialized[key] = value
    return serialized


def evaluate_layout(
    raw_params: jnp.ndarray,
    dome: DomeState,
    material: MaterialConfig | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, object]:
    material = material or MaterialConfig()
    config = config or OptimizationConfig()

    layout = decode_layout(raw_params, dome, config)
    thickness_state = evaluate_thickness_state(dome, layout, material, config)
    kinematics = evaluate_kinematics(dome, layout, material, config)
    structure = evaluate_structural_response(dome, layout, thickness_state, material)

    thickness_gradient = thickness_state["thickness_gradient_mm_per_m"]
    thickness_cliff_excess = jax.nn.relu(thickness_gradient - material.max_thickness_gradient_mm_per_m)
    thickness_penalty = (
        jnp.sum(dome.area_weights * thickness_cliff_excess**2) / jnp.maximum(dome.total_area_m2, 1.0e-8)
        + 8.0 * jnp.max(thickness_cliff_excess) ** 2
    )
    mass_term = thickness_state["total_mass_kg"] / jnp.maximum(
        jnp.sum(dome.area_weights) * material.baseline_helical_plies * material.areal_density_kg_per_m2_per_ply,
        1.0e-8,
    )

    total_loss = (
        config.stress_weight * structure["structural_loss"]
        + config.shear_weight * kinematics["penalty"]
        + config.thickness_weight * thickness_penalty
        + config.mass_weight * mass_term
    )

    metrics = {
        "loss": total_loss,
        "structural_loss": structure["structural_loss"],
        "kinematic_penalty": kinematics["penalty"],
        "thickness_penalty": thickness_penalty,
        "mass_term": mass_term,
        "peak_stress_index": structure["peak_stress_index"],
        "mean_stress_index": structure["mean_stress_index"],
        "max_shear": kinematics["max_shear"],
        "max_areal_distortion": kinematics["max_areal_distortion"],
        "max_thickness_gradient_mm_per_m": jnp.max(thickness_gradient),
        "total_mass_kg": thickness_state["total_mass_kg"],
        "patch_mass_kg": thickness_state["patch_mass_kg"],
        "helical_mass_kg": thickness_state["helical_mass_kg"],
        "cost_savings_vs_all_fpp_pct": thickness_state["cost_savings_vs_all_fpp_pct"],
        "transition_height_m": structure["transition_height_m"],
    }

    return {
        "loss": total_loss,
        "layout": layout,
        "thickness": thickness_state,
        "kinematics": kinematics,
        "structure": structure,
        "metrics": metrics,
    }


def evaluate_baseline_layout(
    dome: DomeState | None = None,
    material: MaterialConfig | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, object]:
    dome = dome or build_type_iv_dome()
    config = config or OptimizationConfig()
    raw_params = initial_raw_layout(dome, config)
    raw_params = raw_params.at[5::6].set(-12.0)
    return evaluate_layout(raw_params, dome, material, config)


def optimize_patch_layout(
    dome_config: DomeConfig | None = None,
    material: MaterialConfig | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, object]:
    dome = build_type_iv_dome(dome_config)
    material = material or MaterialConfig()
    config = config or OptimizationConfig()

    raw_params = initial_raw_layout(dome, config)
    baseline = evaluate_baseline_layout(dome=dome, material=material, config=config)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.grad_clip_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(raw_params)
    best_params = raw_params
    best_loss = jnp.inf
    history: list[dict[str, float]] = []

    def loss_fn(params: jnp.ndarray) -> tuple[jnp.ndarray, dict[str, object]]:
        aux = evaluate_layout(params, dome, material, config)
        return aux["loss"], aux

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    for step in range(config.steps):
        current_params = raw_params
        (loss_value, aux), grads = value_and_grad(current_params)
        updates, opt_state = optimizer.update(grads, opt_state, current_params)
        raw_params = optax.apply_updates(current_params, updates)

        if loss_value < best_loss:
            best_loss = loss_value
            best_params = current_params

        if step % config.history_stride == 0 or step == config.steps - 1:
            history.append(
                {
                    "step": float(step),
                    "loss": _scalarize(aux, "metrics.loss"),
                    "structural_loss": _scalarize(aux, "metrics.structural_loss"),
                    "kinematic_penalty": _scalarize(aux, "metrics.kinematic_penalty"),
                    "thickness_penalty": _scalarize(aux, "metrics.thickness_penalty"),
                    "total_mass_kg": _scalarize(aux, "metrics.total_mass_kg"),
                    "peak_stress_index": _scalarize(aux, "metrics.peak_stress_index"),
                    "max_shear": _scalarize(aux, "metrics.max_shear"),
                }
            )

    final = evaluate_layout(best_params, dome, material, config)
    result = {
        "dome_config": asdict(dome.config),
        "material_config": asdict(material),
        "optimization_config": asdict(config),
        "dome": dome,
        "layout": final["layout"],
        "baseline": baseline,
        "optimized": final,
        "raw_params": np.asarray(best_params).tolist(),
        "history": history,
        "layout_serialized": _serialize_layout(final["layout"]),
    }
    return result
