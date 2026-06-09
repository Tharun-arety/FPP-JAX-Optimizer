from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..config import MaterialConfig, OptimizationConfig
from ..core.geometry import DomeState, surface_tangents


def _bounded(raw: jnp.ndarray, low: float, high: float) -> jnp.ndarray:
    return low + (high - low) * jax.nn.sigmoid(raw)


def _inverse_bounded(value: np.ndarray | float, low: float, high: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    scaled = np.clip((value - low) / (high - low), 1.0e-5, 1.0 - 1.0e-5)
    return np.log(scaled / (1.0 - scaled))


def _soft_abs(values: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    return jnp.sqrt(values**2 + eps)


def _wrap_phi(phi_delta: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(phi_delta), jnp.cos(phi_delta))


def initial_raw_layout(dome: DomeState, config: OptimizationConfig | None = None) -> jnp.ndarray:
    config = config or OptimizationConfig()
    theta_low = dome.config.theta_open + config.boss_margin_theta
    theta_high = 0.5 * math.pi - config.theta_upper_margin
    length_low, length_high = config.patch_length_bounds_m
    width_low, width_high = config.patch_width_bounds_m

    seed_theta_low = min(theta_high, theta_low + 0.02)
    seed_theta_high = min(theta_high - 0.02, theta_low + 0.35)
    if seed_theta_high < seed_theta_low:
        seed_theta_high = seed_theta_low

    # Break the axisymmetric seed so patches can discover different meridional roles.
    centers_theta = np.linspace(seed_theta_low, seed_theta_high, config.patch_count, dtype=np.float32)
    centers_phi = 2.0 * np.pi * (np.arange(config.patch_count, dtype=np.float32) + 0.5) / config.patch_count
    lengths = np.full(config.patch_count, 0.11, dtype=np.float32)
    widths = np.full(config.patch_count, 0.075, dtype=np.float32)
    angles = np.deg2rad(np.linspace(15.0, 50.0, config.patch_count, dtype=np.float32))
    plies = np.full(config.patch_count, 2.0, dtype=np.float32)
    transition_theta = theta_low + 0.55 * (theta_high - theta_low)

    patch_raw = np.stack(
        [
            _inverse_bounded(centers_theta, theta_low, theta_high),
            _inverse_bounded(centers_phi, 0.0, 2.0 * np.pi),
            _inverse_bounded(lengths, length_low, length_high),
            _inverse_bounded(widths, width_low, width_high),
            _inverse_bounded(angles, -0.5 * np.pi, 0.5 * np.pi),
            _inverse_bounded(plies, 0.0, config.max_patch_plies),
        ],
        axis=-1,
    ).reshape(-1)
    transition_raw = _inverse_bounded(transition_theta, theta_low, theta_high)
    return jnp.asarray(np.concatenate([patch_raw, np.asarray([transition_raw], dtype=np.float32)]), dtype=jnp.float32)


def decode_layout(raw_params: jnp.ndarray, dome: DomeState, config: OptimizationConfig | None = None) -> dict[str, jnp.ndarray]:
    config = config or OptimizationConfig()
    patch_raw = raw_params[:-1].reshape(config.patch_count, 6)
    theta_low = dome.config.theta_open + config.boss_margin_theta
    theta_high = 0.5 * math.pi - config.theta_upper_margin
    length_low, length_high = config.patch_length_bounds_m
    width_low, width_high = config.patch_width_bounds_m

    layout = {
        "center_theta": _bounded(patch_raw[:, 0], theta_low, theta_high),
        "center_phi": _bounded(patch_raw[:, 1], 0.0, 2.0 * math.pi),
        "length_m": _bounded(patch_raw[:, 2], length_low, length_high),
        "width_m": _bounded(patch_raw[:, 3], width_low, width_high),
        "angle_rad": _bounded(patch_raw[:, 4], -0.5 * math.pi, 0.5 * math.pi),
        "plies": _bounded(patch_raw[:, 5], 0.0, config.max_patch_plies),
        "transition_theta": _bounded(raw_params[-1], theta_low, theta_high),
    }
    return layout


def patch_masks_from_layout(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    config: OptimizationConfig | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    config = config or OptimizationConfig()

    center_theta = layout["center_theta"][:, None, None]
    center_phi = layout["center_phi"][:, None, None]
    angle = layout["angle_rad"][:, None, None]
    length_m = layout["length_m"][:, None, None]
    width_m = layout["width_m"][:, None, None]

    center_tangent_theta, center_tangent_phi = surface_tangents(dome.config, center_theta, center_phi)
    center_scale_theta = jnp.linalg.norm(center_tangent_theta, axis=-1)
    center_scale_phi = jnp.linalg.norm(center_tangent_phi, axis=-1)

    delta_theta = dome.theta_grid[None, :, :] - center_theta
    delta_phi = _wrap_phi(dome.phi_grid[None, :, :] - center_phi)
    meridional = delta_theta * center_scale_theta
    circumferential = delta_phi * center_scale_phi

    u = jnp.cos(angle) * meridional + jnp.sin(angle) * circumferential
    v = -jnp.sin(angle) * meridional + jnp.cos(angle) * circumferential

    edge_u = jax.nn.sigmoid(config.mask_sharpness * (0.5 * length_m - _soft_abs(u)))
    edge_v = jax.nn.sigmoid(config.mask_sharpness * (0.5 * width_m - _soft_abs(v)))
    masks = edge_u * edge_v

    return masks, {
        "u_local_m": u,
        "v_local_m": v,
        "center_scale_theta": center_scale_theta,
        "center_scale_phi": center_scale_phi,
    }


def evaluate_thickness_state(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    material: MaterialConfig | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, jnp.ndarray]:
    material = material or MaterialConfig()
    config = config or OptimizationConfig()

    masks, local = patch_masks_from_layout(dome, layout, config)
    transition_theta = layout["transition_theta"]
    helical_mask = jax.nn.sigmoid((dome.theta_grid - transition_theta) / config.transition_smooth_theta)
    baseline_plies_map = material.baseline_helical_plies * helical_mask
    patch_plies_map = jnp.sum(masks * layout["plies"][:, None, None], axis=0)
    total_plies = baseline_plies_map + patch_plies_map
    thickness_mm = material.liner_thickness_mm + material.ply_thickness_mm * total_plies

    delta_theta = (dome.theta[-1] - dome.theta[0]) / max(dome.theta.shape[0] - 1, 1)
    delta_phi = 2.0 * jnp.pi / dome.phi.shape[0]
    ds_theta = jnp.maximum(dome.scale_theta[:-1, :] * delta_theta, 1.0e-6)
    ds_phi = jnp.maximum(dome.scale_phi * delta_phi, 1.0e-6)

    grad_theta_core = (thickness_mm[1:, :] - thickness_mm[:-1, :]) / ds_theta
    grad_theta = jnp.concatenate([grad_theta_core[:1, :], grad_theta_core], axis=0)
    grad_phi = (jnp.roll(thickness_mm, -1, axis=1) - thickness_mm) / ds_phi
    thickness_gradient_mm_per_m = jnp.sqrt(grad_theta**2 + grad_phi**2)

    patch_mass_kg = jnp.sum(dome.area_weights * patch_plies_map * material.areal_density_kg_per_m2_per_ply)
    helical_mass_kg = jnp.sum(dome.area_weights * baseline_plies_map * material.areal_density_kg_per_m2_per_ply)
    total_mass_kg = patch_mass_kg + helical_mass_kg
    hybrid_cost = patch_mass_kg * material.fpp_cost_per_kg + helical_mass_kg * material.helical_cost_per_kg
    all_fpp_cost = jnp.maximum((patch_mass_kg + helical_mass_kg) * material.fpp_cost_per_kg, 1.0e-8)
    cost_savings_pct = 100.0 * (1.0 - hybrid_cost / all_fpp_cost)

    return {
        "masks": masks,
        "u_local_m": local["u_local_m"],
        "v_local_m": local["v_local_m"],
        "baseline_plies_map": baseline_plies_map,
        "patch_plies_map": patch_plies_map,
        "total_plies": total_plies,
        "thickness_mm": thickness_mm,
        "thickness_gradient_mm_per_m": thickness_gradient_mm_per_m,
        "helical_mask": helical_mask,
        "patch_mass_kg": patch_mass_kg,
        "helical_mass_kg": helical_mass_kg,
        "total_mass_kg": total_mass_kg,
        "hybrid_cost_index": hybrid_cost,
        "cost_savings_vs_all_fpp_pct": cost_savings_pct,
    }
