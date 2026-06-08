from __future__ import annotations

import math

import jax.numpy as jnp

from ..config import MaterialConfig
from ..core.geometry import DomeState


def preferred_patch_angle_map(dome: DomeState) -> jnp.ndarray:
    theta_span = jnp.maximum(0.5 * jnp.pi - dome.config.theta_open, 1.0e-6)
    theta_norm = (dome.theta_grid - dome.config.theta_open) / theta_span
    preferred = 0.05 * jnp.pi + 0.35 * jnp.pi * jnp.exp(-(theta_norm**2) / (2.0 * 0.25**2))
    preferred = preferred + 0.03 * jnp.pi * (1.0 - theta_norm)
    return jnp.clip(preferred, 0.0, 0.5 * jnp.pi)


def evaluate_structural_response(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    material: MaterialConfig | None = None,
) -> dict[str, jnp.ndarray]:
    material = material or MaterialConfig()

    preferred_angle = preferred_patch_angle_map(dome)
    masks = thickness_state["masks"]
    patch_plies_map = thickness_state["patch_plies_map"]
    baseline_plies_map = thickness_state["baseline_plies_map"]
    transition_theta = layout["transition_theta"]

    weights = masks * layout["plies"][:, None, None]
    angle = layout["angle_rad"][:, None, None]
    weight_sum = jnp.maximum(jnp.sum(weights, axis=0), 1.0e-8)
    cos2 = jnp.sum(weights * jnp.cos(2.0 * angle), axis=0) / weight_sum
    sin2 = jnp.sum(weights * jnp.sin(2.0 * angle), axis=0) / weight_sum
    patch_angle_field = 0.5 * jnp.arctan2(sin2, cos2)

    helical_angle = math.radians(material.helical_reference_angle_deg)
    patch_alignment = 0.35 + 0.65 * jnp.cos(patch_angle_field - preferred_angle) ** 2
    helical_alignment = 0.30 + 0.70 * jnp.cos(helical_angle - preferred_angle) ** 2

    effective_patch_plies = patch_plies_map * patch_alignment
    effective_helical_plies = baseline_plies_map * helical_alignment
    effective_total_plies = effective_patch_plies + effective_helical_plies

    transition_factor = 1.0 + 0.18 * jnp.exp(-((dome.theta_grid - transition_theta) ** 2) / (2.0 * 0.10**2))
    geometry_factor = (1.0 + 0.95 * dome.boss_ring) * dome.curvature_indicator * transition_factor
    denominator = jnp.maximum(0.9 + 0.22 * effective_total_plies, 0.2)
    stress_index = (material.pressure_mpa / 70.0) * geometry_factor / denominator

    mean_loss = jnp.sum(dome.area_weights * stress_index**2) / jnp.maximum(dome.total_area_m2, 1.0e-8)
    peak_loss = 0.35 * jnp.max(stress_index) ** 2
    structural_loss = mean_loss + peak_loss

    transition_height_m = dome.config.minor_radius_m * jnp.cos(transition_theta)
    return {
        "preferred_angle_map": preferred_angle,
        "patch_angle_field": patch_angle_field,
        "effective_total_plies": effective_total_plies,
        "stress_index": stress_index,
        "structural_loss": structural_loss,
        "peak_stress_index": jnp.max(stress_index),
        "mean_stress_index": jnp.sum(dome.area_weights * stress_index) / jnp.maximum(dome.total_area_m2, 1.0e-8),
        "transition_height_m": transition_height_m,
    }
