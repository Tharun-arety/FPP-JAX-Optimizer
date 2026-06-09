from __future__ import annotations

import jax
import jax.numpy as jnp

from ..config import MaterialConfig, OptimizationConfig
from ..topology.mapping import patch_masks_from_layout
from .geometry import DomeState, surface_tangents


def evaluate_kinematics(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    material: MaterialConfig | None = None,
    config: OptimizationConfig | None = None,
) -> dict[str, jnp.ndarray]:
    material = material or MaterialConfig()
    config = config or OptimizationConfig()

    masks, local = patch_masks_from_layout(dome, layout, config)
    effective_patch_plies = masks * layout["plies"][:, None, None]
    active_weight = jnp.clip(effective_patch_plies / material.support_full_weight_plies, 0.0, 1.0)
    report_mask = effective_patch_plies >= material.report_min_patch_plies
    angle = layout["angle_rad"][:, None, None]
    center_theta = layout["center_theta"][:, None, None]
    center_phi = layout["center_phi"][:, None, None]

    center_tangent_theta, center_tangent_phi = surface_tangents(dome.config, center_theta, center_phi)
    center_scale_theta = jnp.linalg.norm(center_tangent_theta, axis=-1)
    center_scale_phi = jnp.linalg.norm(center_tangent_phi, axis=-1)

    dtheta_du = jnp.cos(angle) / jnp.maximum(center_scale_theta, 1.0e-6)
    dphi_du = jnp.sin(angle) / jnp.maximum(center_scale_phi, 1.0e-6)
    dtheta_dv = -jnp.sin(angle) / jnp.maximum(center_scale_theta, 1.0e-6)
    dphi_dv = jnp.cos(angle) / jnp.maximum(center_scale_phi, 1.0e-6)

    tangent_theta = dome.tangent_theta[None, :, :, :]
    tangent_phi = dome.tangent_phi[None, :, :, :]
    ju = tangent_theta * dtheta_du[..., None] + tangent_phi * dphi_du[..., None]
    jv = tangent_theta * dtheta_dv[..., None] + tangent_phi * dphi_dv[..., None]

    g11 = jnp.sum(ju * ju, axis=-1)
    g22 = jnp.sum(jv * jv, axis=-1)
    g12 = jnp.sum(ju * jv, axis=-1)
    stretch_u = jnp.abs(jnp.sqrt(jnp.maximum(g11, 1.0e-8)) - 1.0)
    stretch_v = jnp.abs(jnp.sqrt(jnp.maximum(g22, 1.0e-8)) - 1.0)
    areal_distortion = jnp.abs(jnp.sqrt(jnp.maximum(g11 * g22 - g12**2, 1.0e-8)) - 1.0)
    shear_strain = jnp.abs(g12)

    shear_excess = jax.nn.relu(shear_strain - material.allowable_shear_strain)
    distortion_excess = jax.nn.relu(areal_distortion - material.allowable_distortion)
    stretch_excess = jax.nn.relu(jnp.maximum(stretch_u, stretch_v) - material.allowable_distortion)
    penalty_map_per_patch = active_weight * (shear_excess**2 + distortion_excess**2 + 0.5 * stretch_excess**2)

    active_mask = jnp.sum(active_weight, axis=0)
    mean_penalty = jnp.sum(dome.area_weights * jnp.sum(penalty_map_per_patch, axis=0)) / jnp.maximum(
        dome.total_area_m2,
        1.0e-8,
    )
    peak_violation = jnp.maximum(
        jnp.max(active_weight * shear_excess),
        jnp.maximum(jnp.max(active_weight * distortion_excess), jnp.max(active_weight * stretch_excess)),
    )
    penalty = mean_penalty + 25.0 * peak_violation**2

    weighted_denominator = jnp.maximum(active_mask, 1.0e-8)
    combined_shear_map = jnp.sum(active_weight * shear_strain, axis=0) / weighted_denominator
    combined_areal_map = jnp.sum(active_weight * areal_distortion, axis=0) / weighted_denominator
    combined_wrinkle_map = jnp.maximum(combined_shear_map, combined_areal_map)
    reported_shear = jnp.where(report_mask, shear_strain, 0.0)
    reported_areal = jnp.where(report_mask, areal_distortion, 0.0)

    return {
        "masks": masks,
        "active_weight": active_weight,
        "report_mask": report_mask,
        "u_local_m": local["u_local_m"],
        "v_local_m": local["v_local_m"],
        "shear_per_patch": shear_strain,
        "areal_distortion_per_patch": areal_distortion,
        "stretch_u_per_patch": stretch_u,
        "stretch_v_per_patch": stretch_v,
        "shear_map": combined_shear_map,
        "areal_distortion_map": combined_areal_map,
        "wrinkle_risk_map": combined_wrinkle_map,
        "penalty": penalty,
        "peak_violation": peak_violation,
        "max_shear": jnp.max(reported_shear),
        "max_areal_distortion": jnp.max(reported_areal),
    }
