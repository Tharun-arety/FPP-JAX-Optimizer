import math

import jax.numpy as jnp
import numpy as np

from fpp_jax_optimizer.config import DomeConfig, MaterialConfig, OptimizationConfig
from fpp_jax_optimizer.core.geometry import build_type_iv_dome
from fpp_jax_optimizer.model.fem import evaluate_structural_response, membrane_stress_resultants
from fpp_jax_optimizer.topology.mapping import decode_layout, evaluate_thickness_state, initial_raw_layout


def _single_patch_layout(dome: object, angle_rad: float) -> dict[str, jnp.ndarray]:
    return {
        "center_theta": jnp.asarray([dome.config.theta_open + 0.07], dtype=jnp.float32),
        "center_phi": jnp.asarray([0.0], dtype=jnp.float32),
        "length_m": jnp.asarray([0.10], dtype=jnp.float32),
        "width_m": jnp.asarray([0.10], dtype=jnp.float32),
        "angle_rad": jnp.asarray([angle_rad], dtype=jnp.float32),
        "plies": jnp.asarray([4.0], dtype=jnp.float32),
        "transition_theta": jnp.asarray(dome.config.theta_open + 0.22, dtype=jnp.float32),
    }


def test_membrane_resultants_reduce_to_spherical_solution() -> None:
    dome = build_type_iv_dome(DomeConfig(major_to_minor_ratio=1.0, theta_points=18, phi_points=24))
    material = MaterialConfig()

    N_phi, N_theta = membrane_stress_resultants(dome, material)
    expected = material.pressure_mpa * 1.0e6 * dome.config.major_radius_m / 2.0

    np.testing.assert_allclose(np.asarray(N_phi), expected, rtol=1.0e-5)
    np.testing.assert_allclose(np.asarray(N_theta), expected, rtol=1.0e-5)


def test_initial_layout_breaks_axisymmetric_seed_symmetry() -> None:
    dome = build_type_iv_dome(DomeConfig(theta_points=24, phi_points=36))
    config = OptimizationConfig(patch_count=4)
    layout = decode_layout(initial_raw_layout(dome, config), dome, config)

    assert float(jnp.ptp(layout["center_theta"])) > 0.0
    assert float(jnp.ptp(layout["angle_rad"])) > 0.0


def test_boss_region_response_changes_with_patch_angle() -> None:
    dome = build_type_iv_dome(DomeConfig(theta_points=24, phi_points=36))
    material = MaterialConfig()
    config = OptimizationConfig(patch_count=1)

    meridional_layout = _single_patch_layout(dome, 0.0)
    circumferential_layout = _single_patch_layout(dome, 0.5 * math.pi)

    meridional_thickness = evaluate_thickness_state(dome, meridional_layout, material, config)
    circumferential_thickness = evaluate_thickness_state(dome, circumferential_layout, material, config)

    meridional_response = evaluate_structural_response(dome, meridional_layout, meridional_thickness, material)
    circumferential_response = evaluate_structural_response(
        dome,
        circumferential_layout,
        circumferential_thickness,
        material,
    )

    support_region = np.asarray(meridional_thickness["masks"][0]) > 0.5
    meridional_fi = np.asarray(meridional_response["stress_index"])[support_region]
    circumferential_fi = np.asarray(circumferential_response["stress_index"])[support_region]

    assert float(np.mean(meridional_fi)) < float(np.mean(circumferential_fi))


def test_reported_peak_stress_is_not_the_weighted_surrogate_field() -> None:
    dome = build_type_iv_dome(DomeConfig(theta_points=24, phi_points=36))
    material = MaterialConfig()
    config = OptimizationConfig(patch_count=1)
    layout = {
        "center_theta": jnp.asarray([dome.config.theta_open + 0.08], dtype=jnp.float32),
        "center_phi": jnp.asarray([0.0], dtype=jnp.float32),
        "length_m": jnp.asarray([0.10], dtype=jnp.float32),
        "width_m": jnp.asarray([0.10], dtype=jnp.float32),
        "angle_rad": jnp.asarray([0.0], dtype=jnp.float32),
        "plies": jnp.asarray([0.75], dtype=jnp.float32),
        "transition_theta": jnp.asarray(dome.config.theta_open + 0.28, dtype=jnp.float32),
    }

    thickness = evaluate_thickness_state(dome, layout, material, config)
    response = evaluate_structural_response(dome, layout, thickness, material)

    support_region = np.asarray(thickness["patch_effective_plies"][0]) >= material.report_min_patch_plies
    reported = np.asarray(response["stress_index"])
    surrogate = np.asarray(response["surrogate_stress_index"])

    assert bool(support_region.any())
    assert float(np.max(reported[support_region])) > float(np.max(surrogate[support_region]))
