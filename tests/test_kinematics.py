import jax.numpy as jnp

from fpp_jax_optimizer.config import DomeConfig, MaterialConfig, OptimizationConfig
from fpp_jax_optimizer.core.geometry import build_type_iv_dome
from fpp_jax_optimizer.core.kinematics import evaluate_kinematics


def test_jacobian_penalty_rises_near_polar_opening() -> None:
    dome = build_type_iv_dome(DomeConfig(theta_points=28, phi_points=48))
    material = MaterialConfig()
    config = OptimizationConfig(patch_count=1)
    layout = {
        "center_theta": jnp.asarray([dome.config.theta_open + 0.018], dtype=jnp.float32),
        "center_phi": jnp.asarray([0.0], dtype=jnp.float32),
        "length_m": jnp.asarray([0.18], dtype=jnp.float32),
        "width_m": jnp.asarray([0.12], dtype=jnp.float32),
        "angle_rad": jnp.asarray([0.5 * jnp.pi - 0.03], dtype=jnp.float32),
        "plies": jnp.asarray([3.0], dtype=jnp.float32),
        "transition_theta": jnp.asarray(dome.config.theta_open + 0.2, dtype=jnp.float32),
    }

    kinematics = evaluate_kinematics(dome, layout, material, config)
    assert float(kinematics["max_shear"]) > material.allowable_shear_strain
    assert float(kinematics["penalty"]) > 0.0
