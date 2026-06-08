import jax
import jax.numpy as jnp

from fpp_jax_optimizer.config import DomeConfig, MaterialConfig, OptimizationConfig
from fpp_jax_optimizer.core.geometry import build_type_iv_dome
from fpp_jax_optimizer.model.loss import evaluate_layout
from fpp_jax_optimizer.topology.mapping import initial_raw_layout


def test_gradients_flow_through_sigmoid_patch_masks() -> None:
    dome = build_type_iv_dome(DomeConfig(theta_points=24, phi_points=36))
    material = MaterialConfig()
    config = OptimizationConfig(patch_count=2, steps=4)
    raw_params = initial_raw_layout(dome, config)

    def scalar_loss(params: jnp.ndarray) -> jnp.ndarray:
        return evaluate_layout(params, dome, material, config)["loss"]

    gradient = jax.grad(scalar_loss)(raw_params)
    assert gradient.shape == raw_params.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
