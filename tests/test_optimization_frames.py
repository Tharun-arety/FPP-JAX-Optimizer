from fpp_jax_optimizer.config import DomeConfig, OptimizationConfig
from fpp_jax_optimizer.model.loss import optimize_patch_layout


def test_optimizer_records_animation_frames() -> None:
    result = optimize_patch_layout(
        dome_config=DomeConfig(theta_points=16, phi_points=24),
        config=OptimizationConfig(patch_count=2, steps=4, history_stride=2),
    )

    frames = result["frames"]
    assert [frame["step"] for frame in frames] == [0.0, 2.0, 4.0]
    assert "layout" in frames[0]
    assert "metrics" in frames[-1]
    assert len(frames[-1]["layout"]["plies"]) == 2
