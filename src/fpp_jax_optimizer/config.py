from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DomeConfig:
    minor_radius_m: float = 0.18
    major_to_minor_ratio: float = 1.414
    polar_opening_radius_m: float = 0.05
    theta_points: int = 40
    phi_points: int = 72

    @property
    def major_radius_m(self) -> float:
        return self.major_to_minor_ratio * self.minor_radius_m

    @property
    def theta_open(self) -> float:
        ratio = min(self.polar_opening_radius_m / self.major_radius_m, 0.98)
        return math.asin(ratio)


@dataclass(frozen=True)
class MaterialConfig:
    pressure_mpa: float = 70.0
    ply_thickness_mm: float = 0.125
    liner_thickness_mm: float = 0.60
    baseline_helical_plies: float = 4.0
    areal_density_kg_per_m2_per_ply: float = 0.145
    allowable_shear_strain: float = 0.015
    allowable_distortion: float = 0.015
    max_thickness_gradient_mm_per_m: float = 8.0
    helical_reference_angle_deg: float = 55.0
    helical_cost_per_kg: float = 110.0
    fpp_cost_per_kg: float = 185.0


@dataclass(frozen=True)
class OptimizationConfig:
    patch_count: int = 4
    steps: int = 90
    learning_rate: float = 0.03
    grad_clip_norm: float = 1.0
    patch_length_bounds_m: tuple[float, float] = (0.05, 0.18)
    patch_width_bounds_m: tuple[float, float] = (0.04, 0.12)
    max_patch_plies: float = 6.0
    mask_sharpness: float = 35.0
    transition_smooth_theta: float = 0.06
    boss_margin_theta: float = 0.05
    theta_upper_margin: float = 0.10
    stress_weight: float = 1.0
    shear_weight: float = 40.0
    thickness_weight: float = 18.0
    mass_weight: float = 0.08
    history_stride: int = 5


@dataclass(frozen=True)
class ExportConfig:
    output_dir: Path = Path("outputs")
    bdf_filename: str = "fpp_type_iv_dome.bdf"
    summary_filename: str = "optimization_summary.json"
    plotly_html_filename: str = "fpp_layout.html"
