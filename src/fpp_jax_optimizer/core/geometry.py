from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..config import DomeConfig


@dataclass(frozen=True)
class DomeState:
    config: DomeConfig
    theta: jnp.ndarray
    phi: jnp.ndarray
    theta_grid: jnp.ndarray
    phi_grid: jnp.ndarray
    xyz: jnp.ndarray
    tangent_theta: jnp.ndarray
    tangent_phi: jnp.ndarray
    normals: jnp.ndarray
    scale_theta: jnp.ndarray
    scale_phi: jnp.ndarray
    area_weights: jnp.ndarray
    curvature_indicator: jnp.ndarray
    boss_ring: jnp.ndarray
    total_area_m2: float


def _normalize(vector: jnp.ndarray, eps: float = 1.0e-8) -> jnp.ndarray:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / jnp.maximum(norm, eps)


def surface_xyz(config: DomeConfig, theta: jnp.ndarray, phi: jnp.ndarray) -> jnp.ndarray:
    a = config.major_radius_m
    c = config.minor_radius_m
    x = a * jnp.sin(theta) * jnp.cos(phi)
    y = a * jnp.sin(theta) * jnp.sin(phi)
    z = c * jnp.cos(theta)
    return jnp.stack([x, y, z], axis=-1)


def surface_tangents(config: DomeConfig, theta: jnp.ndarray, phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    a = config.major_radius_m
    c = config.minor_radius_m
    tangent_theta = jnp.stack(
        [
            a * jnp.cos(theta) * jnp.cos(phi),
            a * jnp.cos(theta) * jnp.sin(phi),
            -c * jnp.sin(theta),
        ],
        axis=-1,
    )
    tangent_phi = jnp.stack(
        [
            -a * jnp.sin(theta) * jnp.sin(phi),
            a * jnp.sin(theta) * jnp.cos(phi),
            jnp.zeros_like(theta),
        ],
        axis=-1,
    )
    return tangent_theta, tangent_phi


def build_type_iv_dome(config: DomeConfig | None = None) -> DomeState:
    config = config or DomeConfig()
    theta = jnp.linspace(config.theta_open, 0.5 * jnp.pi, config.theta_points)
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, config.phi_points, endpoint=False)
    theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")

    xyz = surface_xyz(config, theta_grid, phi_grid)
    tangent_theta, tangent_phi = surface_tangents(config, theta_grid, phi_grid)
    normals = _normalize(jnp.cross(tangent_theta, tangent_phi))
    scale_theta = jnp.linalg.norm(tangent_theta, axis=-1)
    scale_phi = jnp.linalg.norm(tangent_phi, axis=-1)

    dtheta = float((0.5 * np.pi - config.theta_open) / max(config.theta_points - 1, 1))
    dphi = float(2.0 * np.pi / config.phi_points)
    area_density = jnp.linalg.norm(jnp.cross(tangent_theta, tangent_phi), axis=-1)
    area_weights = area_density * dtheta * dphi

    theta_span = max(float(0.5 * np.pi - config.theta_open), 1.0e-6)
    theta_norm = (theta_grid - config.theta_open) / theta_span
    boss_ring = jnp.exp(-((theta_norm - 0.08) ** 2) / (2.0 * 0.10**2))
    curvature_indicator = 1.0 + 0.45 * (1.0 / jnp.clip(jnp.sin(theta_grid), 0.12, 1.0) - 1.0)

    return DomeState(
        config=config,
        theta=theta,
        phi=phi,
        theta_grid=theta_grid,
        phi_grid=phi_grid,
        xyz=xyz,
        tangent_theta=tangent_theta,
        tangent_phi=tangent_phi,
        normals=normals,
        scale_theta=scale_theta,
        scale_phi=scale_phi,
        area_weights=area_weights,
        curvature_indicator=curvature_indicator,
        boss_ring=boss_ring,
        total_area_m2=float(jnp.sum(area_weights)),
    )


def preview_surface(dome: DomeState) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(dome.xyz).reshape(-1, 3)
    n_theta, n_phi = dome.theta_grid.shape
    faces: list[list[int]] = []
    for i in range(n_theta - 1):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + (j + 1) % n_phi
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + (j + 1) % n_phi
            faces.append([a, b, d])
            faces.append([a, d, c])
    return points, np.asarray(faces, dtype=np.int32)


def boss_opening_curve(dome: DomeState, points: int = 180) -> np.ndarray:
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, points, endpoint=True)
    theta = jnp.full_like(phi, dome.config.theta_open)
    return np.asarray(surface_xyz(dome.config, theta, phi))
