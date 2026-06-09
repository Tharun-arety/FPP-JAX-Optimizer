from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..config import MaterialConfig
from ..core.geometry import DomeState


def preferred_patch_angle_map(N_phi: jnp.ndarray, N_theta: jnp.ndarray) -> jnp.ndarray:
    ratio = jnp.maximum(N_theta / jnp.maximum(N_phi, 1.0e-8), 0.0)
    return jnp.arctan(jnp.sqrt(ratio))


def patch_angle_field(layout: dict[str, jnp.ndarray], thickness_state: dict[str, jnp.ndarray]) -> jnp.ndarray:
    weights = thickness_state["masks"] * layout["plies"][:, None, None]
    weight_sum = jnp.maximum(jnp.sum(weights, axis=0), 1.0e-8)
    cos2 = jnp.sum(weights * jnp.cos(2.0 * layout["angle_rad"][:, None, None]), axis=0) / weight_sum
    sin2 = jnp.sum(weights * jnp.sin(2.0 * layout["angle_rad"][:, None, None]), axis=0) / weight_sum
    return 0.5 * jnp.arctan2(sin2, cos2)


def ply_Q_matrix(material: MaterialConfig) -> jnp.ndarray:
    E1 = material.E1_gpa * 1.0e9
    E2 = material.E2_gpa * 1.0e9
    G12 = material.G12_gpa * 1.0e9
    nu12 = material.nu12
    nu21 = nu12 * E2 / E1
    denom = 1.0 - nu12 * nu21
    Q11 = E1 / denom
    Q22 = E2 / denom
    Q12 = nu12 * E2 / denom
    Q66 = G12
    return jnp.array(
        [
            [Q11, Q12, 0.0],
            [Q12, Q22, 0.0],
            [0.0, 0.0, Q66],
        ]
    )


def qbar_matrix(Q: jnp.ndarray, angle_rad: jnp.ndarray) -> jnp.ndarray:
    c = jnp.cos(angle_rad)
    s = jnp.sin(angle_rad)
    c2 = c * c
    s2 = s * s
    c4 = c2 * c2
    s4 = s2 * s2
    s2c2 = s2 * c2

    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    Qb11 = Q11 * c4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * s4
    Qb22 = Q11 * s4 + 2.0 * (Q12 + 2.0 * Q66) * s2c2 + Q22 * c4
    Qb12 = (Q11 + Q22 - 4.0 * Q66) * s2c2 + Q12 * (c4 + s4)
    Qb66 = (Q11 + Q22 - 2.0 * Q12 - 2.0 * Q66) * s2c2 + Q66 * (c4 + s4)
    Qb16 = (Q11 - Q12 - 2.0 * Q66) * c * c2 * s - (Q22 - Q12 - 2.0 * Q66) * c * s * s2
    Qb26 = (Q11 - Q12 - 2.0 * Q66) * c * s * s2 - (Q22 - Q12 - 2.0 * Q66) * c * c2 * s

    return jnp.stack(
        [
            jnp.stack([Qb11, Qb12, Qb16], axis=-1),
            jnp.stack([Qb12, Qb22, Qb26], axis=-1),
            jnp.stack([Qb16, Qb26, Qb66], axis=-1),
        ],
        axis=-2,
    )


def membrane_stress_resultants(
    dome: DomeState,
    material: MaterialConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    a = dome.config.major_radius_m
    c = dome.config.minor_radius_m
    p = material.pressure_mpa * 1.0e6

    theta = dome.theta_grid
    f = jnp.sqrt(a**2 * jnp.cos(theta) ** 2 + c**2 * jnp.sin(theta) ** 2)

    N_phi = p * a * f / (2.0 * c)
    N_theta = p * a * (2.0 * f**2 - a**2) / (2.0 * c * f)
    return N_phi, N_theta


def assemble_a_matrix_field(
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    Q: jnp.ndarray,
    material: MaterialConfig,
) -> jnp.ndarray:
    t = material.ply_thickness_mm * 1.0e-3

    alpha = jnp.asarray(math.radians(material.helical_reference_angle_deg))
    qbar_helical = 0.5 * (qbar_matrix(Q, alpha) + qbar_matrix(Q, -alpha))
    helical_t = thickness_state["baseline_plies_map"] * t
    A_helical = helical_t[..., None, None] * qbar_helical

    patch_t = thickness_state["masks"] * layout["plies"][:, None, None] * t
    qbar_patches = qbar_matrix(Q, layout["angle_rad"])
    A_patches = jnp.einsum("ijk,iab->jkab", patch_t, qbar_patches)

    qbar_floor = qbar_matrix(Q, alpha)
    floor_diagonal = jnp.diag(qbar_floor) * t * 0.5
    A_floor = jnp.diag(floor_diagonal)
    return A_helical + A_patches + A_floor


def midplane_strains(
    A_field: jnp.ndarray,
    N_phi: jnp.ndarray,
    N_theta: jnp.ndarray,
) -> jnp.ndarray:
    N_vec = jnp.stack([N_phi, N_theta, jnp.zeros_like(N_phi)], axis=-1)
    return jnp.linalg.solve(A_field, N_vec[..., None]).squeeze(-1)


def tsai_wu_field(
    A_field: jnp.ndarray,
    N_phi: jnp.ndarray,
    N_theta: jnp.ndarray,
    Q: jnp.ndarray,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    material: MaterialConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    Xt = material.Xt_mpa * 1.0e6
    Xc = material.Xc_mpa * 1.0e6
    Yt = material.Yt_mpa * 1.0e6
    Yc = material.Yc_mpa * 1.0e6
    S = material.S_mpa * 1.0e6
    F1 = 1.0 / Xt - 1.0 / Xc
    F2 = 1.0 / Yt - 1.0 / Yc
    F11 = 1.0 / (Xt * Xc)
    F22 = 1.0 / (Yt * Yc)
    F66 = 1.0 / (S * S)
    F12 = -0.5 / jnp.sqrt(Xt * Xc * Yt * Yc)

    eps = midplane_strains(A_field, N_phi, N_theta)
    preferred_angle = preferred_patch_angle_map(N_phi, N_theta)
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]

    def ply_tsai_wu(angle_rad: jnp.ndarray) -> jnp.ndarray:
        c = jnp.cos(angle_rad)
        s = jnp.sin(angle_rad)
        ex, ey, gxy = eps[..., 0], eps[..., 1], eps[..., 2]

        e1 = ex * c**2 + ey * s**2 + gxy * s * c
        e2 = ex * s**2 + ey * c**2 - gxy * s * c
        g12 = -2.0 * ex * s * c + 2.0 * ey * s * c + gxy * (c**2 - s**2)

        s1 = Q11 * e1 + Q12 * e2
        s2 = Q12 * e1 + Q22 * e2
        t12 = Q66 * g12

        return (
            F1 * s1
            + F2 * s2
            + F11 * s1**2
            + F22 * s2**2
            + F66 * t12**2
            + 2.0 * F12 * s1 * s2
        )

    patch_presence = 1.0 - jnp.exp(-(thickness_state["masks"] * layout["plies"][:, None, None]))
    fi_patches = patch_presence * jax.vmap(ply_tsai_wu)(layout["angle_rad"])

    alpha = jnp.asarray(math.radians(material.helical_reference_angle_deg))
    helical_presence = 1.0 - jnp.exp(-0.5 * thickness_state["baseline_plies_map"])
    fi_helical_pos = helical_presence * ply_tsai_wu(alpha)
    fi_helical_neg = helical_presence * ply_tsai_wu(-alpha)

    fi_max = jnp.max(
        jnp.concatenate(
            [
                fi_patches,
                fi_helical_pos[None, ...],
                fi_helical_neg[None, ...],
            ],
            axis=0,
        ),
        axis=0,
    )
    return fi_max, eps, preferred_angle


def evaluate_structural_response(
    dome: DomeState,
    layout: dict[str, jnp.ndarray],
    thickness_state: dict[str, jnp.ndarray],
    material: MaterialConfig | None = None,
) -> dict[str, jnp.ndarray]:
    material = material or MaterialConfig()
    N_phi, N_theta = membrane_stress_resultants(dome, material)
    Q = ply_Q_matrix(material)
    A_field = assemble_a_matrix_field(layout, thickness_state, Q, material)
    stress_index, eps, preferred_angle = tsai_wu_field(
        A_field,
        N_phi,
        N_theta,
        Q,
        layout,
        thickness_state,
        material,
    )
    patch_angles = patch_angle_field(layout, thickness_state)

    mean_loss = jnp.sum(dome.area_weights * stress_index**2) / jnp.maximum(dome.total_area_m2, 1.0e-8)
    peak_loss = 0.35 * jnp.max(stress_index) ** 2
    structural_loss = mean_loss + peak_loss

    transition_height_m = dome.config.minor_radius_m * jnp.cos(layout["transition_theta"])
    return {
        "preferred_angle_map": preferred_angle,
        "patch_angle_field": patch_angles,
        "effective_total_plies": thickness_state["total_plies"],
        "stress_index": stress_index,
        "structural_loss": structural_loss,
        "peak_stress_index": jnp.max(stress_index),
        "mean_stress_index": jnp.sum(dome.area_weights * stress_index) / jnp.maximum(dome.total_area_m2, 1.0e-8),
        "transition_height_m": transition_height_m,
        "tsai_wu_field": stress_index,
        "N_phi_field": N_phi,
        "N_theta_field": N_theta,
        "eps_field": eps,
        "A_field": A_field,
    }
