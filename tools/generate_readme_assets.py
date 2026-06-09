from __future__ import annotations

import math
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fpp_jax_optimizer import (
    MaterialConfig,
    OptimizationConfig,
    boss_opening_curve,
    optimize_patch_layout,
    summarize_result,
    surface_xyz,
)
from fpp_jax_optimizer.topology import decode_layout, evaluate_thickness_state, initial_raw_layout


def _as_numpy(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _layout_from_serialized(layout: dict[str, object]) -> dict[str, jnp.ndarray]:
    return {key: jnp.asarray(value, dtype=jnp.float32) for key, value in layout.items()}


def _percent_change(baseline: float, optimized: float) -> float:
    return 100.0 * (optimized - baseline) / max(abs(baseline), 1.0e-8)


def _set_equal_3d_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    x = xyz[:, :, 0]
    y = xyz[:, :, 1]
    z = xyz[:, :, 2]
    max_range = 0.5 * max(np.ptp(x), np.ptp(y), np.ptp(z))
    center_x = 0.5 * (np.max(x) + np.min(x))
    center_y = 0.5 * (np.max(y) + np.min(y))
    center_z = 0.5 * (np.max(z) + np.min(z))
    ax.set_xlim(center_x - max_range, center_x + max_range)
    ax.set_ylim(center_y - max_range, center_y + max_range)
    ax.set_zlim(center_z - max_range, center_z + max_range)


def _field_extent(dome: object) -> list[float]:
    theta_deg = np.rad2deg(_as_numpy(dome.theta))
    return [0.0, 360.0, float(theta_deg[0]), float(theta_deg[-1])]


def _metric_rows(summary: dict[str, float]) -> list[str]:
    baseline_peak = summary["baseline_peak_stress_index"]
    optimized_peak = summary["optimized_peak_stress_index"]
    baseline_mass = summary["baseline_mass_kg"]
    optimized_mass = summary["optimized_mass_kg"]
    patch_mass_g = 1000.0 * summary["patch_mass_kg"]

    result = [
        f"Peak Tsai-Wu index: {baseline_peak:.3f} -> {optimized_peak:.3f} ({_percent_change(baseline_peak, optimized_peak):+.1f}%)",
        f"Total mass: {baseline_mass:.4f} kg -> {optimized_mass:.4f} kg ({_percent_change(baseline_mass, optimized_mass):+.1f}%)",
        f"Patch reinforcement added: {patch_mass_g:.1f} g",
        f"Cost saving vs all-FPP laminate: {summary['cost_savings_vs_all_fpp_pct']:.1f}%",
        f"Transition height: {1000.0 * summary['transition_height_m']:.1f} mm",
    ]
    return result


def _patch_palette(count: int) -> list[str]:
    palette = [
        "#f94144",
        "#277da1",
        "#f8961e",
        "#43aa8b",
        "#6a4c93",
        "#f9844a",
        "#577590",
        "#90be6d",
    ]
    if count <= len(palette):
        return palette[:count]
    cmap = plt.get_cmap("tab20", count)
    return [colors.to_hex(cmap(index)) for index in range(count)]


def _patch_scale_factors(dome: object, theta_rad: float) -> tuple[float, float]:
    a = float(dome.config.major_radius_m)
    c = float(dome.config.minor_radius_m)
    scale_theta = math.sqrt((a * math.cos(theta_rad)) ** 2 + (c * math.sin(theta_rad)) ** 2)
    scale_phi = max(a * math.sin(theta_rad), 1.0e-8)
    return scale_theta, scale_phi


def _patch_perimeter_uv(length_m: float, width_m: float, edge_points: int = 40) -> tuple[np.ndarray, np.ndarray]:
    half_length = 0.5 * length_m
    half_width = 0.5 * width_m

    top_u = np.linspace(-half_length, half_length, edge_points)
    top_v = np.full(edge_points, half_width)
    right_u = np.full(edge_points, half_length)
    right_v = np.linspace(half_width, -half_width, edge_points)
    bottom_u = np.linspace(half_length, -half_length, edge_points)
    bottom_v = np.full(edge_points, -half_width)
    left_u = np.full(edge_points, -half_length)
    left_v = np.linspace(-half_width, half_width, edge_points)

    u = np.concatenate([top_u, right_u, bottom_u, left_u])
    v = np.concatenate([top_v, right_v, bottom_v, left_v])
    return u, v


def _patch_local_to_surface(
    dome: object,
    center_theta: float,
    center_phi: float,
    angle_rad: float,
    u_local: np.ndarray,
    v_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale_theta, scale_phi = _patch_scale_factors(dome, center_theta)

    meridional = np.cos(angle_rad) * u_local - np.sin(angle_rad) * v_local
    circumferential = np.sin(angle_rad) * u_local + np.cos(angle_rad) * v_local

    theta = np.clip(
        center_theta + meridional / max(scale_theta, 1.0e-8),
        float(np.min(_as_numpy(dome.theta))),
        float(np.max(_as_numpy(dome.theta))),
    )
    phi = np.mod(center_phi + circumferential / max(scale_phi, 1.0e-8), 2.0 * math.pi)
    xyz = _as_numpy(surface_xyz(dome.config, theta, phi))
    return theta, phi, xyz


def _patch_boundary_curve(dome: object, layout: dict[str, object], patch_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_theta = float(layout["center_theta"][patch_index])
    center_phi = float(layout["center_phi"][patch_index])
    angle_rad = float(layout["angle_rad"][patch_index])
    length_m = float(layout["length_m"][patch_index])
    width_m = float(layout["width_m"][patch_index])
    u_local, v_local = _patch_perimeter_uv(length_m, width_m)
    return _patch_local_to_surface(dome, center_theta, center_phi, angle_rad, u_local, v_local)


def _patch_axis_curve(dome: object, layout: dict[str, object], patch_index: int) -> np.ndarray:
    center_theta = float(layout["center_theta"][patch_index])
    center_phi = float(layout["center_phi"][patch_index])
    angle_rad = float(layout["angle_rad"][patch_index])
    length_m = float(layout["length_m"][patch_index])
    u_local = np.linspace(-0.42 * length_m, 0.42 * length_m, 30)
    v_local = np.zeros_like(u_local)
    _, _, xyz = _patch_local_to_surface(dome, center_theta, center_phi, angle_rad, u_local, v_local)
    return xyz


def _wrap_curve_for_plot(phi_rad: np.ndarray, theta_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi_deg = np.rad2deg(np.mod(phi_rad, 2.0 * math.pi))
    theta_deg = np.rad2deg(theta_rad)
    if phi_deg.size == 0:
        return phi_deg, theta_deg

    phi_plot = [float(phi_deg[0])]
    theta_plot = [float(theta_deg[0])]
    for index in range(1, phi_deg.size):
        if abs(float(phi_deg[index]) - float(phi_deg[index - 1])) > 180.0:
            phi_plot.append(float("nan"))
            theta_plot.append(float("nan"))
        phi_plot.append(float(phi_deg[index]))
        theta_plot.append(float(theta_deg[index]))
    return np.asarray(phi_plot), np.asarray(theta_plot)


def _patch_overlay_rgba(masks: np.ndarray, palette: list[str]) -> np.ndarray:
    dominant_patch = np.argmax(masks, axis=0)
    patch_strength = np.max(masks, axis=0)
    rgba = np.zeros(masks.shape[1:] + (4,), dtype=float)
    strength_norm = np.clip((patch_strength - 0.05) / 0.95, 0.0, 1.0)

    for patch_index, color in enumerate(palette):
        selected = (dominant_patch == patch_index) & (patch_strength > 0.06)
        rgba[selected, :3] = colors.to_rgb(color)
        rgba[selected, 3] = 0.22 + 0.58 * strength_norm[selected]
    return rgba


def _draw_patch_scene(
    ax: plt.Axes,
    dome: object,
    layout: dict[str, object],
    thickness_state: dict[str, object],
    title: str,
    palette: list[str],
) -> None:
    xyz = _as_numpy(dome.xyz)
    masks = _as_numpy(thickness_state["masks"])
    overlay_rgba = _patch_overlay_rgba(masks, palette)
    base_rgba = np.broadcast_to(np.asarray(colors.to_rgba("#efe7dc")), xyz.shape[:2] + (4,))

    ax.plot_surface(
        xyz[:, :, 0],
        xyz[:, :, 1],
        xyz[:, :, 2],
        rstride=1,
        cstride=1,
        facecolors=base_rgba,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=0,
    )
    ax.plot_wireframe(
        xyz[:, :, 0],
        xyz[:, :, 1],
        xyz[:, :, 2],
        rstride=4,
        cstride=6,
        color="#d8d0c4",
        linewidth=0.45,
        alpha=0.55,
    )
    ax.plot_surface(
        xyz[:, :, 0],
        xyz[:, :, 1],
        xyz[:, :, 2],
        rstride=1,
        cstride=1,
        facecolors=overlay_rgba,
        linewidth=0,
        antialiased=False,
        shade=False,
        zorder=2,
    )

    boss_curve = boss_opening_curve(dome)
    phi = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=True)
    transition_curve = _as_numpy(
        surface_xyz(dome.config, np.full_like(phi, float(layout["transition_theta"])), phi)
    )
    ax.plot(boss_curve[:, 0], boss_curve[:, 1], boss_curve[:, 2], color="#b91c1c", linewidth=2.3, zorder=5)
    ax.plot(
        transition_curve[:, 0],
        transition_curve[:, 1],
        transition_curve[:, 2],
        color="#111827",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        zorder=5,
    )

    centers = _as_numpy(surface_xyz(dome.config, layout["center_theta"], layout["center_phi"]))
    for patch_index, point in enumerate(centers):
        boundary_xyz = _patch_boundary_curve(dome, layout, patch_index)[2]
        axis_xyz = _patch_axis_curve(dome, layout, patch_index)

        ax.plot(
            boundary_xyz[:, 0],
            boundary_xyz[:, 1],
            boundary_xyz[:, 2],
            color=palette[patch_index],
            linewidth=2.6,
            zorder=6,
        )
        ax.plot(
            axis_xyz[:, 0],
            axis_xyz[:, 1],
            axis_xyz[:, 2],
            color=palette[patch_index],
            linewidth=1.4,
            alpha=0.9,
            zorder=7,
        )
        ax.scatter(
            point[0],
            point[1],
            point[2],
            s=78,
            color=palette[patch_index],
            edgecolors="white",
            linewidths=1.0,
            depthshade=False,
            zorder=8,
        )
        ax.text(
            point[0],
            point[1],
            point[2] + 0.012,
            f"P{patch_index + 1}",
            fontsize=9,
            color="#111827",
            zorder=9,
        )

    _set_equal_3d_axes(ax, xyz)
    ax.set_box_aspect((1.0, 1.0, 0.38))
    ax.view_init(elev=27, azim=-58)
    ax.text2D(
        0.5,
        0.92,
        title,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12.5,
        weight="bold",
        color="#111827",
    )
    ax.set_axis_off()


def _draw_patch_movement_map(
    ax: plt.Axes,
    dome: object,
    seed_layout: dict[str, object],
    optimized_layout: dict[str, object],
    optimized_thickness: dict[str, object],
    palette: list[str],
    summary: dict[str, float],
) -> None:
    extent = _field_extent(dome)
    patch_plies = _as_numpy(optimized_thickness["patch_plies_map"])
    ax.set_facecolor("#fcfaf6")
    ax.imshow(
        patch_plies,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
        alpha=0.14,
        vmin=0.0,
        vmax=max(float(np.max(patch_plies)), 1.0),
    )

    for patch_index, color in enumerate(palette):
        seed_theta, seed_phi, _ = _patch_boundary_curve(dome, seed_layout, patch_index)
        opt_theta, opt_phi, _ = _patch_boundary_curve(dome, optimized_layout, patch_index)
        seed_phi_plot, seed_theta_plot = _wrap_curve_for_plot(seed_phi, seed_theta)
        opt_phi_plot, opt_theta_plot = _wrap_curve_for_plot(opt_phi, opt_theta)

        if not np.isnan(opt_phi_plot).any():
            ax.fill(opt_phi_plot, opt_theta_plot, color=color, alpha=0.16, linewidth=0)

        ax.plot(seed_phi_plot, seed_theta_plot, color=color, linewidth=1.6, linestyle=(0, (4, 3)), alpha=0.7)
        ax.plot(opt_phi_plot, opt_theta_plot, color=color, linewidth=2.4)

        seed_phi_deg = float(np.rad2deg(seed_layout["center_phi"][patch_index]))
        seed_theta_deg = float(np.rad2deg(seed_layout["center_theta"][patch_index]))
        opt_phi_deg = float(np.rad2deg(optimized_layout["center_phi"][patch_index]))
        opt_theta_deg = float(np.rad2deg(optimized_layout["center_theta"][patch_index]))
        delta_phi_deg = ((opt_phi_deg - seed_phi_deg + 180.0) % 360.0) - 180.0

        ax.annotate(
            "",
            xy=(seed_phi_deg + delta_phi_deg, opt_theta_deg),
            xytext=(seed_phi_deg, seed_theta_deg),
            arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.4, shrinkA=6, shrinkB=5, alpha=0.8),
        )
        ax.scatter(seed_phi_deg, seed_theta_deg, s=52, facecolors="white", edgecolors=color, linewidths=1.5, zorder=5)
        ax.scatter(opt_phi_deg, opt_theta_deg, s=62, facecolors=color, edgecolors="white", linewidths=0.9, zorder=6)
        ax.text(opt_phi_deg + 4.0, opt_theta_deg + 0.9, f"P{patch_index + 1}", color=color, fontsize=9.5, weight="bold")

    seed_transition = float(np.rad2deg(seed_layout["transition_theta"]))
    opt_transition = float(np.rad2deg(optimized_layout["transition_theta"]))
    boss_theta = float(np.rad2deg(dome.config.theta_open))

    ax.axhline(seed_transition, color="#6b7280", linewidth=1.6, linestyle=(0, (4, 3)))
    ax.axhline(opt_transition, color="#111827", linewidth=2.0)
    ax.axhline(boss_theta, color="#b91c1c", linewidth=1.2, linestyle=":")

    patch_lines = ["Patch updates"]
    for patch_index in range(len(palette)):
        patch_lines.append(
            "P{idx}  th {th0:>4.1f}->{th1:>4.1f}  phi {ph0:>5.1f}->{ph1:>5.1f}  "
            "ang {a0:>5.1f}->{a1:>5.1f}  n {n0:>3.1f}->{n1:>3.1f}".format(
                idx=patch_index + 1,
                th0=float(np.rad2deg(seed_layout["center_theta"][patch_index])),
                th1=float(np.rad2deg(optimized_layout["center_theta"][patch_index])),
                ph0=float(np.rad2deg(seed_layout["center_phi"][patch_index])),
                ph1=float(np.rad2deg(optimized_layout["center_phi"][patch_index])),
                a0=float(np.rad2deg(seed_layout["angle_rad"][patch_index])),
                a1=float(np.rad2deg(optimized_layout["angle_rad"][patch_index])),
                n0=float(seed_layout["plies"][patch_index]),
                n1=float(optimized_layout["plies"][patch_index]),
            )
        )

    summary_lines = [
        "Outcome",
        f"Peak Tsai-Wu  {summary['baseline_peak_stress_index']:.2f} -> {summary['optimized_peak_stress_index']:.2f}",
        f"Mass          {summary['baseline_mass_kg']:.3f} kg -> {summary['optimized_mass_kg']:.3f} kg",
        f"Transition    {1000.0 * summary['transition_height_m']:.1f} mm",
    ]

    ax.text(
        0.01,
        0.98,
        "\n".join(patch_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
        color="#1f2937",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d6d3d1", alpha=0.95),
    )
    ax.text(
        0.99,
        0.98,
        "\n".join(summary_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9.2,
        color="#1f2937",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d6d3d1", alpha=0.95),
    )

    legend_handles = [
        Line2D([0], [0], color="#374151", linewidth=1.6, linestyle=(0, (4, 3)), label="Seed footprint"),
        Line2D([0], [0], color="#111827", linewidth=2.2, label="Optimized footprint"),
        Line2D([0], [0], marker="o", markersize=7, markerfacecolor="white", markeredgecolor="#111827", linestyle="None", label="Seed center"),
        Line2D([0], [0], marker="o", markersize=7, markerfacecolor="#111827", markeredgecolor="white", linestyle="None", label="Optimized center"),
        Line2D([0], [0], color="#111827", linewidth=1.6, label="Transition line"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#d6d3d1")

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(float(np.rad2deg(np.min(_as_numpy(dome.theta)))), float(np.rad2deg(np.max(_as_numpy(dome.theta)))))
    ax.set_xlabel("Circumferential angle phi (deg)")
    ax.set_ylabel("Meridional angle theta (deg)")
    ax.set_title("Patch movement on the unwrapped dome", fontsize=12.5, weight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.3)


def _draw_patch_animation_map(
    ax: plt.Axes,
    dome: object,
    seed_layout: dict[str, object],
    frames: list[dict[str, object]],
    current_index: int,
    current_thickness: dict[str, object],
    palette: list[str],
    total_steps: int,
) -> None:
    current_layout = frames[current_index]["layout"]
    current_metrics = frames[current_index]["metrics"]
    extent = _field_extent(dome)
    patch_plies = _as_numpy(current_thickness["patch_plies_map"])

    ax.set_facecolor("#fcfaf6")
    ax.imshow(
        patch_plies,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
        alpha=0.16,
        vmin=0.0,
        vmax=max(float(np.max(patch_plies)), 1.0),
    )

    for patch_index, color in enumerate(palette):
        seed_theta, seed_phi, _ = _patch_boundary_curve(dome, seed_layout, patch_index)
        current_theta, current_phi, _ = _patch_boundary_curve(dome, current_layout, patch_index)
        seed_phi_plot, seed_theta_plot = _wrap_curve_for_plot(seed_phi, seed_theta)
        current_phi_plot, current_theta_plot = _wrap_curve_for_plot(current_phi, current_theta)

        ax.plot(seed_phi_plot, seed_theta_plot, color=color, linewidth=1.3, linestyle=(0, (4, 3)), alpha=0.5)
        if not np.isnan(current_phi_plot).any():
            ax.fill(current_phi_plot, current_theta_plot, color=color, alpha=0.18, linewidth=0)
        ax.plot(current_phi_plot, current_theta_plot, color=color, linewidth=2.2)

        phi_trail = np.asarray([float(frame["layout"]["center_phi"][patch_index]) for frame in frames[: current_index + 1]])
        theta_trail = np.asarray([float(frame["layout"]["center_theta"][patch_index]) for frame in frames[: current_index + 1]])
        phi_trail_plot, theta_trail_plot = _wrap_curve_for_plot(phi_trail, theta_trail)
        ax.plot(phi_trail_plot, theta_trail_plot, color=color, linewidth=1.8, alpha=0.55)

        seed_phi_deg = float(np.rad2deg(seed_layout["center_phi"][patch_index]))
        seed_theta_deg = float(np.rad2deg(seed_layout["center_theta"][patch_index]))
        current_phi_deg = float(np.rad2deg(current_layout["center_phi"][patch_index]))
        current_theta_deg = float(np.rad2deg(current_layout["center_theta"][patch_index]))

        ax.scatter(seed_phi_deg, seed_theta_deg, s=38, facecolors="white", edgecolors=color, linewidths=1.3, zorder=5)
        ax.scatter(current_phi_deg, current_theta_deg, s=58, facecolors=color, edgecolors="white", linewidths=0.8, zorder=6)
        ax.text(current_phi_deg + 3.0, current_theta_deg + 0.9, f"P{patch_index + 1}", color=color, fontsize=9.0, weight="bold")

    seed_transition = float(np.rad2deg(seed_layout["transition_theta"]))
    current_transition = float(np.rad2deg(current_layout["transition_theta"]))
    ax.axhline(seed_transition, color="#6b7280", linewidth=1.4, linestyle=(0, (4, 3)))
    ax.axhline(current_transition, color="#111827", linewidth=1.9)

    info_lines = [
        f"Step {int(frames[current_index]['step'])}/{total_steps}",
        f"Peak Tsai-Wu  {float(current_metrics['peak_stress_index']):.2f}",
        f"Mass          {float(current_metrics['total_mass_kg']):.3f} kg",
        f"Transition    {1000.0 * float(current_metrics['transition_height_m']):.1f} mm",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9.0,
        color="#1f2937",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d6d3d1", alpha=0.95),
    )

    legend_handles = [
        Line2D([0], [0], color="#374151", linewidth=1.3, linestyle=(0, (4, 3)), label="Seed footprint"),
        Line2D([0], [0], color="#111827", linewidth=2.0, label="Current footprint"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="white", markeredgecolor="#111827", linestyle="None", label="Seed center"),
        Line2D([0], [0], marker="o", markersize=6.5, markerfacecolor="#111827", markeredgecolor="white", linestyle="None", label="Current center"),
        Line2D([0], [0], color="#6b7280", linewidth=1.6, label="Center trail"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, facecolor="white", edgecolor="#d6d3d1")

    ax.set_xlim(0.0, 360.0)
    ax.set_ylim(float(np.rad2deg(np.min(_as_numpy(dome.theta)))), float(np.rad2deg(np.max(_as_numpy(dome.theta)))))
    ax.set_xlabel("Circumferential angle phi (deg)")
    ax.set_ylabel("Meridional angle theta (deg)")
    ax.set_title("Patch placement during optimization", fontsize=12.0, weight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.28)


def _draw_convergence_frame(ax: plt.Axes, frames: list[dict[str, object]], current_index: int) -> None:
    steps = np.asarray([float(frame["step"]) for frame in frames], dtype=float)
    loss = np.asarray([float(frame["metrics"]["loss"]) for frame in frames], dtype=float)
    peak = np.asarray([float(frame["metrics"]["peak_stress_index"]) for frame in frames], dtype=float)
    mass = np.asarray([float(frame["metrics"]["total_mass_kg"]) for frame in frames], dtype=float)

    loss_norm = loss / max(loss[0], 1.0e-8)
    peak_norm = peak / max(peak[0], 1.0e-8)
    mass_norm = mass / max(mass[0], 1.0e-8)

    curve_specs = [
        ("Loss / seed", loss_norm, "#111827"),
        ("Peak Tsai-Wu / seed", peak_norm, "#be123c"),
        ("Mass / seed", mass_norm, "#0f766e"),
    ]

    for label, values, color in curve_specs:
        ax.plot(steps, values, color=color, linewidth=1.8, alpha=0.14)
        ax.plot(steps[: current_index + 1], values[: current_index + 1], color=color, linewidth=2.5, label=label)
        ax.scatter(steps[current_index], values[current_index], s=34, color=color, edgecolors="white", linewidths=0.7, zorder=4)

    ax.axvline(steps[current_index], color="#6b7280", linewidth=1.3, linestyle=(0, (4, 3)))
    ax.set_xlim(float(steps[0]), float(steps[-1]))
    ax.set_ylim(0.0, 1.05 * max(2.3, float(np.max(mass_norm)), float(np.max(loss_norm)), float(np.max(peak_norm))))
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Normalized value")
    ax.set_title("Convergence", fontsize=12.0, weight="bold")
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=9.0)

    best_peak = float(np.min(peak[: current_index + 1]))
    best_loss = float(np.min(loss[: current_index + 1]))
    info_lines = [
        f"Current step   {int(steps[current_index])}",
        f"Best loss      {best_loss:.3e}",
        f"Best peak FI   {best_peak:.2f}",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.9,
        color="#1f2937",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d6d3d1", alpha=0.95),
    )


def _add_resultant_quivers(ax: plt.Axes, result: dict[str, object], stride: int = 5) -> None:
    dome = result["dome"]
    xyz = _as_numpy(dome.xyz)
    t_theta = _as_numpy(dome.tangent_theta)
    t_phi = _as_numpy(dome.tangent_phi)
    N_phi = _as_numpy(result["optimized"]["structure"]["N_phi_field"])
    N_theta = _as_numpy(result["optimized"]["structure"]["N_theta_field"])

    max_resultant = max(float(np.max(np.abs(N_phi))), float(np.max(np.abs(N_theta))), 1.0e-8)
    arrow_scale = 0.04 / max_resultant

    n_theta, n_phi, _ = xyz.shape
    for it in range(0, n_theta, stride):
        for ip in range(0, n_phi, stride):
            point = xyz[it, ip]
            meridional = t_theta[it, ip]
            circumferential = t_phi[it, ip]

            meridional = meridional / max(np.linalg.norm(meridional), 1.0e-8)
            circumferential = circumferential / max(np.linalg.norm(circumferential), 1.0e-8)

            mer_arrow = meridional * abs(float(N_phi[it, ip])) * arrow_scale
            cir_arrow = circumferential * abs(float(N_theta[it, ip])) * arrow_scale

            ax.quiver(
                point[0],
                point[1],
                point[2],
                mer_arrow[0],
                mer_arrow[1],
                mer_arrow[2],
                color="#88ccff",
                linewidth=0.55,
                arrow_length_ratio=0.28,
                alpha=0.42,
            )
            ax.quiver(
                point[0],
                point[1],
                point[2],
                cir_arrow[0],
                cir_arrow[1],
                cir_arrow[2],
                color="#ffcc44",
                linewidth=0.55,
                arrow_length_ratio=0.28,
                alpha=0.42,
            )


def create_overview_figure(result: dict[str, object], summary: dict[str, float], output_path: Path) -> None:
    dome = result["dome"]
    optimized = result["optimized"]
    material = MaterialConfig(**result["material_config"])
    optimization_config = OptimizationConfig(**result["optimization_config"])
    seed_layout = decode_layout(initial_raw_layout(dome, optimization_config), dome, optimization_config)
    seed_thickness = evaluate_thickness_state(dome, seed_layout, material, optimization_config)
    optimized_layout = optimized["layout"]
    optimized_thickness = optimized["thickness"]
    palette = _patch_palette(len(_as_numpy(optimized_layout["plies"])))

    fig = plt.figure(figsize=(15.5, 9.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 0.95])

    ax_seed = fig.add_subplot(grid[0, 0], projection="3d")
    ax_optimized = fig.add_subplot(grid[0, 1], projection="3d")
    ax_map = fig.add_subplot(grid[1, :])

    _draw_patch_scene(ax_seed, dome, seed_layout, seed_thickness, "Seed layout before optimization", palette)
    _draw_patch_scene(ax_optimized, dome, optimized_layout, optimized_thickness, "Optimized layout on the dome", palette)
    _draw_patch_movement_map(ax_map, dome, seed_layout, optimized_layout, optimized_thickness, palette, summary)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_optimization_animation(result: dict[str, object], output_path: Path) -> None:
    dome = result["dome"]
    material = MaterialConfig(**result["material_config"])
    optimization_config = OptimizationConfig(**result["optimization_config"])
    serialized_frames = result["frames"]
    frames = [
        {
            "step": float(frame["step"]),
            "layout": _layout_from_serialized(frame["layout"]),
            "metrics": {key: float(value) for key, value in frame["metrics"].items()},
        }
        for frame in serialized_frames
    ]

    seed_layout = frames[0]["layout"]
    palette = _patch_palette(len(_as_numpy(seed_layout["plies"])))
    images: list[Image.Image] = []

    for current_index, frame in enumerate(frames):
        current_layout = frame["layout"]
        current_thickness = evaluate_thickness_state(dome, current_layout, material, optimization_config)

        fig = plt.figure(figsize=(11.8, 6.7), layout="constrained")
        grid = fig.add_gridspec(2, 2, width_ratios=[1.24, 1.0], height_ratios=[1.0, 0.88])

        ax_dome = fig.add_subplot(grid[:, 0], projection="3d")
        ax_map = fig.add_subplot(grid[0, 1])
        ax_conv = fig.add_subplot(grid[1, 1])

        _draw_patch_scene(
            ax_dome,
            dome,
            current_layout,
            current_thickness,
            f"Patch layout on the dome  |  step {int(frame['step'])}/{optimization_config.steps}",
            palette,
        )
        ax_dome.text2D(
            0.04,
            0.06,
            "\n".join(
                [
                    f"peak FI  {float(frame['metrics']['peak_stress_index']):.2f}",
                    f"mass     {float(frame['metrics']['total_mass_kg']):.3f} kg",
                    f"patches  {len(palette)}",
                ]
            ),
            transform=ax_dome.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            color="#1f2937",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d6d3d1", alpha=0.94),
        )

        _draw_patch_animation_map(ax_map, dome, seed_layout, frames, current_index, current_thickness, palette, optimization_config.steps)
        _draw_convergence_frame(ax_conv, frames, current_index)

        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=112, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buffer.seek(0)
        frame_image = Image.open(buffer)
        images.append(frame_image.convert("P", palette=Image.ADAPTIVE).copy())
        frame_image.close()
        buffer.close()

    if not images:
        return

    hold_frames = [images[-1].copy() for _ in range(6)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:] + hold_frames,
        duration=[140] * max(len(images) - 1, 0) + [220] + [250] * len(hold_frames),
        loop=0,
        disposal=2,
    )


def create_field_comparison_figure(result: dict[str, object], output_path: Path) -> None:
    dome = result["dome"]
    baseline = result["baseline"]
    optimized = result["optimized"]
    extent = _field_extent(dome)

    field_specs = [
        (
            "Tsai-Wu failure index",
            "Laminate failure envelope on the dome surface",
            _as_numpy(baseline["structure"]["stress_index"]),
            _as_numpy(optimized["structure"]["stress_index"]),
            "RdYlGn_r",
            "Tsai-Wu FI",
        ),
        (
            "Wrinkle risk",
            "Combined shear / areal distortion indicator",
            _as_numpy(baseline["kinematics"]["wrinkle_risk_map"]),
            _as_numpy(optimized["kinematics"]["wrinkle_risk_map"]),
            "cividis",
            "Risk index",
        ),
        (
            "Thickness gradient",
            "Gradient of the resulting laminate thickness field",
            _as_numpy(baseline["thickness"]["thickness_gradient_mm_per_m"]),
            _as_numpy(optimized["thickness"]["thickness_gradient_mm_per_m"]),
            "viridis",
            "mm / m",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), layout="constrained")
    row_labels = ["Helical-only baseline", "Optimized hybrid layout"]

    for col, (title, subtitle, baseline_field, optimized_field, cmap, cbar_label) in enumerate(field_specs):
        if col == 0:
            vmin = 0.0
            vmax = max(1.0, float(np.max(baseline_field)), float(np.max(optimized_field)))
        else:
            vmin = min(float(np.min(baseline_field)), float(np.min(optimized_field)))
            vmax = max(float(np.max(baseline_field)), float(np.max(optimized_field)))
        images = []
        for row, field in enumerate((baseline_field, optimized_field)):
            ax = axes[row, col]
            image = ax.imshow(
                field,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            images.append(image)
            if row == 0:
                ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, weight="bold")
            ax.text(
                0.02,
                0.96,
                f"max = {np.max(field):.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.35, edgecolor="none"),
            )
            ax.set_xlabel("Circumferential angle phi (deg)")
            if col == 0:
                ax.set_ylabel(f"{row_labels[row]}\nMeridional angle theta (deg)")
            else:
                ax.set_ylabel("")
        fig.colorbar(images[-1], ax=axes[:, col], shrink=0.84, label=cbar_label)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_tsai_wu_comparison_figure(result: dict[str, object], output_path: Path) -> None:
    dome = result["dome"]
    baseline = result["baseline"]
    optimized = result["optimized"]
    extent = _field_extent(dome)

    fi_baseline = _as_numpy(baseline["structure"]["stress_index"])
    fi_optimized = _as_numpy(optimized["structure"]["stress_index"])
    vmax = max(1.0, float(np.max(fi_baseline)), float(np.max(fi_optimized)))

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8), layout="constrained")
    fig.patch.set_facecolor("#0e1117")

    for ax, field, title in (
        (axes[0], fi_baseline, "Helical-only baseline"),
        (axes[1], fi_optimized, "Optimized hybrid layout"),
    ):
        ax.set_facecolor("#0e1117")
        image = ax.imshow(
            field,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="RdYlGn_r",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(title, color="#f0f0f0", fontsize=12, weight="bold")
        ax.set_xlabel("Circumferential angle phi (deg)", color="#b8c0cc")
        ax.set_ylabel("Meridional angle theta (deg)", color="#b8c0cc")
        ax.tick_params(colors="#b8c0cc")
        ax.text(
            0.02,
            0.96,
            f"peak FI = {np.max(field):.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color="white",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="black", alpha=0.35, edgecolor="none"),
        )

    fig.suptitle(
        "Tsai-Wu failure index comparison (green = safe, red = near failure, 1 = failure)",
        color="#f0f0f0",
        fontsize=13,
    )
    colorbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="Tsai-Wu FI")
    colorbar.ax.yaxis.label.set_color("#f0f0f0")
    colorbar.ax.tick_params(colors="#b8c0cc")

    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="#0e1117")
    plt.close(fig)


def create_convergence_figure(result: dict[str, object], output_path: Path) -> None:
    history = result["history"]
    steps = np.asarray([entry["step"] for entry in history], dtype=float)
    loss = np.asarray([entry["loss"] for entry in history], dtype=float)
    structural = np.asarray([entry["structural_loss"] for entry in history], dtype=float)
    kinematic = np.asarray([entry["kinematic_penalty"] for entry in history], dtype=float)
    thickness = np.asarray([entry["thickness_penalty"] for entry in history], dtype=float)
    mass = np.asarray([entry["total_mass_kg"] for entry in history], dtype=float)
    peak_stress = np.asarray([entry["peak_stress_index"] for entry in history], dtype=float)
    max_shear = np.asarray([entry["max_shear"] for entry in history], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), layout="constrained")

    axes[0].semilogy(steps, loss, linewidth=2.4, color="#111827", label="Total loss")
    axes[0].semilogy(steps, structural, linewidth=1.8, color="#0f766e", label="Structural term")
    axes[0].semilogy(steps, kinematic, linewidth=1.8, color="#1d4ed8", label="Kinematic term")
    axes[0].semilogy(steps, thickness, linewidth=1.8, color="#d97706", label="Thickness term")
    axes[0].set_title("Objective terms over the optimization run", fontsize=12.5, weight="bold")
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Value (log scale)")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].plot(steps, peak_stress / peak_stress[0], linewidth=2.2, color="#be123c", label="Peak Tsai-Wu / initial")
    axes[1].plot(steps, max_shear / max_shear[0], linewidth=2.2, color="#2563eb", label="Max shear / initial")
    axes[1].plot(steps, mass / mass[0], linewidth=2.2, color="#7c3aed", label="Mass / initial")
    axes[1].set_title("Normalized response metrics", fontsize=12.5, weight="bold")
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("Relative value")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=False)

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    assets_dir = REPO_ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    result = optimize_patch_layout()
    summary = summarize_result(result)

    create_optimization_animation(result, assets_dir / "optimization_evolution.gif")
    create_overview_figure(result, summary, assets_dir / "optimized_dome_overview.png")
    create_field_comparison_figure(result, assets_dir / "field_comparison.png")
    create_tsai_wu_comparison_figure(result, assets_dir / "tsai_wu_comparison.png")
    create_convergence_figure(result, assets_dir / "optimization_convergence.png")


if __name__ == "__main__":
    main()
