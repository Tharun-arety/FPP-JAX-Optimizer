from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fpp_jax_optimizer import boss_opening_curve, optimize_patch_layout, summarize_result, surface_xyz


def _as_numpy(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


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
        f"Peak stress index: {baseline_peak:.3f} -> {optimized_peak:.3f} ({_percent_change(baseline_peak, optimized_peak):+.1f}%)",
        f"Total mass: {baseline_mass:.4f} kg -> {optimized_mass:.4f} kg ({_percent_change(baseline_mass, optimized_mass):+.1f}%)",
        f"Patch reinforcement added: {patch_mass_g:.1f} g",
        f"Cost saving vs all-FPP laminate: {summary['cost_savings_vs_all_fpp_pct']:.1f}%",
        f"Transition height: {1000.0 * summary['transition_height_m']:.1f} mm",
    ]
    return result


def create_overview_figure(result: dict[str, object], summary: dict[str, float], output_path: Path) -> None:
    dome = result["dome"]
    optimized = result["optimized"]
    layout = result["layout"]

    xyz = _as_numpy(dome.xyz)
    total_plies = _as_numpy(optimized["thickness"]["total_plies"])
    centers = _as_numpy(surface_xyz(dome.config, layout["center_theta"], layout["center_phi"]))
    boss_curve = boss_opening_curve(dome)
    phi = np.linspace(0.0, 2.0 * math.pi, 240, endpoint=True)
    transition_curve = _as_numpy(
        surface_xyz(dome.config, np.full_like(phi, float(layout["transition_theta"])), phi)
    )

    norm = colors.Normalize(vmin=float(np.min(total_plies)), vmax=float(np.max(total_plies)))
    facecolors = cm.viridis(norm(total_plies))

    fig = plt.figure(figsize=(13.5, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])

    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax3d.plot_surface(
        xyz[:, :, 0],
        xyz[:, :, 1],
        xyz[:, :, 2],
        rstride=1,
        cstride=1,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax3d.plot(boss_curve[:, 0], boss_curve[:, 1], boss_curve[:, 2], color="#ff6f61", linewidth=2.4)
    ax3d.plot(
        transition_curve[:, 0],
        transition_curve[:, 1],
        transition_curve[:, 2],
        color="#101010",
        linewidth=2.0,
        linestyle="--",
    )
    ax3d.scatter(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        s=70,
        c="#ffd54f",
        edgecolors="#111111",
        linewidths=0.8,
        depthshade=False,
    )
    for idx, point in enumerate(centers, start=1):
        ax3d.text(point[0], point[1], point[2] + 0.01, f"P{idx}", fontsize=9, color="#111111")
    _set_equal_3d_axes(ax3d, xyz)
    ax3d.view_init(elev=26, azim=-60)
    ax3d.set_title("Optimized hybrid FPP layout on the Type IV dome", fontsize=14, weight="bold", pad=16)
    ax3d.set_axis_off()
    fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=ax3d,
        fraction=0.035,
        pad=0.02,
        label="Total plies",
    )

    ax_text = fig.add_subplot(grid[0, 1])
    ax_text.axis("off")
    ax_text.set_title("Default optimization run", fontsize=13, weight="bold", loc="left")
    ax_text.text(
        0.0,
        0.98,
        "The optimizer moves patch centers, orientation, size, ply count,\n"
        "and the helical-to-FPP transition boundary in one differentiable loop.",
        va="top",
        ha="left",
        fontsize=10.5,
        color="#202020",
        linespacing=1.45,
    )
    ax_text.text(
        0.0,
        0.66,
        "\n".join(_metric_rows(summary)),
        va="top",
        ha="left",
        fontsize=10.5,
        color="#202020",
        linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f4f6f8", edgecolor="#d0d7de"),
    )

    ax_delta = fig.add_subplot(grid[1, 1])
    labels = [
        "Peak stress",
        "Max shear",
        "Areal distortion",
        "Thickness gradient",
        "Mass",
    ]
    baseline_metrics = result["baseline"]["metrics"]
    optimized_metrics = result["optimized"]["metrics"]
    deltas = np.asarray(
        [
            _percent_change(float(baseline_metrics["peak_stress_index"]), float(optimized_metrics["peak_stress_index"])),
            _percent_change(float(baseline_metrics["max_shear"]), float(optimized_metrics["max_shear"])),
            _percent_change(
                float(baseline_metrics["max_areal_distortion"]),
                float(optimized_metrics["max_areal_distortion"]),
            ),
            _percent_change(
                float(baseline_metrics["max_thickness_gradient_mm_per_m"]),
                float(optimized_metrics["max_thickness_gradient_mm_per_m"]),
            ),
            _percent_change(float(baseline_metrics["total_mass_kg"]), float(optimized_metrics["total_mass_kg"])),
        ],
        dtype=float,
    )
    colors_bar = ["#0f766e" if value <= 0.0 else "#c2410c" for value in deltas]
    y_pos = np.arange(len(labels))
    ax_delta.barh(y_pos, deltas, color=colors_bar, alpha=0.92)
    ax_delta.axvline(0.0, color="#6b7280", linewidth=1.0)
    ax_delta.set_yticks(y_pos, labels)
    ax_delta.invert_yaxis()
    ax_delta.set_xlabel("Relative change vs helical-only baseline (%)")
    ax_delta.set_title("What the optimizer improves, and what it trades off", fontsize=12.5, weight="bold")
    ax_delta.grid(axis="x", linestyle="--", alpha=0.35)
    for idx, value in enumerate(deltas):
        text_x = value + (1.0 if value >= 0.0 else -1.0)
        ax_delta.text(
            text_x,
            idx,
            f"{value:+.1f}%",
            va="center",
            ha="left" if value >= 0.0 else "right",
            fontsize=9.5,
            color="#111111",
        )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_field_comparison_figure(result: dict[str, object], output_path: Path) -> None:
    dome = result["dome"]
    baseline = result["baseline"]
    optimized = result["optimized"]
    extent = _field_extent(dome)

    field_specs = [
        (
            "Stress proxy",
            "Peak stress concentration on the dome surface",
            _as_numpy(baseline["structure"]["stress_index"]),
            _as_numpy(optimized["structure"]["stress_index"]),
            "magma",
            "Stress index",
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

    axes[1].plot(steps, peak_stress / peak_stress[0], linewidth=2.2, color="#be123c", label="Peak stress / initial")
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

    create_overview_figure(result, summary, assets_dir / "optimized_dome_overview.png")
    create_field_comparison_figure(result, assets_dir / "field_comparison.png")
    create_convergence_figure(result, assets_dir / "optimization_convergence.png")


if __name__ == "__main__":
    main()
