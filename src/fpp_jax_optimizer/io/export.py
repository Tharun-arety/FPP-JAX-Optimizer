from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExportConfig, MaterialConfig


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    optimized = result["optimized"]
    baseline = result["baseline"]
    metrics = optimized["metrics"]
    baseline_metrics = baseline["metrics"]
    baseline_mass = float(baseline_metrics["total_mass_kg"])
    optimized_mass = float(metrics["total_mass_kg"])
    baseline_helical_mass = float(baseline_metrics["helical_mass_kg"])
    optimized_helical_mass = float(metrics["helical_mass_kg"])
    return {
        "baseline_peak_stress_index": float(baseline_metrics["peak_stress_index"]),
        "optimized_peak_stress_index": float(metrics["peak_stress_index"]),
        "baseline_mass_kg": baseline_mass,
        "optimized_mass_kg": optimized_mass,
        "added_total_mass_kg": optimized_mass - baseline_mass,
        "patch_mass_kg": float(metrics["patch_mass_kg"]),
        "baseline_helical_mass_kg": baseline_helical_mass,
        "helical_mass_kg": optimized_helical_mass,
        "added_helical_mass_kg": optimized_helical_mass - baseline_helical_mass,
        "cost_savings_vs_all_fpp_pct": float(metrics["cost_savings_vs_all_fpp_pct"]),
        "max_shear": float(metrics["max_shear"]),
        "max_areal_distortion": float(metrics["max_areal_distortion"]),
        "max_thickness_gradient_mm_per_m": float(metrics["max_thickness_gradient_mm_per_m"]),
        "transition_height_m": float(metrics["transition_height_m"]),
        "patch_layout": result["layout_serialized"],
        "history": result["history"],
    }


def write_summary_json(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_result(result), indent=2), encoding="utf-8")
    return path


def _bdf_number(value: float) -> str:
    return f"{value:.6E}"


def write_nastran_bdf(
    result: dict[str, Any],
    path: str | Path,
    material: MaterialConfig | None = None,
) -> Path:
    material = material or MaterialConfig()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dome = result["dome"]
    optimized = result["optimized"]
    total_plies = np.asarray(optimized["thickness"]["total_plies"], dtype=float)
    angle_field = np.asarray(optimized["structure"]["patch_angle_field"], dtype=float)
    xyz = np.asarray(dome.xyz, dtype=float)

    n_theta, n_phi, _ = xyz.shape
    node_ids = np.arange(1, n_theta * n_phi + 1, dtype=int).reshape(n_theta, n_phi)

    prop_map: dict[tuple[int, int], int] = {}
    prop_cards: list[str] = []
    elem_cards: list[str] = []
    next_pid = 100
    next_eid = 1

    for i in range(n_theta - 1):
        for j in range(n_phi):
            n1 = int(node_ids[i, j])
            n2 = int(node_ids[i, (j + 1) % n_phi])
            n3 = int(node_ids[i + 1, (j + 1) % n_phi])
            n4 = int(node_ids[i + 1, j])

            ply_count = int(max(1, round(np.mean([total_plies[i, j], total_plies[i, (j + 1) % n_phi], total_plies[i + 1, j], total_plies[i + 1, (j + 1) % n_phi]]))))
            angle_deg = int(10 * round(np.degrees(np.mean([angle_field[i, j], angle_field[i, (j + 1) % n_phi], angle_field[i + 1, j], angle_field[i + 1, (j + 1) % n_phi]])) / 10.0))
            prop_key = (ply_count, angle_deg)

            if prop_key not in prop_map:
                pid = next_pid
                prop_map[prop_key] = pid
                prop_cards.append(f"PCOMP,{pid},,0.0")
                for _ in range(ply_count):
                    prop_cards.append(f"+,1,{_bdf_number(material.ply_thickness_mm / 1000.0)},{float(angle_deg):.2f},YES")
                next_pid += 1
            pid = prop_map[prop_key]
            elem_cards.append(f"CQUAD4,{next_eid},{pid},{n1},{n2},{n3},{n4}")
            next_eid += 1

    lines = [
        "$ FPP-JAX-Optimizer Nastran export",
        "BEGIN BULK",
        "MAT8,1,1.350000E+11,9.000000E+09,0.30,5.000000E+09,1.500000E+09,1.500000E+09",
    ]

    for nid, point in enumerate(xyz.reshape(-1, 3), start=1):
        lines.append(f"GRID,{nid},,{_bdf_number(float(point[0]))},{_bdf_number(float(point[1]))},{_bdf_number(float(point[2]))}")

    lines.extend(prop_cards)
    lines.extend(elem_cards)
    lines.append("ENDDATA")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
