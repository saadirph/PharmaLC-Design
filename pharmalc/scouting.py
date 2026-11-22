"""Module 4: Anchor-Free Inverse-Gradient t_R Projection, Critical-Pair Scouting, & ICH Q14 Robustness."""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def predict_gradient_tr(
    phi_e: float,
    phi_init: float = 5.0,
    phi_final: float = 95.0,
    t_g: float = 15.0,
    t0: float = 0.45,
    v_d: float = 0.11,
    flow: float = 0.45,
    t_hold: float = 0.0,
) -> float:
    """Anchor-free inverse-gradient retention time t_R = phi^-1(phi_e) + t0 + V_D/F."""
    t_d = float(v_d) / float(flow)
    if phi_e <= phi_init:
        return round(float(t0) + t_d, 3)
    frac = min(1.0, (float(phi_e) - float(phi_init)) / max(1e-6, float(phi_final) - float(phi_init)))
    t_r = float(t_hold) + frac * float(t_g) + float(t0) + t_d
    return round(t_r, 3)


def predict_isocratic_tr(
    phi_e: float,
    phi_iso: float = 60.0,
    t0: float = 2.45,
    v_ext: float = 0.25,
    flow: float = 1.0,
    k_star: float = 1.5,
    slope_s: float = 3.5,
) -> float:
    """LSS isocratic projection t_R = t0 * (1 + k* * 10^(S*(phi_e - phi_iso)/100)) + V_ext/F."""
    t_ext = float(v_ext) / float(flow)
    k_iso = float(k_star) * (10.0 ** (float(slope_s) * (float(phi_e) - float(phi_iso)) / 100.0))
    return round(float(t0) * (1.0 + k_iso) + t_ext, 3)


def evaluate_separation_metrics(
    compounds: List[str],
    tr_values: List[float],
    t0: float,
    w05_est: float = 0.12,
) -> Dict[str, object]:
    """Compute elution order, critical pair, min delta_tR, estimated Rs, and selectivity alpha."""
    order = np.argsort(tr_values)
    sorted_names = [compounds[i] for i in order]
    sorted_tr = [float(tr_values[i]) for i in order]

    if len(sorted_tr) < 2:
        return {
            "elution_order": sorted_names,
            "critical_pair": None,
            "min_gap_min": np.nan,
            "min_rs": np.nan,
            "min_alpha": np.nan,
        }

    gaps = np.diff(sorted_tr)
    min_idx = int(np.argmin(gaps))
    min_gap = float(gaps[min_idx])
    min_rs = 1.18 * min_gap / (2.0 * float(w05_est))
    k1 = max(1e-3, (sorted_tr[min_idx] - t0) / t0)
    k2 = max(1e-3, (sorted_tr[min_idx + 1] - t0) / t0)
    return {
        "elution_order": sorted_names,
        "critical_pair": f"{sorted_names[min_idx]} / {sorted_names[min_idx + 1]}",
        "min_gap_min": round(min_gap, 3),
        "min_rs": round(min_rs, 2),
        "min_alpha": round(k2 / k1, 3),
    }


def scout_systems(
    drug_phi_e: Dict[str, Dict[str, float]],
    systems_df: pd.DataFrame,
    w05_default: float = 0.12,
) -> pd.DataFrame:
    """Rank candidate chromatographic systems by smallest predicted critical-pair gap and Rs."""
    rows = []
    for _, sys_row in systems_df.iterrows():
        mod = sys_row["modifier"]
        names, trs = [], []
        for drug, mod_map in drug_phi_e.items():
            phi_e = mod_map.get(mod, mod_map.get("ACN", 40.0))
            if sys_row.get("mode", "gradient") == "isocratic":
                tr = predict_isocratic_tr(
                    phi_e, phi_iso=sys_row["phi_init"], t0=sys_row["t0"],
                    v_ext=sys_row["v_d"], flow=sys_row["flow"],
                )
            else:
                tr = predict_gradient_tr(
                    phi_e, phi_init=sys_row["phi_init"], phi_final=sys_row["phi_final"],
                    t_g=sys_row["t_g"], t0=sys_row["t0"], v_d=sys_row["v_d"], flow=sys_row["flow"],
                )
            names.append(drug)
            trs.append(tr)
        sep = evaluate_separation_metrics(names, trs, t0=sys_row["t0"], w05_est=w05_default)
        rows.append({
            "system_id": sys_row["system_id"],
            "column": sys_row["column"],
            "modifier": mod,
            "run_min": sys_row["t_g"],
            "min_gap_min": sep["min_gap_min"],
            "est_min_rs": sep["min_rs"],
            "critical_pair": sep["critical_pair"],
            "elution_order": " < ".join(sep["elution_order"]),
        })
    return pd.DataFrame(rows).sort_values("min_gap_min", ascending=False).reset_index(drop=True)


def lss_robustness_envelope(
    nominal_rs: float,
    tg_pct: float = 10.0,
    flow_pct: float = 5.0,
    vd_delta_ml: float = 0.05,
) -> Dict[str, float]:
    """Compute conservative worst-case LSS robustness resolution under ICH Q14 perturbations."""
    # Gradient steepness & flow/dwell perturbations reduce resolution conservatively by ~10%
    factor = max(0.5, 1.0 - 0.006 * tg_pct - 0.008 * flow_pct - 0.1 * vd_delta_ml)
    worst_rs = round(float(nominal_rs) * factor, 2)
    return {
        "nominal_rs": round(float(nominal_rs), 2),
        "worst_case_rs": worst_rs,
        "passes_baseline_1_5": worst_rs >= 1.5,
    }
