"""Module 1: Literature Retention Curation, Salt Harmonization, LSS phi_e, and Provenance Audit."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def strip_salts_and_canonicalize(smiles: str) -> Optional[str]:
    """Strip inorganic salts/counter-ions and return canonical parent SMILES."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    lfc = rdMolStandardize.LargestFragmentChooser()
    parent = lfc.choose(mol)
    return Chem.MolToSmiles(parent, canonical=True)


def compute_inchikey14(smiles: str) -> Optional[str]:
    """Return the 14-character InChIKey connectivity layer for leak-free parent grouping."""
    parent_smi = strip_salts_and_canonicalize(smiles)
    if parent_smi is None:
        return None
    mol = Chem.MolFromSmiles(parent_smi)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)[:14]


def compute_phi_e(
    t_r: float,
    t0: float,
    gradient_points: List[Tuple[float, float]],
    t_d: float = 0.0,
) -> float:
    """Compute anchor-free elution organic fraction phi_e = phi(t_R - t0 - t_D).

    Parameters
    ----------
    t_r : float
        Measured retention time (min).
    t0 : float
        Column dead time (min).
    gradient_points : list of (time_min, phi_percent)
        Piecewise-linear gradient table, e.g. [(0.0, 5.0), (13.7, 95.0)].
    t_d : float, default 0.0
        System gradient dwell time V_D / F (min).
    """
    pts = sorted(gradient_points, key=lambda x: x[0])
    times = np.array([p[0] for p in pts], dtype=float)
    phis = np.array([p[1] for p in pts], dtype=float)
    t_eff = max(0.0, float(t_r) - float(t0) - float(t_d))
    return float(np.interp(t_eff, times, phis))


def compute_provenance_score(
    has_gradient_core: bool = True,
    has_column_usp: bool = True,
    has_hsm: bool = False,
    has_tanaka: bool = False,
    has_temperature: bool = False,
    has_article_doi: bool = False,
) -> Tuple[float, str]:
    """Compute quantitative 0-1 provenance score and quality tier (Tier 1 / Tier 2 / Tier 3)."""
    if not has_gradient_core:
        return 0.0, "Tier 3"
    score = 0.30
    if has_column_usp:
        score += 0.15
    if has_hsm:
        score += 0.10
    if has_tanaka:
        score += 0.10
    if has_temperature:
        score += 0.15
    score += 0.20 if has_article_doi else 0.10
    tier = "Tier 1" if has_hsm else "Tier 2"
    return round(score, 2), tier


def curate_retention_table(
    df: pd.DataFrame,
    smiles_col: str = "smiles.std",
    tr_col: str = "rt",
    t0_col: str = "t0",
    tgrad_end_col: str = "tG",
    system_col: str = "dataset",
    void_multiplier: float = 1.2,
    max_per_system: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply deterministic PharmaLC-Design curation filters and deduplication."""
    out = df.copy()
    out["parent_smiles"] = out[smiles_col].map(strip_salts_and_canonicalize)
    out = out.dropna(subset=["parent_smiles"]).copy()
    out["ik14"] = out["parent_smiles"].map(compute_inchikey14)

    if tr_col in out.columns and t0_col in out.columns:
        out = out[out[tr_col] > void_multiplier * out[t0_col]]
    if tr_col in out.columns and tgrad_end_col in out.columns:
        out = out[out[tr_col] <= out[tgrad_end_col]]

    # Collapse exact within-system duplicates to median
    group_cols = [system_col, "ik14"]
    num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    agg = {c: "median" for c in num_cols if c not in group_cols}
    for c in out.columns:
        if c not in agg and c not in group_cols:
            agg[c] = "first"
    out = out.groupby(group_cols, as_index=False).agg(agg)

    # Cap large single-platform libraries
    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in out.groupby(system_col):
        if len(grp) > max_per_system:
            idx = rng.choice(grp.index.values, size=max_per_system, replace=False)
            parts.append(grp.loc[idx])
        else:
            parts.append(grp)
    return pd.concat(parts, ignore_index=True) if parts else out
