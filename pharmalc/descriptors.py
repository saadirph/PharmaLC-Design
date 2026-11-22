"""Module 2: Molecular (RDKit 2D, Ionization, Scaffolds) & Chromatographic System Descriptors."""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ACID_SMARTS = Chem.MolFromSmarts(
    "[$([CX3](=O)[OX2H1]),$([SX4](=O)(=O)[OX2H1]),$([PX4](=O)([OX2H1])[OX2H1]),$(c[OX2H1]),$([BX3]([OX2H1])[OX2H1])]"
)
BASE_SMARTS = Chem.MolFromSmarts(
    "[$([NX3;H2,H1;!$(NC=O);!$(NS=O)]),$([NX3;H0;!$(NC=O);!$(NS=O);!$([nX3])]),$([nX2])]"
)


def compute_murcko_scaffold(smiles: str) -> str:
    """Return stereo-free Bemis-Murcko scaffold SMILES (or ACYCLIC_<smiles> if acyclic)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "INVALID"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return scaf if scaf else f"ACYCLIC_{Chem.MolToSmiles(mol, canonical=True)}"


def estimate_ionization_and_logd(smiles: str, pH: float = 2.7) -> Dict[str, float]:
    """Estimate ionizable acid/base counts, fractional ionization, and Henderson-Hasselbalch logD."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    logp = float(Crippen.MolLogP(mol))
    n_acid = len(mol.GetSubstructMatches(ACID_SMARTS)) if ACID_SMARTS else 0
    n_base = len(mol.GetSubstructMatches(BASE_SMARTS)) if BASE_SMARTS else 0

    # Representative small-molecule pKa approximations (carboxylic/phenolic ~4.5; aliphatic/aromatic amine ~8.5)
    pka_acid = 4.5 if n_acid > 0 else 14.0
    pka_base = 8.5 if n_base > 0 else 0.0

    f_deprot = 1.0 / (1.0 + 10.0 ** (pka_acid - pH)) if n_acid > 0 else 0.0
    f_prot = 1.0 / (1.0 + 10.0 ** (pH - pka_base)) if n_base > 0 else 0.0
    neutral_frac = max(1e-4, (1.0 - f_deprot) * (1.0 - f_prot))
    logd = logp + float(np.log10(neutral_frac))
    return {
        "Crippen_logP": round(logp, 3),
        "n_acid": n_acid,
        "n_base": n_base,
        "f_deprotonated": round(f_deprot, 4),
        "f_protonated": round(f_prot, 4),
        "logD_pH": round(logd, 3),
    }


def compute_core_condition_descriptors(smiles: str) -> Dict[str, float]:
    """Compute the 11 core physicochemical descriptors used in SIELC condition-selection models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "logP": float(Crippen.MolLogP(mol)),
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "MW": float(Descriptors.MolWt(mol)),
        "HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
        "HBA": float(rdMolDescriptors.CalcNumHBA(mol)),
        "aromRings": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "rotB": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "acid": float(len(mol.GetSubstructMatches(ACID_SMARTS)) if ACID_SMARTS else 0),
        "base": float(len(mol.GetSubstructMatches(BASE_SMARTS)) if BASE_SMARTS else 0),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "MolMR": float(Crippen.MolMR(mol)),
    }


def compute_molecular_descriptors(smiles_list: List[str]) -> pd.DataFrame:
    """Compute the full RDKit 2D descriptor table for a list of SMILES."""
    calc_fns = list(Descriptors._descList)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append({name: np.nan for name, _ in calc_fns})
            continue
        row = {}
        for name, fn in calc_fns:
            try:
                val = float(fn(mol))
                row[name] = val if np.isfinite(val) else np.nan
            except Exception:
                row[name] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
