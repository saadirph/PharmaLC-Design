# PharmaLC-Design

**Anchor-Free Machine Learning for Transferable Retention Prediction and Experimental Optimization of Pharmaceutical Reversed-Phase Liquid Chromatography**

## Overview

`PharmaLC-Design` implements the four-module anchor-free RP-LC method-development workflow described in the manuscript:
1. **`pharmalc.curation`**: Salt/counter-ion stripping, 14-character InChIKey parent merging, LSS elution-composition ($\varphi_e$) calculation, and 0–1 provenance scoring.
2. **`pharmalc.descriptors`**: 213 RDKit 2D descriptors, SMARTS-based Henderson–Hasselbalch ionization/$\log D$, and Bemis–Murcko scaffold extraction.
3. **`pharmalc.models`**: SIELC condition-selection Random Forests, LightGBM/MLP/RF/Ridge $\varphi_e$ regressors, 20-PC PCA Euclidean $k$NN applicability-domain distance (`AD > 5.5` flag), and Mondrian split-conformal intervals.
4. **`pharmalc.scouting`**: Anchor-free inverse-gradient $t_R(\varphi_e)$ projection with dwell-volume ($V_D$) correction, LSS isocratic projection, multi-system C18 ranking, and ICH Q14 robustness checks.

## Installation

```bash
pip install -e ".[app,dev]"
```

## Quick Start

### 1. Command-Line Interface (CLI)
```bash
# Rank candidate C18 systems for Panel B (verifies ACN -> MeOH selectivity flip)
pharmalc scout --drugs Bromazepam Methylphenidate Alprazolam

# Recommend stationary phase & additive from SMILES
pharmalc recommend --smiles "O=C1CN=C(c2ccccn2)c2cc(Br)ccc2N1" --ph 2.7

# Project anchor-free phi_e to retention time with measured dwell volume
pharmalc project --phi-e 39.1 --tg 21.1 --t0 0.42 --vd 0.11 --flow 0.40
```

### 2. Interactive Streamlit Web App
```bash
streamlit run app.py
```

### 3. Run Unit Tests
```bash
pytest -v
```

### 4. Full-Repository LPO & Scaffold Cross-Validation
```bash
python scripts/kb2_splits.py --data Code_and_data/ --splits lpo scaffold --out splits_out/
```

## License
MIT License.
