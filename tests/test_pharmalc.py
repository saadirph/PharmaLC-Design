"""Unit tests for PharmaLC-Design modules and CLI."""

import numpy as np
from pharmalc.curation import (
    compute_inchikey14,
    compute_phi_e,
    compute_provenance_score,
    strip_salts_and_canonicalize,
)
from pharmalc.data import REFERENCE_SYSTEMS, VALIDATED_DRUGS
from pharmalc.descriptors import (
    compute_core_condition_descriptors,
    compute_murcko_scaffold,
    estimate_ionization_and_logd,
)
from pharmalc.models import ConditionRecommender, MondrianSplitConformal, PCAApplicabilityDomain
from pharmalc.scouting import (
    lss_robustness_envelope,
    predict_gradient_tr,
    predict_isocratic_tr,
    scout_systems,
)


def test_salt_stripping_and_inchikey():
    # Paracetamol + NaCl salt -> should match neutral Paracetamol connectivity
    smi_salt = "CC(=O)Nc1ccc(O)cc1.[Na+].[Cl-]"
    smi_pure = "CC(=O)Nc1ccc(O)cc1"
    assert strip_salts_and_canonicalize(smi_salt) == strip_salts_and_canonicalize(smi_pure)
    assert compute_inchikey14(smi_salt) == compute_inchikey14(smi_pure)
    assert len(compute_inchikey14(smi_pure)) == 14


def test_phi_e_and_inverse_gradient_roundtrip():
    grad = [(0.0, 5.0), (21.1, 95.0)]
    tr_pred = predict_gradient_tr(phi_e=39.1, phi_init=5.0, phi_final=95.0, t_g=21.1, t0=0.42, v_d=0.11, flow=0.40)
    assert abs(tr_pred - 8.69) < 0.02
    phi_back = compute_phi_e(t_r=tr_pred, t0=0.42, gradient_points=grad, t_d=0.11 / 0.40)
    assert abs(phi_back - 39.1) < 0.1


def test_isocratic_projection_panel_c():
    tr_fav = predict_isocratic_tr(phi_e=18.2, phi_iso=60.0, t0=2.45, v_ext=0.25, flow=1.0)
    tr_rof = predict_isocratic_tr(phi_e=58.7, phi_iso=60.0, t0=2.45, v_ext=0.25, flow=1.0)
    assert abs(tr_fav - 2.83) < 0.05
    assert abs(tr_rof - 6.01) < 0.05
    assert tr_rof > tr_fav


def test_provenance_score():
    score, tier = compute_provenance_score(True, True, True, True, True, True)
    assert score == 1.0 and tier == "Tier 1"
    score2, tier2 = compute_provenance_score(True, True, False, False, False, False)
    assert tier2 == "Tier 2" and score2 == 0.55


def test_descriptors_and_recommender():
    smi = VALIDATED_DRUGS["Bromazepam"]["smiles"]
    d = compute_core_condition_descriptors(smi)
    assert len(d) == 11 and d["MW"] > 300
    ion = estimate_ionization_and_logd(smi, pH=2.7)
    assert "logD_pH" in ion
    scaf = compute_murcko_scaffold(smi)
    assert len(scaf) > 5
    rec = ConditionRecommender().recommend(smi)
    assert "recommended_phase" in rec


def test_pca_ad_and_mondrian_conformal():
    rng = np.random.default_rng(42)
    X_tr = rng.normal(0, 1, size=(120, 25))
    X_te = rng.normal(0, 1, size=(40, 25))
    ad = PCAApplicabilityDomain(n_components=10, n_neighbors=5).fit(X_tr)
    d_te = ad.distance(X_te)
    assert d_te.shape == (40,) and np.all(d_te > 0)

    y_true = rng.uniform(10, 90, size=120)
    y_pred = y_true + rng.normal(0, 5.0, size=120)
    cp = MondrianSplitConformal(alpha=0.10, n_bins=3).fit(y_true, y_pred, ad.distance(X_tr))
    lo, hi = cp.predict_interval(y_pred[:40], d_te)
    assert np.all(hi > lo)


def test_scouting_acn_meoh_inversion():
    panel_b = {k: VALIDATED_DRUGS[k] for k in ["Bromazepam", "Methylphenidate", "Alprazolam"]}
    grad_sys = REFERENCE_SYSTEMS[REFERENCE_SYSTEMS["mode"] == "gradient"]
    res = scout_systems(panel_b, grad_sys)
    order_0186 = res.loc[res["system_id"] == "0186", "elution_order"].values[0]
    order_0350 = res.loc[res["system_id"] == "0350", "elution_order"].values[0]
    assert order_0186.startswith("Bromazepam < Methylphenidate")
    assert order_0350.startswith("Methylphenidate < Bromazepam")


def test_lss_robustness():
    env = lss_robustness_envelope(1.85)
    assert env["passes_baseline_1_5"] is True
