"""Streamlit Web Application for PharmaLC-Design."""

import streamlit as st
import pandas as pd
from pharmalc.data import REFERENCE_SYSTEMS, VALIDATED_DRUGS
from pharmalc.descriptors import compute_murcko_scaffold, estimate_ionization_and_logd
from pharmalc.models import ConditionRecommender
from pharmalc.scouting import (
    lss_robustness_envelope,
    predict_gradient_tr,
    predict_isocratic_tr,
    scout_systems,
)

st.set_page_config(page_title="PharmaLC-Design", layout="wide")
st.title("PharmaLC-Design: Anchor-Free RP-LC Method Scouting")
st.caption("Companion software for Nageen & Salman (Microchemical Journal, 2026) — Zero target-column calibration.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Multi-System C18 Scouting",
    "2. Gradient / Isocratic tR Projector",
    "3. Column & Additive Recommender",
    "4. ICH Q14 Robustness Check",
])

with tab1:
    st.subheader("Rank Candidate C18 Systems by Critical-Pair Separation")
    selected = st.multiselect(
        "Select pharmaceutical panel",
        options=list(VALIDATED_DRUGS.keys()),
        default=["Bromazepam", "Methylphenidate", "Alprazolam"],
    )
    if len(selected) >= 2:
        subset = {d: VALIDATED_DRUGS[d] for d in selected}
        grad_sys = REFERENCE_SYSTEMS[REFERENCE_SYSTEMS["mode"] == "gradient"]
        ranked = scout_systems(subset, grad_sys)
        st.dataframe(ranked, use_container_width=True)
    else:
        st.info("Select at least 2 drugs to rank candidate C18 systems.")

with tab2:
    st.subheader("Anchor-Free Elution Composition (φₑ) → Retention Time (tR)")
    mode = st.radio("Elution mode", ["Linear Gradient", "Isocratic (LSS)"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    phi_e = c1.number_input("Predicted φₑ (% organic)", 0.0, 100.0, 48.3)
    t0 = c2.number_input("Column dead time t₀ (min)", 0.05, 10.0, 0.42)
    flow = c3.number_input("Flow rate F (mL/min)", 0.05, 5.0, 0.40)
    if mode == "Linear Gradient":
        c4, c5, c6, c7 = st.columns(4)
        phi_init = c4.number_input("φ_init (%)", 0.0, 90.0, 5.0)
        phi_final = c5.number_input("φ_final (%)", 10.0, 100.0, 95.0)
        t_g = c6.number_input("Gradient time t_G (min)", 1.0, 120.0, 21.1)
        v_d = c7.number_input("Dwell volume V_D (mL)", 0.0, 3.0, 0.11)
        tr = predict_gradient_tr(phi_e, phi_init, phi_final, t_g, t0, v_d, flow)
    else:
        c4, c5 = st.columns(2)
        phi_iso = c4.number_input("Isocratic organic φ_iso (%)", 10.0, 95.0, 60.0)
        v_ext = c5.number_input("Extra-column volume V_ext (mL)", 0.0, 2.0, 0.25)
        tr = predict_isocratic_tr(phi_e, phi_iso, t0, v_ext, flow)
    st.metric("Predicted Retention Time tR (min)", f"{tr:.3f} min")

with tab3:
    st.subheader("Qualitative Stationary-Phase & MS-Compatible Additive Recommender")
    smi = st.text_input("Enter canonical SMILES", value=VALIDATED_DRUGS["Bromazepam"]["smiles"])
    ph = st.slider("Eluent pH", 1.5, 10.5, 2.7, 0.1)
    if smi:
        rec = ConditionRecommender().recommend(smi)
        ion = estimate_ionization_and_logd(smi, pH=ph)
        st.write("**Recommended Stationary Phase:**", rec["recommended_phase"])
        st.write("**Suggested Additive:**", rec["suggested_additive"])
        st.write("**Bemis–Murcko Scaffold:**", f"`{compute_murcko_scaffold(smi)}`")
        st.json(ion)

with tab4:
    st.subheader("ICH Q14 Analytical Robustness Envelope")
    nom_rs = st.number_input("Nominal critical-pair resolution (Rs)", 0.5, 20.0, 1.85)
    env = lss_robustness_envelope(nom_rs)
    st.metric("Worst-Case Perturbed Rs (±10% tG, ±5% F, ±0.05 mL VD)", f"{env['worst_case_rs']:.2f}")
    if env["passes_baseline_1_5"]:
        st.success("PASS: Worst-case critical-pair resolution remains above Rs ≥ 1.50.")
    else:
        st.warning("CAUTION: Perturbed resolution falls below Rs = 1.50.")
