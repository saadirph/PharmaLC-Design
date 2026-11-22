"""Command-line interface for PharmaLC-Design."""

import argparse
import json
from .data import REFERENCE_SYSTEMS, VALIDATED_DRUGS
from .descriptors import compute_murcko_scaffold, estimate_ionization_and_logd
from .models import ConditionRecommender
from .scouting import predict_gradient_tr, scout_systems


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pharmalc",
        description="PharmaLC-Design: Anchor-Free RP-LC Retention Prediction & Method Scouting",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand 1: recommend
    p_rec = sub.add_parser("recommend", help="Recommend stationary phase & additive from SMILES")
    p_rec.add_argument("--smiles", required=True, help="Canonical SMILES of target analyte")
    p_rec.add_argument("--ph", type=float, default=2.7, help="Eluent pH (default: 2.7)")

    # Subcommand 2: scout
    p_sct = sub.add_parser("scout", help="Rank candidate C18 systems for a validated drug panel")
    p_sct.add_argument(
        "--drugs",
        nargs="+",
        default=["Bromazepam", "Methylphenidate", "Alprazolam"],
        help="List of drug names from built-in validated library",
    )

    # Subcommand 3: project
    p_prj = sub.add_parser("project", help="Convert anchor-free phi_e to gradient retention time t_R")
    p_prj.add_argument("--phi-e", type=float, required=True, help="Predicted elution composition (% organic)")
    p_prj.add_argument("--phi-init", type=float, default=5.0)
    p_prj.add_argument("--phi-final", type=float, default=95.0)
    p_prj.add_argument("--tg", type=float, default=15.0, help="Gradient time (min)")
    p_prj.add_argument("--t0", type=float, default=0.45, help="Column dead time (min)")
    p_prj.add_argument("--vd", type=float, default=0.11, help="System dwell volume (mL)")
    p_prj.add_argument("--flow", type=float, default=0.45, help="Flow rate (mL/min)")

    args = parser.parse_args(argv)

    if args.command == "recommend":
        rec = ConditionRecommender().recommend(args.smiles)
        ion = estimate_ionization_and_logd(args.smiles, pH=args.ph)
        scaf = compute_murcko_scaffold(args.smiles)
        out = {
            "smiles": args.smiles,
            "murcko_scaffold": scaf,
            "ionization_at_pH": ion,
            "recommendation": {
                "stationary_phase": rec["recommended_phase"],
                "mobile_phase_additive": rec["suggested_additive"],
            },
        }
        print(json.dumps(out, indent=2))
    elif args.command == "scout":
        subset = {d: VALIDATED_DRUGS[d] for d in args.drugs if d in VALIDATED_DRUGS}
        if len(subset) < 2:
            raise SystemExit("Provide at least 2 valid drug names from: " + ", ".join(VALIDATED_DRUGS.keys()))
        ranked = scout_systems(subset, REFERENCE_SYSTEMS[REFERENCE_SYSTEMS["mode"] == "gradient"])
        print(ranked.to_string(index=False))
    elif args.command == "project":
        tr = predict_gradient_tr(
            phi_e=args.phi_e, phi_init=args.phi_init, phi_final=args.phi_final,
            t_g=args.tg, t0=args.t0, v_d=args.vd, flow=args.flow,
        )
        print(json.dumps({"phi_e": args.phi_e, "predicted_tR_min": tr}, indent=2))


if __name__ == "__main__":
    main()
