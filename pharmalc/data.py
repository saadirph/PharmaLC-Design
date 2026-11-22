"""Built-in reference library of shortlisted C18 systems and prospectively validated drug panels."""

import pandas as pd

REFERENCE_SYSTEMS = pd.DataFrame([
    {"system_id": "0186", "column": "Agilent ZORBAX Extend-C18 (50x2.1 mm, 1.8 um)", "modifier": "ACN", "mode": "gradient", "phi_init": 5.0, "phi_final": 95.0, "t_g": 21.1, "t0": 0.42, "v_d": 0.11, "flow": 0.40},
    {"system_id": "0350", "column": "Waters ACQUITY UPLC HSS C18 (100x2.1 mm, 1.8 um)", "modifier": "MeOH", "mode": "gradient", "phi_init": 5.0, "phi_final": 95.0, "t_g": 7.0, "t0": 0.44, "v_d": 0.12, "flow": 0.45},
    {"system_id": "0004", "column": "Thermo Hypersil GOLD C18 (150x2.1 mm, 1.9 um)", "modifier": "ACN", "mode": "gradient", "phi_init": 5.0, "phi_final": 95.0, "t_g": 13.7, "t0": 0.45, "v_d": 0.11, "flow": 0.50},
    {"system_id": "SHIM-ISO", "column": "Shimadzu Shim-pack C18 (250x4.6 mm, 5.0 um)", "modifier": "ACN", "mode": "isocratic", "phi_init": 60.0, "phi_final": 60.0, "t_g": 10.0, "t0": 2.45, "v_d": 0.25, "flow": 1.00},
    {"system_id": "0433", "column": "Merck Supelco Ascentis Express C18 (100x2.1 mm)", "modifier": "MeOH", "mode": "gradient", "phi_init": 5.0, "phi_final": 95.0, "t_g": 20.7, "t0": 0.55, "v_d": 0.12, "flow": 0.30},
    {"system_id": "0019", "column": "Waters XBridge C18 (50x2.1 mm)", "modifier": "MeOH", "mode": "gradient", "phi_init": 5.0, "phi_final": 95.0, "t_g": 17.6, "t0": 0.45, "v_d": 0.12, "flow": 0.20},
])

VALIDATED_DRUGS = {
    "Bromazepam": {"smiles": "O=C1CN=C(c2ccccn2)c2cc(Br)ccc2N1", "ACN": 39.1, "MeOH": 65.0},
    "Methylphenidate": {"smiles": "COC(=O)C(c1ccccc1)C1CCCCN1", "ACN": 48.3, "MeOH": 58.2},
    "Alprazolam": {"smiles": "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2", "ACN": 58.1, "MeOH": 70.7},
    "Favipiravir": {"smiles": "NC(=O)c1nc(F)cnc1O", "ACN": 18.2, "MeOH": 26.0},
    "Roflumilast": {"smiles": "O=C(Nc1c(Cl)cncc1Cl)c1ccc(OC(F)F)c(OCC2CC2)c1", "ACN": 58.7, "MeOH": 72.5},
    "Paracetamol": {"smiles": "CC(=O)Nc1ccc(O)cc1", "ACN": 16.5, "MeOH": 24.5},
    "Bisoprolol": {"smiles": "CC(C)NCC(O)COc1ccc(COCCOC(C)C)cc1", "ACN": 38.4, "MeOH": 52.0},
    "Amlodipine": {"smiles": "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", "ACN": 48.7, "MeOH": 64.0},
    "Clopidogrel": {"smiles": "COC(=O)C(c1ccccc1Cl)N1CCc2sccc2C1", "ACN": 64.2, "MeOH": 77.5},
}
