"""PharmaLC-Design: Anchor-Free RP-LC Retention Prediction & Method Scouting."""

__version__ = "1.0.0"

from .curation import (
    strip_salts_and_canonicalize,
    compute_inchikey14,
    compute_phi_e,
    compute_provenance_score,
)
from .descriptors import (
    compute_molecular_descriptors,
    compute_core_condition_descriptors,
    compute_murcko_scaffold,
    estimate_ionization_and_logd,
)
from .models import (
    PCAApplicabilityDomain,
    MondrianSplitConformal,
    ConditionRecommender,
    RetentionModelSuite,
)
from .scouting import (
    predict_gradient_tr,
    predict_isocratic_tr,
    evaluate_separation_metrics,
    scout_systems,
    lss_robustness_envelope,
)

__all__ = [
    "__version__",
    "strip_salts_and_canonicalize",
    "compute_inchikey14",
    "compute_phi_e",
    "compute_provenance_score",
    "compute_molecular_descriptors",
    "compute_core_condition_descriptors",
    "compute_murcko_scaffold",
    "estimate_ionization_and_logd",
    "PCAApplicabilityDomain",
    "MondrianSplitConformal",
    "ConditionRecommender",
    "RetentionModelSuite",
    "predict_gradient_tr",
    "predict_isocratic_tr",
    "evaluate_separation_metrics",
    "scout_systems",
    "lss_robustness_envelope",
]
