"""Module 3: Condition Classifiers, Retention Regressors, PCA AD Filter, and Mondrian Conformal Intervals."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .descriptors import compute_core_condition_descriptors


class PCAApplicabilityDomain:
    """20-component PCA Euclidean kNN (k=5) applicability-domain distance & reliability flag."""

    def __init__(self, n_components: int = 20, n_neighbors: int = 5, flag_threshold: float = 5.5):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.flag_threshold = flag_threshold
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.nn: Optional[NearestNeighbors] = None

    def fit(self, X: np.ndarray) -> "PCAApplicabilityDomain":
        Xs = self.scaler.fit_transform(self.imputer.fit_transform(X))
        n_comp = min(self.n_components, Xs.shape[0], Xs.shape[1])
        self.pca = PCA(n_components=n_comp, random_state=42)
        Z = self.pca.fit_transform(Xs)
        k = min(self.n_neighbors, Z.shape[0])
        self.nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Z)
        return self

    def distance(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(self.imputer.transform(X))
        Z = self.pca.transform(Xs)
        dists, _ = self.nn.kneighbors(Z)
        return dists.mean(axis=1)

    def is_flagged(self, X: np.ndarray) -> np.ndarray:
        return self.distance(X) > self.flag_threshold


class MondrianSplitConformal:
    """AD-tercile stratified (Mondrian) split-conformal prediction intervals."""

    def __init__(self, alpha: float = 0.10, n_bins: int = 3):
        self.alpha = alpha
        self.n_bins = n_bins
        self.edges_: Optional[np.ndarray] = None
        self.quantiles_: Dict[int, float] = {}

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray, ad_dist: np.ndarray) -> "MondrianSplitConformal":
        resid = np.abs(np.asarray(y_true) - np.asarray(y_pred))
        ad = np.asarray(ad_dist)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = np.quantile(ad, qs)
        bins = np.digitize(ad, self.edges_)
        for b in range(self.n_bins):
            r_b = resid[bins == b]
            n = len(r_b)
            if n == 0:
                self.quantiles_[b] = float(np.quantile(resid, 1.0 - self.alpha))
            else:
                level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
                self.quantiles_[b] = float(np.quantile(r_b, level))
        return self

    def predict_interval(self, y_pred: np.ndarray, ad_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yp = np.asarray(y_pred, dtype=float)
        bins = np.digitize(np.asarray(ad_dist), self.edges_)
        hw = np.array([self.quantiles_[int(b)] for b in bins], dtype=float)
        return yp - hw, yp + hw


class ConditionRecommender:
    """Physicochemical recommender for stationary-phase family and MS-compatible additives."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.col_clf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed)
        self.ms_clf = RandomForestClassifier(n_estimators=500, random_state=seed)
        self.fitted_ = False
        self.classes_: List[str] = ["Primesep", "Obelisc", "Newcrom", "C18 / Specialty"]

    def fit(self, X: pd.DataFrame, y_family: np.ndarray, y_ms: np.ndarray) -> "ConditionRecommender":
        self.col_clf.fit(X.values, y_family)
        self.ms_clf.fit(X.values, y_ms)
        self.classes_ = list(self.col_clf.classes_)
        self.fitted_ = True
        return self

    def recommend(self, smiles: str) -> Dict[str, object]:
        """Recommend stationary phase and volatile additive from SMILES (uses ML if fitted, else mechanistic rules)."""
        d = compute_core_condition_descriptors(smiles)
        if self.fitted_:
            x = np.array([[d[k] for k in ["logP", "TPSA", "MW", "HBD", "HBA", "aromRings", "rotB", "acid", "base", "fsp3", "MolMR"]]])
            probs = self.col_clf.predict_proba(x)[0]
            top_idx = int(np.argmax(probs))
            ms_prob = float(self.ms_clf.predict_proba(x)[0, 1])
            return {
                "recommended_phase": self.classes_[top_idx],
                "phase_confidence": round(float(probs[top_idx]), 3),
                "ms_compatible_additive_prob": round(ms_prob, 3),
                "suggested_additive": "0.1% Formic acid / NH4 formate (MS-compatible)" if ms_prob >= 0.5 else "Phosphate / H3PO4 buffer (UV)",
                "descriptors": d,
            }
        # Mechanistic fallback when no training table is loaded
        ionizable = (d["acid"] + d["base"]) >= 1
        if d["logP"] < 1.5 and ionizable:
            phase = "Primesep (Mixed-Mode)"
            add = "0.1% Formic acid or TFA / NH4 formate"
        elif d["logP"] < 2.2 and d["TPSA"] > 65:
            phase = "Obelisc (Zwitterionic) or Polar-Embedded C18"
            add = "NH4 acetate / Formic acid (MS-compatible)"
        else:
            phase = "USP L1 (C18)"
            add = "0.1% Formic acid (ACN or MeOH gradient)"
        return {
            "recommended_phase": phase,
            "phase_confidence": None,
            "ms_compatible_additive_prob": None,
            "suggested_additive": add,
            "descriptors": d,
        }


class RetentionModelSuite:
    """Factory for the benchmarked PharmaLC-Design retention models (LightGBM, MLP, RF, Ridge)."""

    @staticmethod
    def build(name: str = "LightGBM", seed: int = 42):
        if name == "Ridge":
            return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
        if name == "RF":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestRegressor(n_estimators=200, max_features=0.33, min_samples_leaf=2, random_state=seed, n_jobs=-1),
            )
        if name == "MLP":
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu", alpha=1e-3,
                             learning_rate_init=1e-3, batch_size=256, max_iter=200,
                             early_stopping=True, random_state=seed),
            )
        if name == "LightGBM":
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(
                    n_estimators=1500, learning_rate=0.04, num_leaves=63,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                    reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
                )
            except ImportError:
                return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=63, random_state=seed)
        raise ValueError(f"Unknown model name: {name}")
