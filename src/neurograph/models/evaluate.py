import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from neurograph.models.confounds import SexResidualizer


def make_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, n_repeats: int,
                seed: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    out = []
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            out.append((r, f, tr, te))
    return out


def build_model(kind: str, cfg: dict):
    if kind.endswith("_cal"):  # Platt scaling learned by an inner CV on the training fold
        cal = cfg["calibration"]
        return CalibratedClassifierCV(build_model(kind[:-4], cfg), method=cal["method"], cv=cal["cv"])
    if kind == "dummy":
        return DummyClassifier(strategy="prior")
    if kind == "xgb":  # shallow trees, fixed settings: too few people to tune them honestly
        return XGBClassifier(eval_metric="logloss", **cfg["xgb"])
    # imputer, scaler and C are all fitted on the training fold only
    lr = cfg["logreg"]
    steps = [SimpleImputer(strategy="median")]
    if kind == "logreg_sexadj":  # remove what sex predicts from every feature, then drop sex
        steps.append(SexResidualizer())
    return make_pipeline(*steps, StandardScaler(),
                         LogisticRegressionCV(Cs=lr["Cs"], cv=lr["inner_cv"], scoring="roc_auc",
                                              class_weight="balanced", max_iter=lr["max_iter"]))


def cross_validate(X: np.ndarray, y: np.ndarray, splits: list, kind: str, cfg: dict,
                   threshold: float) -> tuple[pd.DataFrame, np.ndarray]:
    n_rep = max(s[0] for s in splits) + 1
    oof = np.full((n_rep, len(y)), np.nan)  # out-of-fold P(DEP) per repeat
    rows = []
    for r, f, tr, te in splits:
        model = build_model(kind, cfg).fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        oof[r, te] = p
        rows.append({"repeat": r, "fold": f, "n_test": len(te), "auc": roc_auc_score(y[te], p),
                     "bal_acc": balanced_accuracy_score(y[te], p >= threshold),
                     "brier": brier_score_loss(y[te], p)})
    return pd.DataFrame(rows), oof


def confusion(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    pred = p >= threshold
    return {"TN": int(((y == 0) & ~pred).sum()), "FP": int(((y == 0) & pred).sum()),
            "FN": int(((y == 1) & ~pred).sum()), "TP": int(((y == 1) & pred).sum())}
