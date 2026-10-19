import re
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from neurograph.models.evaluate import build_model, make_splits

FEATURE_RE = re.compile(r"^(closed|open)_(delta|theta|alpha|beta|gamma)_(abs|rel)_(\w+)$")


def lr_coefficients(X: np.ndarray, y: np.ndarray, splits: list, cfg: dict,
                    cols: list[str]) -> pd.DataFrame:
    rows = []
    for r, f, tr, te in splits:
        model = build_model("logreg", cfg).fit(X[tr], y[tr])
        rows.append(model[-1].coef_[0])  # weights on standardised features
    return pd.DataFrame(rows, columns=cols)


def shap_importance(X: np.ndarray, y: np.ndarray, splits: list, cfg: dict,
                    cols: list[str]) -> pd.DataFrame:
    # xgboost's built-in tree SHAP on each test fold (repeat 0 covers every person once)
    parts = []
    for r, f, tr, te in [s for s in splits if s[0] == 0]:
        model = build_model("xgb", cfg).fit(X[tr], y[tr])
        contrib = model.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)[:, :-1]
        parts.append(pd.DataFrame(contrib, columns=cols))
    shap = pd.concat(parts)
    values = pd.DataFrame(np.vstack([X[s[3]] for s in splits if s[0] == 0]), columns=cols)
    direction = [np.corrcoef(values[c].fillna(values[c].median()), shap[c])[0, 1]
                 if shap[c].std() > 0 else 0.0 for c in cols]
    return pd.DataFrame({"mean_abs_shap": shap.abs().mean(), "direction": direction}, index=cols)


def _perm_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict, n_splits: int,
              seed: int) -> float:
    warnings.filterwarnings("ignore", category=FutureWarning)  # runs in a worker process
    rng = np.random.default_rng(seed)
    yp = rng.permutation(y)  # break the link between EEG and label
    splits = make_splits(yp, groups, n_splits, 1, seed)
    aucs = [roc_auc_score(yp[te], build_model("logreg", cfg).fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1])
            for _, _, tr, te in splits]
    return float(np.mean(aucs))


def permutation_null(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cfg: dict, n_perm: int,
                     n_splits: int, seed: int) -> np.ndarray:
    return np.array(Parallel(n_jobs=-1)(delayed(_perm_auc)(X, y, groups, cfg, n_splits, seed + k)
                                        for k in range(n_perm)))


def grouped_importance(weights: pd.Series) -> pd.DataFrame:
    # sum of |weight| per condition x band; FAA and IAF reported separately
    rows = []
    for name, w in weights.items():
        m = FEATURE_RE.match(name)
        if m:
            rows.append({"condition": m[1], "band": m[2], "w": abs(w)})
        else:
            rows.append({"condition": name.split("_")[0] if "faa" in name else "-",
                         "band": "faa" if "faa" in name else name, "w": abs(w)})
    return pd.DataFrame(rows).pivot_table(index="condition", columns="band", values="w", aggfunc="sum")
