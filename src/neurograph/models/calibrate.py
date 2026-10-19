import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def calibration_summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    # slope 1 / intercept 0 = well calibrated; slope < 1 = overconfident
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    logit = np.log(pc / (1 - pc)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    return {"brier": brier_score_loss(y, p), "log_loss": log_loss(y, pc), "mean_p": float(p.mean()),
            "slope": float(fit.coef_[0, 0]), "intercept": float(fit.intercept_[0])}


def reliability(y: np.ndarray, p: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    frac, mean_p = calibration_curve(y, p, n_bins=bins, strategy="quantile")
    return mean_p, frac  # predicted vs observed share of DEP per bin


def assign_tiers(p: np.ndarray, tiers: dict[str, list[float]]) -> np.ndarray:
    out = np.empty(len(p), dtype=object)
    for name, (lo, hi) in tiers.items():
        out[(p >= lo) & ((p < hi) | (hi >= 1.0))] = name
    return out


def tier_table(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    g = df.groupby("tier")
    tab = pd.DataFrame({"n": g.size(), "CTL": g["y_dep"].apply(lambda s: int((s == 0).sum())),
                        "DEP": g["y_dep"].sum(),
                        "DEP_elevated": g["dep_severity"].apply(lambda s: int((s == "elevated").sum())),
                        "DEP_high": g["dep_severity"].apply(lambda s: int((s == "high").sum())),
                        "median_BDI": g["BDI"].median(), "median_STAI": g["STAI"].median()})
    tab["pct_DEP"] = (100 * tab["DEP"] / tab["n"]).round(0)
    tab = tab.reindex(order)  # keep empty tiers visible
    counts = ["n", "CTL", "DEP", "DEP_elevated", "DEP_high"]
    tab[counts] = tab[counts].fillna(0).astype(int)
    return tab


def severity_tracking(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    dep = df[df["y_dep"] == 1]
    out = {}
    for name, sub, col in [("BDI (all)", df, "BDI"), ("BDI (within DEP)", dep, "BDI"),
                           ("STAI (within DEP)", dep, "STAI")]:
        rho, pval = spearmanr(sub["p_dep"], sub[col], nan_policy="omit")
        out[name] = (float(rho), float(pval))
    return out
