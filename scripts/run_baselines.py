import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.features import find_input_derivatives
from neurograph.models.evaluate import confusion, cross_validate, make_splits


def load_table(in_dir: Path) -> pd.DataFrame:
    feats = pd.read_csv(in_dir / "features" / "features.csv", dtype={"sub": str})
    labels = pd.read_csv(in_dir / "labels.csv", dtype={"sub": str})
    df = labels[["sub", "group", "y_dep", "sex"]].merge(feats, on="sub", validate="one_to_one")
    return df.assign(female=(df["sex"] == 1).astype(int)).copy()


def design(df: pd.DataFrame, spec: dict, sets: dict) -> tuple[np.ndarray, list[str]]:
    cols = [] if spec["features"] is None else list(df.filter(regex=sets[spec["features"]]).columns)
    if spec["sex"]:
        cols = cols + ["female"]
    if not cols:
        return np.zeros((len(df), 1)), []  # dummy model ignores X
    return df[cols].to_numpy(float), cols


def subset_mask(df: pd.DataFrame, name: str, flagged: list[str]) -> np.ndarray:
    if name == "female":
        return (df["female"] == 1).to_numpy()
    if name == "no_flagged":
        return (~df["sub"].isin(flagged)).to_numpy()
    return np.ones(len(df), dtype=bool)


def plot_auc(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(summary) + 1.2))
    y = np.arange(len(summary))[::-1]
    ax.barh(y, summary["auc_mean"], xerr=summary["auc_sd"], color="tab:blue", alpha=0.7, capsize=3)
    ax.axvline(0.5, c="grey", ls="--", lw=1, label="chance")
    sex = summary.loc[summary["model"] == "sex_only", "auc_mean"]
    if len(sex):
        ax.axvline(float(sex.iloc[0]), c="tab:red", ls=":", lw=1.2, label="sex only")
    ax.set_yticks(y, [f"{m} (n={n})" for m, n in zip(summary["model"], summary["n"])])
    ax.set(xlim=(0.3, 1.0), xlabel="ROC AUC (mean ± SD over folds)", title="Step 4.1 baselines")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn API-change notices
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    t0 = time.perf_counter()
    paths, m, study = load_config("paths"), load_config("models"), load_config("study")
    in_dir = find_input_derivatives(m["input_glob"]).parent  # .../derivatives
    out = Path(paths["paths"]["derivatives_dir"]) / "models"
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(in_dir)
    cv, thr = m["cv"], m["threshold"]

    folds, oofs, summary, splits_cache = [], [], [], {}
    for name, spec in m["models"].items():
        mask = subset_mask(df, spec["subset"], study["qc_flagged"])
        sub = df[mask].reset_index(drop=True)
        X, cols = design(sub, spec, m["feature_sets"])
        y = sub["y_dep"].to_numpy()
        if spec["subset"] not in splits_cache:  # same folds for every model on the same subjects
            splits_cache[spec["subset"]] = make_splits(y, sub["sub"].to_numpy(), cv["n_splits"],
                                                       cv["n_repeats"], cv["random_state"])
        res, oof = cross_validate(X, y, splits_cache[spec["subset"]], spec.get("dummy", False),
                                  m["logreg"], thr)
        folds.append(res.assign(model=name))
        oofs.append(pd.DataFrame({"model": name, "repeat": np.repeat(np.arange(len(oof)), len(y)),
                                  "sub": np.tile(sub["sub"], len(oof)), "y": np.tile(y, len(oof)),
                                  "p_dep": oof.ravel()}))
        summary.append({"model": name, "n": len(y), "n_dep": int(y.sum()), "n_features": len(cols),
                        "auc_mean": res["auc"].mean(), "auc_sd": res["auc"].std(),
                        "bal_acc": res["bal_acc"].mean(), "brier": res["brier"].mean(),
                        "cm_repeat0": confusion(y, oof[0], thr)})
    folds, oofs, summary = pd.concat(folds), pd.concat(oofs), pd.DataFrame(summary)

    all_splits = splits_cache["all"]
    pd.DataFrame([{"sub": s, "repeat": r, "fold": f} for r, f, _, te in all_splits
                  for s in df["sub"].to_numpy()[te]]).to_csv(out / "splits.csv", index=False)
    folds.to_csv(out / "cv_folds.csv", index=False)
    oofs.to_csv(out / "oof_predictions.csv", index=False)
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "auc_baselines.png"
    plot_auc(summary, fig_path)

    n_dep = int(df["y_dep"].sum())
    print(f"== step 4.1 baselines | commit {git_hash()} | input {in_dir} ==")
    print(f"subjects {len(df)} ({len(df) - n_dep} CTL / {n_dep} DEP) | women {int(df['female'].sum())} "
          f"({int(df.loc[df.female == 1, 'y_dep'].sum())} DEP) | flagged excluded in sensitivity: {len(study['qc_flagged'])}")
    print(f"CV: StratifiedGroupKFold {cv['n_splits']} folds x {cv['n_repeats']} repeats = "
          f"{len(all_splits)} folds | groups = subject | logistic regression, C tuned in inner {m['logreg']['inner_cv']}-fold CV")
    print(f"\n{'model':16s} {'n':>4s} {'feat':>5s}  {'AUC mean ± SD':>15s}  {'bal acc':>7s}  {'Brier':>6s}  confusion (repeat 0, p>=0.5)")
    for _, r in summary.iterrows():
        print(f"{r['model']:16s} {r['n']:4d} {r['n_features']:5d}  {r['auc_mean']:.3f} ± {r['auc_sd']:.3f}  "
              f"{r['bal_acc']:7.3f}  {r['brier']:6.3f}  {r['cm_repeat0']}")

    print("\n-- does EEG add? paired over the same 50 folds --")
    for a, b in m["paired"]:
        d = (folds[folds.model == a].set_index(["repeat", "fold"])["auc"]
             - folds[folds.model == b].set_index(["repeat", "fold"])["auc"])
        print(f"  {a} - {b}: ΔAUC {d.mean():+.3f} ± {d.std():.3f} | EEG better in {(d > 0).mean():.0%} of folds")
    print("\nleakage checklist: one row per subject ✓ | folds split by subject ✓ | imputer, scaler, C fitted "
          "inside training folds ✓ | no feature selection on the full data ✓ | same folds across models ✓")
    print(f"time {time.perf_counter() - t0:.0f} s")
    print(f"saved: {out}/cv_folds.csv | oof_predictions.csv | splits.csv | figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
