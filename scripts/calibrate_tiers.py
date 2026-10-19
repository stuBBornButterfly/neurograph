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
from neurograph.models.calibrate import (assign_tiers, calibration_summary, reliability,
                                         severity_tracking, tier_table)
from neurograph.models.evaluate import cross_validate, make_splits
from scripts.run_baselines import design, load_table, subset_mask


def plot_tiers(curves: dict, people: pd.DataFrame, tiers: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot([0, 1], [0, 1], c="grey", ls="--", lw=1)
    for name, (mp, frac) in curves.items():
        axes[0].plot(mp, frac, "o-", ms=3, label=name)
    axes[0].set(title="reliability (pooled out-of-fold)", xlabel="predicted P(DEP)",
                ylabel="observed share DEP", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=7)
    bins = np.linspace(0, 1, 21)
    for g, color in [("CTL", "tab:blue"), ("DEP", "tab:red")]:
        axes[1].hist(people.loc[people.group == g, "p_dep"], bins=bins, alpha=0.6, color=color, label=g)
    for lo, _ in list(tiers.values())[1:]:
        axes[1].axvline(lo, c="k", ls=":", lw=1)
    axes[1].set(title="calibrated P(DEP) per person (tier cut-offs dotted)", xlabel="P(DEP)", ylabel="people")
    axes[1].legend(fontsize=8)
    for g, color in [("CTL", "tab:blue"), ("DEP", "tab:red")]:
        s = people[people.group == g]
        axes[2].scatter(s["BDI"], s["p_dep"], s=14, c=color, label=g)
    axes[2].set(title="P(DEP) vs BDI", xlabel="BDI", ylabel="P(DEP)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    t0 = time.perf_counter()
    paths, m, study = load_config("paths"), load_config("models"), load_config("study")
    tc, cv, thr = m["tiers"], m["cv"], m["threshold"]
    in_dir = find_input_derivatives(m["input_glob"]).parent
    out = Path(paths["paths"]["derivatives_dir"]) / "models"
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(in_dir)
    labels = pd.read_csv(in_dir / "labels.csv", dtype={"sub": str})

    folds, oofs, summary, cache = [], {}, [], {}
    for name, spec in tc["models"].items():
        sub = df[subset_mask(df, spec["subset"], study["qc_flagged"])].reset_index(drop=True)
        X, cols = design(sub, spec, m["feature_sets"])
        y = sub["y_dep"].to_numpy()
        if spec["subset"] not in cache:  # same seeds as steps 4.1-4.2 -> identical folds
            cache[spec["subset"]] = make_splits(y, sub["sub"].to_numpy(), cv["n_splits"],
                                                cv["n_repeats"], cv["random_state"])
        res, oof = cross_validate(X, y, cache[spec["subset"]], spec["kind"], m, thr)
        folds.append(res.assign(model=name))
        oofs[name] = (oof, y, sub["sub"].to_numpy())
        summary.append({"model": name, "auc": res["auc"].mean(), "sd": res["auc"].std(),
                        "inputs": len(cols) - (1 if spec["kind"] == "logreg_sexadj" else 0)})
    folds = pd.concat(folds)
    folds.to_csv(out / "tiers_folds.csv", index=False)

    print(f"== step 4.3 calibration + risk tiers + sex-adjusted check | commit {git_hash()} ==")
    print(f"{'model':20s} {'inputs':>6s}  AUC mean ± SD")
    for r in summary:
        print(f"{r['model']:20s} {r['inputs']:6d}  {r['auc']:.3f} ± {r['sd']:.3f}")
    print("(sexadj: sex's linear effect removed from every feature inside each training fold; sex itself not used)")

    print("\n-- paired over the same 50 folds --")
    for a, b in tc["paired"]:
        d = (folds[folds.model == a].set_index(["repeat", "fold"])["auc"]
             - folds[folds.model == b].set_index(["repeat", "fold"])["auc"])
        print(f"  {a} - {b}: ΔAUC {d.mean():+.3f} ± {d.std():.3f} | first better in {(d > 0).mean():.0%} of folds")

    print(f"\n-- calibration (pooled out-of-fold, prevalence {df['y_dep'].mean():.3f}) --")
    print(f"{'model':20s} {'Brier':>6s} {'logloss':>8s} {'mean P':>7s} {'slope':>6s} {'intercept':>9s}")
    curves = {}
    for name in tc["calibrate_compare"]:
        oof, y, _ = oofs[name]
        yy, pp = np.tile(y, len(oof)), oof.ravel()
        c = calibration_summary(yy, pp)
        curves[name] = reliability(yy, pp, m["calibration"]["bins"])
        print(f"{name:20s} {c['brier']:6.3f} {c['log_loss']:8.3f} {c['mean_p']:7.3f} "
              f"{c['slope']:6.2f} {c['intercept']:9.2f}")
    print("(well calibrated: mean P ≈ prevalence, slope ≈ 1, intercept ≈ 0)")

    oof, y, subs = oofs[tc["tier_model"]]
    people = pd.DataFrame({"sub": subs, "p_dep": oof.mean(axis=0)})  # mean over the 10 repeats
    people = people.merge(labels[["sub", "group", "y_dep", "sex", "BDI", "STAI", "dep_severity"]], on="sub")
    people["tier"] = assign_tiers(people["p_dep"].to_numpy(), study["risk_tiers"])
    people.to_csv(out / "risk_tiers.csv", index=False)
    tab = tier_table(people, list(study["risk_tiers"]))
    tab["female_pct"] = people.groupby("tier")["sex"].apply(lambda s: round(100 * (s == 1).mean())).reindex(tab.index)
    print(f"\n-- risk tiers from {tc['tier_model']} (cut-offs {study['risk_tiers']}) --")
    print(tab.to_string())
    print("\n-- does P(DEP) track symptom severity? (Spearman) --")
    for k, (rho, pval) in severity_tracking(people).items():
        print(f"  {k:18s} rho {rho:+.2f} | p {pval:.3f}")

    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "calibration_tiers.png"
    plot_tiers(curves, people, study["risk_tiers"], fig_path)
    print(f"\ntime {time.perf_counter() - t0:.0f} s")
    print(f"saved: {out}/tiers_folds.csv | risk_tiers.csv | figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
