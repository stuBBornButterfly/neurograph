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
from neurograph.models.evaluate import cross_validate, make_splits
from neurograph.models.explain import (grouped_importance, lr_coefficients, permutation_null,
                                       shap_importance)
from scripts.run_baselines import design, load_table, subset_mask


def plot_explain(grouped: pd.DataFrame, shap: pd.DataFrame, null: np.ndarray, observed: float,
                 out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    im = axes[0].imshow(grouped.fillna(0).to_numpy(), cmap="Blues")
    axes[0].set_xticks(range(grouped.shape[1]), grouped.columns, rotation=30)
    axes[0].set_yticks(range(grouped.shape[0]), grouped.index)
    axes[0].set_title("LR |weight| summed by condition x band")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    top = shap.sort_values("mean_abs_shap").tail(15)
    colors = ["tab:red" if d > 0 else "tab:blue" for d in top["direction"]]
    axes[1].barh(range(len(top)), top["mean_abs_shap"], color=colors)
    axes[1].set_yticks(range(len(top)), top.index, fontsize=7)
    axes[1].set(title="XGBoost mean |SHAP| (red: higher value -> more DEP)", xlabel="mean |SHAP|")
    axes[2].hist(null, bins=25, color="grey")
    axes[2].axvline(observed, c="tab:red", lw=2, label=f"observed {observed:.3f}")
    axes[2].set(title="eeg_closed: AUC with shuffled labels", xlabel="AUC", ylabel="permutations")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    t0 = time.perf_counter()
    paths, m, study = load_config("paths"), load_config("models"), load_config("study")
    ex, cv, thr = m["explain"], m["cv"], m["threshold"]
    in_dir = find_input_derivatives(m["input_glob"]).parent
    out = Path(paths["paths"]["derivatives_dir"]) / "models"
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(in_dir)

    data, folds, summary, cache = {}, [], [], {}
    for name, spec in ex["models"].items():
        sub = df[subset_mask(df, spec["subset"], study["qc_flagged"])].reset_index(drop=True)
        X, cols = design(sub, spec, m["feature_sets"])
        y = sub["y_dep"].to_numpy()
        if spec["subset"] not in cache:  # same seeds as step 4.1 -> identical folds
            cache[spec["subset"]] = make_splits(y, sub["sub"].to_numpy(), cv["n_splits"],
                                                cv["n_repeats"], cv["random_state"])
        res, _ = cross_validate(X, y, cache[spec["subset"]], spec["kind"], m, thr)
        data[name] = (X, y, cols, sub["sub"].to_numpy(), cache[spec["subset"]])
        folds.append(res.assign(model=name))
        summary.append({"model": name, "n": len(y), "feat": len(cols), "auc": res["auc"].mean(),
                        "sd": res["auc"].std(), "bal_acc": res["bal_acc"].mean(), "brier": res["brier"].mean()})
    folds = pd.concat(folds)
    folds.to_csv(out / "explain_folds.csv", index=False)

    print(f"== step 4.2 XGBoost + permutation + importance | commit {git_hash()} ==")
    print(f"{'model':16s} {'n':>4s} {'feat':>5s}  {'AUC mean ± SD':>15s}  {'bal acc':>7s}  {'Brier':>6s}")
    for r in summary:
        print(f"{r['model']:16s} {r['n']:4d} {r['feat']:5d}  {r['auc']:.3f} ± {r['sd']:.3f}  "
              f"{r['bal_acc']:7.3f}  {r['brier']:6.3f}")

    print("\n-- paired over the same 50 folds --")
    for a, b in ex["paired"]:
        d = (folds[folds.model == a].set_index(["repeat", "fold"])["auc"]
             - folds[folds.model == b].set_index(["repeat", "fold"])["auc"])
        print(f"  {a} - {b}: ΔAUC {d.mean():+.3f} ± {d.std():.3f} | first better in {(d > 0).mean():.0%} of folds")

    pm = ex["permutation"]
    print(f"\n-- permutation test: {pm['n_perm']} label shuffles, {pm['n_splits']}-fold CV each --")
    perm_rows, null_closed, obs_closed = [], None, None
    for name in pm["models"]:
        X, y, cols, groups, _ = data[name]
        null = permutation_null(X, y, groups, m, pm["n_perm"], pm["n_splits"], pm["random_state"])
        obs = float(folds.loc[folds.model == name, "auc"].mean())
        p = (1 + int((null >= obs).sum())) / (1 + len(null))
        perm_rows.append({"model": name, "observed": obs, "null_mean": null.mean(),
                          "null_95": np.percentile(null, 95), "p": p})
        print(f"  {name:16s} observed {obs:.3f} | shuffled mean {null.mean():.3f}, 95th pct "
              f"{np.percentile(null, 95):.3f}, max {null.max():.3f} | p = {p:.4f}")
        if name == "eeg_closed_lr":
            null_closed, obs_closed = null, obs
    pd.DataFrame(perm_rows).to_csv(out / "permutation.csv", index=False)

    X, y, cols, _, splits = data[ex["coef_model"]]
    coefs = lr_coefficients(X, y, splits, m, cols)
    mean = coefs.mean()
    stab = (np.sign(coefs) == np.sign(mean)).mean()
    coef_tab = pd.DataFrame({"mean": mean, "sd": coefs.std(), "same_sign": stab})
    coef_tab.to_csv(out / "lr_coefficients.csv")
    print(f"\n-- {ex['coef_model']}: top 12 weights (standardised; + = higher value -> more DEP) --")
    for f, r in coef_tab.reindex(mean.abs().sort_values(ascending=False).index).head(12).iterrows():
        print(f"  {f:28s} {r['mean']:+.3f} ± {r['sd']:.3f} | same sign in {r['same_sign']:.0%} of folds")
    grouped = grouped_importance(mean)
    print("\n-- LR |weight| summed by condition x band --")
    print(grouped.round(2).to_string())

    X, y, cols, _, splits = data[ex["shap_model"]]
    shap = shap_importance(X, y, splits, m, cols)
    shap.to_csv(out / "shap_importance.csv")
    print(f"\n-- {ex['shap_model']}: top 12 mean |SHAP| (direction: + = higher value -> more DEP) --")
    for f, r in shap.sort_values("mean_abs_shap", ascending=False).head(12).iterrows():
        print(f"  {f:28s} {r['mean_abs_shap']:.3f} | direction {r['direction']:+.2f}")
    print("\n-- SHAP summed by condition x band --")
    print(grouped_importance(shap["mean_abs_shap"]).round(3).to_string())

    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "model_explain.png"
    plot_explain(grouped, shap, null_closed, obs_closed, fig_path)
    print(f"\ntime {time.perf_counter() - t0:.0f} s")
    print(f"saved: {out}/explain_folds.csv | permutation.csv | lr_coefficients.csv | shap_importance.csv")
    print(f"figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
