import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.features import CONDITIONS, band_power, faa, region_map, regional_row


def group_diff(df: pd.DataFrame, col: str) -> tuple[float, float]:
    a, b = df.loc[df["group"] == "CTL", col], df.loc[df["group"] == "DEP", col]
    return float(b.median() - a.median()), float(mannwhitneyu(a, b).pvalue)


def plot_checks(df: pd.DataFrame, regions: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    groups = ["CTL", "DEP"]
    axes[0].boxplot([df.loc[df["group"] == g, "closed_faa_F4_F3"] for g in groups])
    axes[0].set_xticks([1, 2], groups)
    axes[0].axhline(0, c="grey", lw=0.8)
    axes[0].set(title="FAA ln(F4) - ln(F3), eyes closed", ylabel="FAA")
    for k, g in enumerate(groups):
        m = [df.loc[df["group"] == g, f"closed_gamma_abs_{r}"] for r in regions]
        axes[1].errorbar(np.arange(len(regions)) + 0.1 * k, [x.mean() for x in m],
                         yerr=[x.std() / np.sqrt(len(x)) for x in m], fmt="o-", label=g, capsize=3)
    axes[1].set_xticks(range(len(regions)), regions)
    axes[1].set(title="gamma 30-45 Hz by region, closed", ylabel="log10 µV²")
    axes[1].legend(fontsize=8)
    cells = [(s, g) for s in ["F", "M"] for g in groups]
    axes[2].boxplot([df.loc[(df["sex_label"] == s) & (df["group"] == g), "closed_beta_abs_central"]
                     for s, g in cells])
    axes[2].set_xticks(range(1, len(cells) + 1), [f"{s}-{g}" for s, g in cells])
    axes[2].set(title="central beta by sex and group, closed", ylabel="log10 µV²")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    paths, f = load_config("paths"), load_config("features")
    out = Path(paths["paths"]["derivatives_dir"])
    z = np.load(out / "features" / "psd.npz")
    psd, freqs, chs = z["psd"].astype(np.float64), z["freqs"], z["ch_names"].tolist()
    subs = z["subjects"].tolist()
    labels = pd.read_csv(out / "labels.csv", dtype={"sub": str}).set_index("sub").loc[subs]
    bp = pd.read_csv(out / "features" / "band_power.csv", dtype={"sub": str}).set_index("sub").loc[subs]

    bands = list(f["bands"])
    absolute, relative = band_power(psd, freqs, f["bands"])  # (subject, condition, channel, band)
    a = bands.index("alpha")
    regions = region_map(chs, f["regions"])

    rows = []
    for i, s in enumerate(subs):
        weak = bp.loc[s, "iaf_peak_db"] < f["iaf_min_peak_db"]
        row = {"sub": s, "iaf_hz": np.nan if weak else bp.loc[s, "iaf_hz"],
               "iaf_peak_db": bp.loc[s, "iaf_peak_db"]}
        row.update(faa(absolute[i, :, :, a], chs, f["faa_pairs"]))
        row.update(regional_row(absolute[i], relative[i], chs, regions, bands))
        rows.append(row)
    feats = pd.DataFrame(rows)
    feats.to_csv(out / "features" / "features.csv", index=False)
    (out / "features" / "regions.json").write_text(json.dumps(regions, indent=1))

    # checks: descriptive only, never used to choose features
    chk = feats.assign(group=labels["group"].to_numpy(),
                       sex_label=labels["sex"].map({1: "F", 2: "M"}).to_numpy())
    print(f"== step 3.2 FAA + regions + DE | commit {git_hash()} ==")
    print("regions: " + " | ".join(f"{r} {len(c)}" for r, c in regions.items())
          + f" | not used: {sorted(set(chs) - {c for v in regions.values() for c in v})}")
    n_eeg = feats.shape[1] - 1
    print(f"features.csv: {feats.shape} | NaN {int(feats.isna().sum().sum())} "
          f"(IAF missing for weak peaks: {int(feats['iaf_hz'].isna().sum())})")
    r, _ = pearsonr(feats["closed_alpha_de_occipital"], feats["closed_alpha_abs_occipital"])
    print(f"DE vs log power (occipital alpha): r = {r:.4f}  -> same information, different scale")

    print("\n-- FAA (eyes closed), DEP - CTL median | p (descriptive) --")
    for rr, ll in f["faa_pairs"]:
        col = f"closed_faa_{rr}_{ll}"
        d, p = group_diff(chk, col)
        print(f"  {rr}-{ll}: CTL {chk.loc[chk.group == 'CTL', col].median():+.3f} | "
              f"DEP {chk.loc[chk.group == 'DEP', col].median():+.3f} | diff {d:+.3f} | p {p:.3f}")

    print("\n-- muscle check: gamma 30-45 Hz (closed, log10 µV²), DEP - CTL by region --")
    for reg in regions:
        d, p = group_diff(chk, f"closed_gamma_abs_{reg}")
        print(f"  {reg:9s} diff {d:+.3f} | p {p:.3f}")
    chk["temporal_minus_central_gamma"] = chk["closed_gamma_abs_temporal"] - chk["closed_gamma_abs_central"]
    d, p = group_diff(chk, "temporal_minus_central_gamma")
    print(f"  temporal - central gamma: diff {d:+.3f} | p {p:.3f}  (muscle -> DEP excess largest at temporal)")

    print("\n-- sex check: central beta & gamma (closed), DEP - CTL within each sex --")
    for s in ["F", "M"]:
        sub = chk[chk["sex_label"] == s]
        n = sub["group"].value_counts().to_dict()
        db, pb = group_diff(sub, "closed_beta_abs_central")
        dg, pg = group_diff(sub, "closed_gamma_abs_central")
        print(f"  {s}: n {n} | beta diff {db:+.3f} (p {pb:.3f}) | gamma diff {dg:+.3f} (p {pg:.3f})")
    for g in ["CTL", "DEP"]:
        sub = chk[chk["group"] == g]
        d = (sub.loc[sub.sex_label == "F", "closed_beta_abs_central"].median()
             - sub.loc[sub.sex_label == "M", "closed_beta_abs_central"].median())
        print(f"  within {g}: female - male central beta {d:+.3f}")

    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "features_checks.png"
    plot_checks(chk, list(regions), fig_path)
    print(f"\nsaved: {out / 'features' / 'features.csv'} ({n_eeg} features) | regions.json")
    print(f"figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
