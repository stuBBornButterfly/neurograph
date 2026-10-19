import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu

from neurograph.config import REPO_ROOT, git_hash, load_config

METRICS = [("pct_rejected", "% epochs rejected"), ("n_bads", "bad channels"),
           ("ica_n_excluded", "ICs removed"), ("alpha_ratio_o1", "alpha O1 closed/open")]


def load_qc(qc_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(qc_dir.glob("*_qc.json")):
        r = json.loads(p.read_text())
        rep = r.pop("most_repaired", {}) or {}
        r.pop("bad_reasons", None)
        r["bads"] = ";".join(r["bads"])
        r["top_repaired"] = next(iter(rep), "")
        r["top_repaired_pct"] = next(iter(rep.values()), 0.0)
        rows.append(r)
    df = pd.json_normalize(rows)
    return df.fillna({c: 0 for c in df.columns if c.startswith("ica_excluded.")})


def flag_subjects(df: pd.DataFrame, f: dict) -> pd.Series:
    def reasons(r: pd.Series) -> str:
        out = []
        if r["pct_rejected"] > f["max_pct_rejected"]:
            out.append(f"rejected {r['pct_rejected']}%")
        if r["n_bads"] > f["max_bads"]:
            out.append(f"{r['n_bads']} bad ch")
        if r["ica_n_brain"] < f["min_brain_ics"]:
            out.append(f"{r['ica_n_brain']} brain ICs")
        if r["ica_n_excluded"] > f["max_ica_excluded"]:
            out.append(f"{r['ica_n_excluded']} ICs removed")
        if r["alpha_ratio_o1"] < f["min_alpha_ratio"]:
            out.append(f"alpha x{r['alpha_ratio_o1']}")
        if min(r["epochs_kept.closed"], r["epochs_kept.open"]) < f["min_epochs_per_condition"]:
            out.append("few epochs")
        return "; ".join(out)
    return df.apply(reasons, axis=1)


def group_compare(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    m = df.merge(labels[["sub", "group"]], left_on="subject", right_on="sub")
    rows = []
    for col, _ in METRICS:
        a, b = m.loc[m["group"] == "CTL", col], m.loc[m["group"] == "DEP", col]
        rows.append({"metric": col, "CTL_median": a.median(), "DEP_median": b.median(),
                     "p": mannwhitneyu(a, b).pvalue})
    return pd.DataFrame(rows).round(3)


def plot_qc(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.3))
    for ax, (col, title) in zip(axes, METRICS):
        ax.hist(df[col], bins=20, color="tab:blue")
        ax.set(title=title, ylabel="subjects")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    paths, pre = load_config("paths"), load_config("preprocess")
    out = Path(paths["paths"]["derivatives_dir"])
    df = load_qc(out / "qc")
    labels = pd.read_csv(out / "labels.csv", dtype={"sub": str})
    df["flags"] = flag_subjects(df, pre["qc_flags"])
    df.to_csv(out / "qc_table.csv", index=False)
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "qc_summary.png"
    plot_qc(df, fig_path)

    print(f"== QC summary | commit {git_hash()} | {len(df)} of {len(labels)} subjects processed ==")
    failed = out / "qc" / "failed.csv"
    if failed.exists():
        print(f"failed: {pd.read_csv(failed, dtype={'subject': str})[['subject', 'error']].to_dict('records')}")
    for col, title in METRICS:
        s = df[col]
        print(f"{title:22s} median {s.median():6.1f} | IQR {s.quantile(.25):.1f}-{s.quantile(.75):.1f} | "
              f"min {s.min():.1f} | max {s.max():.1f}")
    print(f"kept epochs per condition: min closed {int(df['epochs_kept.closed'].min())}, "
          f"min open {int(df['epochs_kept.open'].min())}")
    print(f"brain ICs: median {df['ica_n_brain'].median():.0f} | min {df['ica_n_brain'].min()}")
    print(f"EOG verdicts: {df['eog_verdict'].value_counts().to_dict()} | "
          f"swapped: {int(df['eog_swapped'].sum())} of {len(df)}")
    bads = Counter(c for b in df["bads"] for c in b.split(";") if c)
    print(f"most common bad channels: {dict(bads.most_common(8))}")
    print(f"most repaired channel (n subjects): {df['top_repaired'].value_counts().head(6).to_dict()} | "
          f"median {df['top_repaired_pct'].median():.0f}% of epochs")
    print(f"time per subject: median {df['total_s'].median():.0f} s | total {df['total_s'].sum() / 60:.0f} min")

    print("\n== CTL vs DEP (preprocessing must not differ by group) ==")
    print(group_compare(df, labels).to_string(index=False))

    flagged = df[df["flags"] != ""]
    print(f"\n== flagged for a manual look: {len(flagged)} ==")
    for _, r in flagged.iterrows():
        print(f"  sub-{r['subject']}: {r['flags']}")
    print(f"\nsaved: {out / 'qc_table.csv'} | {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
