import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.qc import eyes_blocks


def run_info(eeg_dir: Path, sub: str, run: str, task: str) -> dict:
    stem = eeg_dir / f"sub-{sub}_task-{task}_run-{run}"
    ev_path = Path(f"{stem}_events.tsv")
    if not ev_path.exists():
        return {f"run{run}": False}
    blocks = eyes_blocks(pd.read_csv(ev_path, sep="\t"))
    info = json.loads(Path(f"{stem}_eeg.json").read_text())
    return {f"run{run}": True,
            f"run{run}_dur_s": round(float(info.get("RecordingDuration", np.nan)), 1),
            f"run{run}_blocks": len(blocks),
            f"run{run}_n_closed": int((blocks["eyes"] == "closed").sum()),
            f"run{run}_n_open": int((blocks["eyes"] == "open").sum()),
            f"run{run}_order": "".join("C" if e == "closed" else "O" for e in blocks["eyes"]),
            f"run{run}_min_markers": int(blocks["n"].min()) if len(blocks) else 0}


def bdi_group(bdi: float, ctl_max: int, dep_min: int) -> str:
    if np.isnan(bdi):
        return "missing"
    if bdi <= ctl_max:
        return "CTL"
    return "DEP" if bdi >= dep_min else "between"


def bdi_tier(bdi: float, tiers: dict) -> str:
    if np.isnan(bdi):
        return "missing"
    for name, (lo, hi) in tiers.items():
        if lo <= bdi <= hi:
            return name
    return "out_of_range"


def exclusion_reason(row: pd.Series, manual: dict, min_blocks: int) -> str:
    if row["sub"] in manual:
        return manual[row["sub"]]
    if np.isnan(row["BDI"]):
        return "missing BDI"
    if not row["run01"]:
        return "no run-01"
    if min(row["run01_n_closed"], row["run01_n_open"]) < min_blocks:
        return f"run-01 has {row['run01_n_closed']} closed / {row['run01_n_open']} open blocks"
    return ""


def make_labels(use: pd.DataFrame, study: dict) -> pd.DataFrame:
    cols = ["participant_id", "sub", "sex", "age", "BDI", "STAI", "group", "bdi_ii_tier",
            "run01_order", "run01_n_closed", "run01_n_open", "run02"]
    out = use[cols].copy()
    out["y_dep"] = (out["group"] == "DEP").astype(int)
    out["stai_high"] = (out["STAI"] >= study["stai_high"]).astype(int)
    sb = study["severity_bands"]
    out["dep_severity"] = np.select([(out["y_dep"] == 1) & (out["BDI"] >= sb["high_min"]),
                                     out["y_dep"] == 1], ["high", "elevated"], "low")
    return out.reset_index(drop=True)


def plot_scores(df: pd.DataFrame, out_path: Path, ctl_max: int, dep_min: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    axes[0].hist(df["BDI"].dropna(), bins=range(0, 33), color="tab:blue")
    axes[0].axvline(ctl_max + 0.5, ls="--", c="green", label="CTL max")
    axes[0].axvline(dep_min - 0.5, ls="--", c="red", label="DEP min")
    axes[0].set(xlabel="BDI", ylabel="people", title="BDI")
    axes[0].legend()
    axes[1].hist(df["STAI"].dropna(), bins=range(20, 72, 2), color="tab:orange")
    axes[1].set(xlabel="STAI trait", title="STAI")
    axes[2].scatter(df["BDI"], df["STAI"], s=12)
    axes[2].set(xlabel="BDI", ylabel="STAI", title="BDI vs STAI")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    paths, eeg, study = load_config("paths"), load_config("eeg"), load_config("study")
    bids = Path(paths["paths"]["bids_dir"])
    g = study["design_groups"]

    df = pd.read_csv(bids / "participants.tsv", sep="\t")
    df["sub"] = df["participant_id"].str.replace("sub-", "", regex=False)
    df["BDI"] = pd.to_numeric(df["BDI"], errors="coerce")
    df["STAI"] = pd.to_numeric(df["STAI"], errors="coerce")
    runs = [{**run_info(bids / f"sub-{s}" / "eeg", s, "01", eeg["task"]),
             **run_info(bids / f"sub-{s}" / "eeg", s, "02", eeg["task"])} for s in df["sub"]]
    df = pd.concat([df, pd.DataFrame(runs)], axis=1)
    df["group_raw"] = df["BDI"].apply(lambda b: bdi_group(b, g["ctl_max"], g["dep_min"]))
    df["group"] = df["group_raw"].replace({"between": study["between_label"]})
    df["bdi_ii_tier"] = df["BDI"].apply(lambda b: bdi_tier(b, study["bdi_ii_tiers"]))
    df["excluded"] = df.apply(lambda r: exclusion_reason(r, study["exclude"],
                                                        study["min_blocks_per_condition"]), axis=1)
    use = df[df["excluded"] == ""]
    labels = make_labels(use, study)

    out_dir = Path(paths["paths"]["derivatives_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "subject_summary.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "participants_bdi_stai.png"
    plot_scores(labels, fig_path, g["ctl_max"], g["dep_min"])

    print(f"\n== recordings | commit {git_hash()} ==")
    print(f"participants: {len(df)} | run-01: {int(df['run01'].sum())} | run-02: {int(df['run02'].sum())}")
    print(f"run-01 orders: {df['run01_order'].value_counts().to_dict()}")

    print("\n== exclusions ==")
    for _, r in df[df["excluded"] != ""].iterrows():
        print(f"  {r['participant_id']}: {r['excluded']}")
    moved = use.loc[use["group_raw"] == "between", "participant_id"].tolist()
    print(f"relabelled between -> {study['between_label']}: {moved}")

    print("\n== final labels ==")
    n = len(labels)
    counts = labels["y_dep"].value_counts().to_dict()
    print(f"included: {n} | CTL (y=0): {counts.get(0, 0)} | DEP (y=1): {counts.get(1, 0)} | "
          f"majority baseline: {max(counts.values()) / n:.1%}")
    print(f"dep_severity: {labels['dep_severity'].value_counts().to_dict()}")
    print(f"stai_high by group: {pd.crosstab(labels['group'], labels['stai_high']).to_dict('index')}")
    print(f"sex by group (1=F, 2=M): {pd.crosstab(labels['group'], labels['sex']).to_dict('index')}")
    print(f"age median by group: {labels.groupby('group')['age'].median().to_dict()}")
    rho, p = spearmanr(labels["BDI"], labels["STAI"], nan_policy="omit")
    print(f"BDI~STAI Spearman rho = {rho:.2f} (p = {p:.1e})")
    print(f"\nsaved: {out_dir / 'labels.csv'} ({n} rows) | {out_dir / 'subject_summary.csv'}")


if __name__ == "__main__":
    main()