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


def exclusion_reason(row: pd.Series, manual: dict) -> str:
    if row["sub"] in manual:
        return manual[row["sub"]]
    if np.isnan(row["BDI"]):
        return "missing BDI"
    if not row["run01"]:
        return "no run-01"
    if row["run01_blocks"] != 6:
        return f"run-01 has {row['run01_blocks']} blocks"
    return ""


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
    df["group"] = df["BDI"].apply(lambda b: bdi_group(b, g["ctl_max"], g["dep_min"]))
    df["bdi_ii_tier"] = df["BDI"].apply(lambda b: bdi_tier(b, study["bdi_ii_tiers"]))
    df["excluded"] = df.apply(lambda r: exclusion_reason(r, study["exclude"]), axis=1)
    use = df[df["excluded"] == ""]

    out_dir = Path(paths["paths"]["derivatives_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "subject_summary.csv", index=False)
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "participants_bdi_stai.png"
    plot_scores(use, fig_path, g["ctl_max"], g["dep_min"])

    r01 = df[df["run01"] == True]
    print(f"\n== recordings | commit {git_hash()} ==")
    print(f"participants: {len(df)} | run-01: {int(df['run01'].sum())} | run-02: {int(df['run02'].sum())}")
    print(f"run-01 blocks per subject: {r01['run01_blocks'].value_counts().to_dict()}")
    print(f"run-01 duration s: min {r01['run01_dur_s'].min()} | median {r01['run01_dur_s'].median()} | max {r01['run01_dur_s'].max()}")
    print(f"run-01 fewest markers in any block: {int(r01['run01_min_markers'].min())}")
    print(f"run-01 orders: {r01['run01_order'].value_counts().to_dict()}")
    print(f"no run-02: {df.loc[df['run02'] != True, 'participant_id'].tolist()}")

    print("\n== exclusions ==")
    for _, r in df[df["excluded"] != ""].iterrows():
        print(f"  {r['participant_id']}: {r['excluded']}")
    print(f"usable: {len(use)}")

    print("\n== labels (usable subjects) ==")
    print(f"sex (1=female, 2=male): {use['sex'].value_counts().to_dict()} | age median {use['age'].median()}")
    print(f"design groups: {use['group'].value_counts().to_dict()}")
    print(f"BDI-II tiers: {use['bdi_ii_tier'].value_counts().reindex(list(study['bdi_ii_tiers'])).fillna(0).astype(int).to_dict()}")
    print(f"BDI in DEP: {use.loc[use['group'] == 'DEP', 'BDI'].describe()[['min', '50%', 'max']].to_dict()}")
    for c in study["stai_cutoffs"]:
        print(f"STAI >= {c}: {int((use['STAI'] >= c).sum())} "
              f"(CTL {int(((use['STAI'] >= c) & (use['group'] == 'CTL')).sum())}, "
              f"DEP {int(((use['STAI'] >= c) & (use['group'] == 'DEP')).sum())})")
    print(f"STAI median: CTL {use.loc[use['group'] == 'CTL', 'STAI'].median()} | DEP {use.loc[use['group'] == 'DEP', 'STAI'].median()}")
    rho, p = spearmanr(use["BDI"], use["STAI"], nan_policy="omit")
    print(f"BDI~STAI Spearman rho = {rho:.2f} (p = {p:.1e})")
    print(f"\nsaved: {out_dir / 'subject_summary.csv'} | {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()