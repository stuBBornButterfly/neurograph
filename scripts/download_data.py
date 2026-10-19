import argparse
from pathlib import Path

import openneuro
import pandas as pd

from neurograph.config import git_hash, load_config


def download_subset(dataset: str, tag: str, target: Path, subjects: list[str]) -> None:
    # essential metadata (participants.tsv, README...) is always included
    openneuro.download(dataset=dataset, tag=tag, target_dir=target,
                       include=[f"sub-{s}" for s in subjects])


def list_files(folder: Path) -> list[tuple[str, float]]:
    return [(str(f.relative_to(folder)), f.stat().st_size / 1e6)
            for f in sorted(folder.rglob("*")) if f.is_file()]


def describe_score(df: pd.DataFrame, col: str) -> str:
    s = pd.to_numeric(df[col], errors="coerce")
    return (f"{col}: n={s.notna().sum()} missing={s.isna().sum()} "
            f"min={s.min():.0f} median={s.median():.0f} max={s.max():.0f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=["001"])
    args = parser.parse_args()

    cfg = load_config("paths")
    bids_dir = Path(cfg["paths"]["bids_dir"])
    download_subset(cfg["dataset"]["id"], cfg["dataset"]["version"], bids_dir, args.subjects)

    files = list_files(bids_dir)
    df = pd.read_csv(bids_dir / "participants.tsv", sep="\t")

    print("\n== download ==")
    print(f"commit: {git_hash()} | target: {bids_dir}")
    print(f"files: {len(files)} | size: {sum(s for _, s in files):.1f} MB")
    print("\n== participants.tsv ==")
    print(f"rows: {len(df)} | columns: {list(df.columns)}")
    for col in ["BDI", "STAI", "age"]:
        print(describe_score(df, col))
    print(f"original ID 544 -> {df.loc[df['Original_ID'] == 544, 'participant_id'].tolist()}")
    print("\n== files ==")
    for name, size in files:
        print(f"  {size:8.1f} MB  {name}")


if __name__ == "__main__":
    main()