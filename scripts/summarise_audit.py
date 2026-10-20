import time

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config

SCORES = {"correct (2 yes / 1 partly / 0 no)": "correct", "grounded (2 / 1 / 0)": "grounded",
          "safe_and_appropriate (1 / 0)": "safe", "helpful (1-5)": "helpful"}


def main() -> None:
    t0 = time.perf_counter()
    a = load_config("benchmark")["audit"]
    out = REPO_ROOT / a["out"]
    sheet = pd.read_csv(out).rename(columns=SCORES)
    key = pd.read_csv(out.with_name(out.stem + "_key.csv"))
    df = sheet.merge(key[["audit_id", "model"]], on="audit_id")
    for c in SCORES.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    rated = df.dropna(subset=["correct", "grounded"])
    if rated.empty:
        raise SystemExit(f"no ratings yet in {out}: fill the rating columns first")
    tab = rated.groupby("model").agg(n=("audit_id", "size"), correct=("correct", "mean"), grounded=("grounded", "mean"),
                                     safe=("safe", "mean"), helpful=("helpful", "mean")).round(2)
    print(f"== human audit summary | commit {git_hash()} ==")
    print(f"rated {len(rated)} of {len(df)} answers (correct / grounded on 0-2, safe 0-1, helpful 1-5)")
    print(tab.to_string())
    print(rated.groupby("set")[["correct", "grounded"]].mean().round(2).to_string())
    print(f"time {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
