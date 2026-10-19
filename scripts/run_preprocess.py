import argparse
import gc
import time
import traceback
from pathlib import Path

import pandas as pd

from neurograph.config import git_hash, load_config
from neurograph.eeg.pipeline import process_subject


def subjects_to_run(labels_csv: Path, only: list[str] | None) -> list[str]:
    subs = pd.read_csv(labels_csv, dtype={"sub": str})["sub"].tolist()
    return [s for s in subs if not only or s in only]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--redo", action="store_true")  # reprocess even if results exist
    args = parser.parse_args()

    paths, eeg, pre = load_config("paths"), load_config("eeg"), load_config("preprocess")
    bids, out = Path(paths["paths"]["bids_dir"]), Path(paths["paths"]["derivatives_dir"])
    run = eeg["main_run"]
    subs = subjects_to_run(out / "labels.csv", args.subjects)
    print(f"== run_preprocess | commit {git_hash()} | {len(subs)} subjects ==", flush=True)

    done, skipped, failed = 0, 0, []
    t0 = time.perf_counter()
    for i, s in enumerate(subs, 1):
        head = f"[{i:3d}/{len(subs)}] sub-{s}"
        if (out / "qc" / f"sub-{s}_run-{run}_qc.json").exists() and not args.redo:
            skipped += 1
            print(f"{head} skip (already done)", flush=True)
            continue
        try:
            qc, _ = process_subject(bids, s, run, eeg, pre, out)
            done += 1
            print(f"{head} ok | bads {qc['n_bads']} | ICA -{qc['ica_n_excluded']} "
                  f"(brain {qc['ica_n_brain']}) | rej {qc['pct_rejected']}% | "
                  f"alpha x{qc['alpha_ratio_o1']} | {qc['total_s']:.0f} s", flush=True)
        except Exception as e:  # one bad recording must not stop the batch
            failed.append({"subject": s, "error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc()})
            print(f"{head} FAILED {type(e).__name__}: {e}", flush=True)
        gc.collect()

    if failed:
        (out / "qc").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed).to_csv(out / "qc" / "failed.csv", index=False)
    print(f"\ndone {done} | skipped {skipped} | failed {len(failed)} | "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
