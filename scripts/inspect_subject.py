import argparse
from pathlib import Path

import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.load import bids_path, load_raw
from neurograph.eeg.qc import channel_std_uv, eog_swap_check, eyes_blocks, plot_psd, psd_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="001")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    paths, eeg = load_config("paths"), load_config("eeg")
    bids_dir = Path(paths["paths"]["bids_dir"])
    run = args.run or eeg["main_run"]
    raw = load_raw(bids_dir, args.subject, run, eeg["task"], eeg["eog_channels"])

    bp = bids_path(bids_dir, args.subject, run, eeg["task"])
    events = pd.read_csv(bp.copy().update(suffix="events", extension=".tsv").fpath, sep="\t")
    std = channel_std_uv(raw).sort_values()
    blocks = eyes_blocks(events)
    eog = eog_swap_check(raw)
    ps = psd_summary(raw, eeg["line_freq"])
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / f"sub-{args.subject}_run-{run}_psd.png"
    plot_psd(ps["psd"], ps["freqs"], ps["ch_names"], fig_path, f"sub-{args.subject} run-{run} raw PSD")

    types = raw.get_channel_types()
    print(f"\n== sub-{args.subject} run-{run} | commit {git_hash()} ==")
    print(f"sfreq: {raw.info['sfreq']} Hz | duration: {raw.times[-1]:.1f} s | "
          f"channels: {len(types)} (eeg {types.count('eeg')}, eog {types.count('eog')})")
    print(f"montage set: {raw.get_montage() is not None} | "
          f"file filters: {raw.info['highpass']}-{raw.info['lowpass']} Hz")
    print("\n== amplitude, 1-40 Hz std (µV) ==")
    print(f"median {std.median():.1f} | lowest: {std.head(3).round(1).to_dict()} | "
          f"highest: {std.tail(3).round(1).to_dict()}")
    print("\n== eyes blocks ==")
    print("order:", "".join("C" if e == "closed" else "O" for e in blocks["eyes"]))
    print(blocks.round(1).to_string(index=False))
    print("\n== EOG swap check (|corr|, 1-10 Hz) ==")
    swapped = eog["HEOG~FP"] > eog["VEOG~FP"]
    print(eog, "-> looks SWAPPED" if swapped else "-> labels look correct")
    print("\n== spectrum ==")
    print(f"alpha peak O1: {ps['alpha_peak_o1']:.1f} Hz | {eeg['line_freq']} Hz line: "
          f"{ps['line_db']:+.1f} dB above neighbours")
    print(f"figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()