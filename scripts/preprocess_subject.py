import argparse
from pathlib import Path

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.load import crop_to_blocks, load_raw, read_events
from neurograph.eeg.preprocess import (average_reference, channel_metrics, filter_raw,
                                       find_bad_channels, fix_eog_labels, set_template_montage)
from neurograph.eeg.qc import channel_std_uv, eyes_blocks, plot_psd, psd_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="001")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    paths, eeg, pre = load_config("paths"), load_config("eeg"), load_config("preprocess")
    bids_dir = Path(paths["paths"]["bids_dir"])
    run = args.run or eeg["main_run"]
    fig_dir = REPO_ROOT / paths["paths"]["figures_dir"]
    tag = f"sub-{args.subject}_run-{run}"

    raw = load_raw(bids_dir, args.subject, run, eeg["task"], eeg["eog_channels"])
    raw = crop_to_blocks(raw, eyes_blocks(read_events(bids_dir, args.subject, run, eeg["task"])),
                         eeg["block_pad_s"])
    before = psd_summary(raw, eeg["line_freq"])
    amp_before = channel_std_uv(raw).median()
    plot_psd(before["psd"], before["freqs"], before["ch_names"], fig_dir / f"{tag}_psd_before.png",
             f"{tag} before cleaning")

    verdict = fix_eog_labels(raw)
    dropped = set_template_montage(raw, pre["montage"], pre["drop_channels"])
    filter_raw(raw, pre["l_freq"], pre["h_freq"], pre["notch_freqs"])
    metrics = channel_metrics(raw, pre["bads"]["lf_band"], pre["bads"]["hf_band"])
    bads = find_bad_channels(metrics, pre["bads"]["z_thresh"], pre["bads"]["min_corr"])
    raw.info["bads"] = list(bads)
    average_reference(raw)

    after = psd_summary(raw, eeg["line_freq"])
    amp_after = channel_std_uv(raw.copy().drop_channels(raw.info["bads"])).median()
    plot_psd(after["psd"], after["freqs"], after["ch_names"], fig_dir / f"{tag}_psd_after.png",
             f"{tag} after filter + notch + avg ref (bads excluded)")

    n_eeg = len(raw.copy().pick("eeg").ch_names)
    print(f"\n== {tag} | step 2.1 | commit {git_hash()} ==")
    print(f"EOG check: {verdict} -> {'renamed HEOG<->VEOG' if verdict == 'looks SWAPPED' else 'unchanged'}")
    print(f"montage: {pre['montage']} | dropped: {dropped} | eeg channels: {n_eeg}")
    print(f"filter: {pre['l_freq']}-{pre['h_freq']} Hz + notch {pre['notch_freqs']}")
    print(f"\nbad channels ({len(bads)}): {bads}")
    print("closest calls:")
    print(f"  highest |amplitude z|: {metrics['z_sd'].abs().nlargest(3).round(1).to_dict()}")
    print(f"  highest high-freq z : {metrics['z_hf'].nlargest(3).round(1).to_dict()}")
    print(f"  highest high-freq power: {metrics.nlargest(3, 'z_hfpow')[['z_hfpow', 'hf_db']].round(1).to_dict('index')}")
    print(f"  lowest max corr     : {metrics['max_corr'].nsmallest(3).round(2).to_dict()}")
    print(f"reference: average of {n_eeg - len(bads)} good channels")
    print("\n== before -> after ==")
    print(f"amplitude median (1-40 Hz std): {amp_before:.1f} -> {amp_after:.1f} µV")
    print(f"alpha peak O1: {before['alpha_peak_o1']:.1f} -> {after['alpha_peak_o1']:.1f} Hz")
    print(f"60 Hz vs neighbours: {before['line_db']:+.1f} -> {after['line_db']:+.1f} dB")
    print(f"figures: {tag}_psd_before.png, {tag}_psd_after.png")


if __name__ == "__main__":
    main()