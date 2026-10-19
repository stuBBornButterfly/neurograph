import argparse
from pathlib import Path

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.ica import eog_check, fit_ica, label_ica, pick_exclude, plot_ica_topos
from neurograph.eeg.pipeline import prepare_for_ica
from neurograph.eeg.preprocess import channel_metrics
from neurograph.eeg.qc import plot_before_after, plot_psd, psd_summary


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
    ic = pre["ica"]

    raw, rep = prepare_for_ica(bids_dir, args.subject, run, eeg, pre)
    ica, secs = fit_ica(raw, ic["n_components"], ic["method"], ic["decim"], ic["random_state"])
    labels = label_ica(raw, ica)
    exclude = pick_exclude(labels, ic["reject_labels"], ic["min_prob"])
    eog_inds, eog_top, eog_r = eog_check(raw, ica)

    ica.exclude = exclude
    clean = ica.apply(raw.copy(), verbose="error")
    interpolated = list(clean.info["bads"])
    clean.interpolate_bads(reset_bads=True, verbose="error")
    m0 = rep["metrics"]
    m1 = channel_metrics(clean, pre["bads"]["lf_band"], pre["bads"]["hf_band"])
    ps = psd_summary(clean, eeg["line_freq"])

    plot_ica_topos(ica, labels, exclude, fig_dir / f"{tag}_ica_components.png")
    plot_before_after(raw, clean, ["FP1", "FPZ", "FP2", "F8", "T8", "O1"], 5.0, 10.0,
                      fig_dir / f"{tag}_ica_before_after.png")
    plot_psd(ps["psd"], ps["freqs"], ps["ch_names"], fig_dir / f"{tag}_psd_after_ica.png",
             f"{tag} after ICA + interpolation")

    agree = eog_top in exclude and labels.iloc[eog_top]["label"] == "eye blink"
    unsure = labels[(labels["label"] != "brain") & (labels["prob"] < ic["min_prob"])]
    print(f"\n== {tag} | step 2.2 ICA | commit {git_hash()} ==")
    print(f"bads: {rep['bads']} | EOG: {rep['eog_verdict']}")
    print(f"ICA: {ic['method']} (extended) | {ica.n_components_} comps from "
          f"{len(ica.ch_names)} channels | decim {ic['decim']} | fit {secs:.1f} s")
    print(f"ICLabel counts: {labels['label'].value_counts().to_dict()}")
    print("excluded (p >= {:.1f}): {}".format(ic["min_prob"], [
        f"IC{i:02d} {labels.iloc[i]['label']} {labels.iloc[i]['prob']:.2f}" for i in exclude]))
    print(f"not excluded, unsure (non-brain, p < {ic['min_prob']}): "
          f"{[f'{k} {r.label} {r.prob:.2f}' for k, r in unsure.iterrows()]}")
    print(f"VEOG check: EOG-linked ICs {eog_inds} | best IC{eog_top:02d} |r|={eog_r:.2f} "
          f"-> agrees with ICLabel: {'yes' if agree else 'NO'}")
    print(f"interpolated: {interpolated}")
    print("\n== effect ==")
    print(f"FP1 amplitude: {m0.loc['FP1', 'sd_uv']:.1f} -> {m1.loc['FP1', 'sd_uv']:.1f} µV")
    print(f"T8 high-freq power: {m0.loc['T8', 'hf_db']:.1f} -> {m1.loc['T8', 'hf_db']:.1f} dB")
    print(f"alpha peak O1: {ps['alpha_peak_o1']:.1f} Hz | 60 Hz: {ps['line_db']:+.1f} dB")
    print(f"figures: {tag}_ica_components.png, {tag}_ica_before_after.png, {tag}_psd_after_ica.png")


if __name__ == "__main__":
    main()