import argparse
from pathlib import Path

from neurograph.config import REPO_ROOT, load_config
from neurograph.eeg.pipeline import process_subject
from neurograph.eeg.qc import plot_condition_psd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="001")
    parser.add_argument("--run", default=None)
    args = parser.parse_args()

    paths, eeg, pre = load_config("paths"), load_config("eeg"), load_config("preprocess")
    run = args.run or eeg["main_run"]
    out_dir = Path(paths["paths"]["derivatives_dir"])
    qc, clean = process_subject(Path(paths["paths"]["bids_dir"]), args.subject, run, eeg, pre, out_dir)

    tag = f"sub-{args.subject}_run-{run}"
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / f"{tag}_psd_open_closed.png"
    plot_condition_psd(clean, ["O1", "OZ", "F3", "F4"], fig_path, f"{tag} clean epochs")

    print(f"\n== {tag} | step 2.3 | commit {qc['commit']} ==")
    print(f"EOG: {qc['eog_verdict']} | bads: {qc['bad_reasons']}")
    print(f"ICA: removed {qc['ica_n_excluded']} {qc['ica_excluded']} | brain comps {qc['ica_n_brain']}")
    print(f"analysis copy: {pre['analysis']['l_freq']}-{pre['analysis']['h_freq']} Hz, avg ref, "
          f"bads interpolated | {qc['n_channels']} ch @ {qc['sfreq']:.0f} Hz")
    print(f"epochs ({pre['epochs']['length_s']} s): made {qc['epochs_made']} -> kept {qc['epochs_kept']}")
    print(f"AutoReject: rejected {qc['pct_rejected']}% | repaired {qc['n_repaired']} epochs")
    print(f"alpha O1 closed/open: {qc['alpha_ratio_o1']}x")
    print(f"time: ICA fit {qc['ica_fit_s']} s | AutoReject {qc['autoreject_s']} s | total {qc['total_s']} s")
    for sub, suffix in [("epochs", "epo.fif"), ("ica", "ica.fif"), ("qc", "qc.json")]:
        f = out_dir / sub / f"{tag}_{suffix}"
        print(f"saved: {f} ({f.stat().st_size / 1e6:.1f} MB)")
    print(f"figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
