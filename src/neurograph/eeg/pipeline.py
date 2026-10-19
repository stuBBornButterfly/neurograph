import json
import time
from pathlib import Path

import mne
import pandas as pd

from neurograph.config import git_hash
from neurograph.eeg.epochs import (alpha_ratio, block_epochs, count_conditions, n_repaired,
                                   repair_by_channel, run_autoreject)
from neurograph.eeg.ica import fit_ica, label_ica, pick_exclude
from neurograph.eeg.load import crop_to_blocks, load_raw, read_events
from neurograph.eeg.preprocess import (average_reference, channel_metrics, filter_raw,
                                       find_bad_channels, fix_eog_labels, set_template_montage,
                                       swap_eog)
from neurograph.eeg.qc import eyes_blocks


def load_cropped(bids_dir: Path, subject: str, run: str,
                 eeg: dict) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    raw = load_raw(bids_dir, subject, run, eeg["task"], eeg["eog_channels"])
    blocks = eyes_blocks(read_events(bids_dir, subject, run, eeg["task"]))
    return crop_to_blocks(raw, blocks, eeg["block_pad_s"]), blocks


def prepare_for_ica(bids_dir: Path, subject: str, run: str, eeg: dict,
                    pre: dict) -> tuple[mne.io.BaseRaw, dict]:
    raw, _ = load_cropped(bids_dir, subject, run, eeg)
    verdict = fix_eog_labels(raw)
    dropped = set_template_montage(raw, pre["montage"], pre["drop_channels"])
    filter_raw(raw, pre["l_freq"], pre["h_freq"], pre["notch_freqs"])
    metrics = channel_metrics(raw, pre["bads"]["lf_band"], pre["bads"]["hf_band"])
    bads = find_bad_channels(metrics, pre["bads"]["z_thresh"], pre["bads"]["min_corr"])
    raw.info["bads"] = list(bads)
    average_reference(raw)
    return raw, {"eog_verdict": verdict, "dropped": dropped, "bads": bads, "metrics": metrics}


def prepare_analysis(bids_dir: Path, subject: str, run: str, eeg: dict, pre: dict,
                     swapped: bool, bads: list[str]) -> tuple[mne.io.BaseRaw, pd.DataFrame]:
    raw, blocks = load_cropped(bids_dir, subject, run, eeg)
    if swapped:
        swap_eog(raw)  # same decision as the ICA copy
    set_template_montage(raw, pre["montage"], pre["drop_channels"])
    raw.filter(pre["analysis"]["l_freq"], pre["analysis"]["h_freq"], picks=["eeg", "eog"],
               verbose="error")
    raw.info["bads"] = list(bads)
    average_reference(raw)
    return raw, blocks


def process_subject(bids_dir: Path, subject: str, run: str, eeg: dict, pre: dict,
                    out_dir: Path) -> tuple[dict, mne.Epochs]:
    t0 = time.perf_counter()
    ic, ep = pre["ica"], pre["epochs"]
    raw_ica, rep = prepare_for_ica(bids_dir, subject, run, eeg, pre)
    ica, fit_s = fit_ica(raw_ica, ic["n_components"], ic["method"], ic["decim"], ic["random_state"])
    labels = label_ica(raw_ica, ica)
    ica.exclude = pick_exclude(labels, ic["reject_labels"], ic["min_prob"])
    del raw_ica

    raw, blocks = prepare_analysis(bids_dir, subject, run, eeg, pre,
                                   rep["eog_verdict"] == "looks SWAPPED", list(rep["bads"]))
    ica.apply(raw, verbose="error")
    raw.interpolate_bads(reset_bads=True, verbose="error")
    average_reference(raw)  # re-reference now that bad channels are rebuilt

    epochs = block_epochs(raw, blocks, ep["length_s"], eeg["block_pad_s"]).pick("eeg")
    if ep["decim"] > 1:
        epochs.decimate(ep["decim"], verbose="error")
    made = count_conditions(epochs)
    t_ar = time.perf_counter()
    clean, log = run_autoreject(epochs, pre["autoreject"])
    ar_s = time.perf_counter() - t_ar

    tag = f"sub-{subject}_run-{run}"
    for sub in ("epochs", "ica", "qc"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    clean.save(out_dir / "epochs" / f"{tag}_epo.fif", overwrite=True, verbose="error")
    ica.save(out_dir / "ica" / f"{tag}_ica.fif", overwrite=True, verbose="error")

    excluded = labels.iloc[ica.exclude]["label"].value_counts().to_dict()
    qc = {"subject": subject, "run": run, "commit": git_hash(),
          "eog_verdict": rep["eog_verdict"], "bads": list(rep["bads"]),
          "bad_reasons": rep["bads"], "n_bads": len(rep["bads"]),
          "ica_n_excluded": len(ica.exclude), "ica_excluded": {k: int(v) for k, v in excluded.items()},
          "ica_n_brain": int((labels["label"] == "brain").sum()),
          "epochs_made": made, "epochs_kept": count_conditions(clean),
          "pct_rejected": round(100 * float(log.bad_epochs.mean()), 1),
          "n_repaired": n_repaired(log), "most_repaired": repair_by_channel(log),
          "alpha_ratio_o1": round(alpha_ratio(clean), 2),
          "sfreq": float(clean.info["sfreq"]), "n_channels": len(clean.ch_names),
          "ica_fit_s": round(fit_s, 1), "autoreject_s": round(ar_s, 1),
          "total_s": round(time.perf_counter() - t0, 1)}
    (out_dir / "qc" / f"{tag}_qc.json").write_text(json.dumps(qc, indent=2))
    return qc, clean
