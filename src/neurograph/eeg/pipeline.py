from pathlib import Path

import mne

from neurograph.eeg.load import crop_to_blocks, load_raw, read_events
from neurograph.eeg.preprocess import (average_reference, channel_metrics, filter_raw,
                                       find_bad_channels, fix_eog_labels, set_template_montage)
from neurograph.eeg.qc import eyes_blocks


def prepare_for_ica(bids_dir: Path, subject: str, run: str, eeg: dict,
                    pre: dict) -> tuple[mne.io.BaseRaw, dict]:
    raw = load_raw(bids_dir, subject, run, eeg["task"], eeg["eog_channels"])
    blocks = eyes_blocks(read_events(bids_dir, subject, run, eeg["task"]))
    raw = crop_to_blocks(raw, blocks, eeg["block_pad_s"])
    verdict = fix_eog_labels(raw)
    dropped = set_template_montage(raw, pre["montage"], pre["drop_channels"])
    filter_raw(raw, pre["l_freq"], pre["h_freq"], pre["notch_freqs"])
    metrics = channel_metrics(raw, pre["bads"]["lf_band"], pre["bads"]["hf_band"])
    bads = find_bad_channels(metrics, pre["bads"]["z_thresh"], pre["bads"]["min_corr"])
    raw.info["bads"] = list(bads)
    average_reference(raw)
    return raw, {"eog_verdict": verdict, "dropped": dropped, "bads": bads, "metrics": metrics}