from pathlib import Path

import mne
from mne_bids import BIDSPath, read_raw_bids


def bids_path(bids_dir: Path, subject: str, run: str, task: str) -> BIDSPath:
    return BIDSPath(root=bids_dir, subject=subject, task=task, run=run, datatype="eeg")


def load_raw(bids_dir: Path, subject: str, run: str, task: str, eog: list[str]) -> mne.io.BaseRaw:
    raw = read_raw_bids(bids_path(bids_dir, subject, run, task), verbose="error")
    raw.load_data()
    # channels.tsv lists every type as n/a, so set them here
    types = {ch: ("eog" if ch in eog else "eeg") for ch in raw.ch_names}
    raw.set_channel_types(types, on_unit_change="ignore")
    return raw