from pathlib import Path

import mne
import pandas as pd
from mne_bids import BIDSPath, read_raw_bids


def bids_path(bids_dir: Path, subject: str, run: str, task: str) -> BIDSPath:
    return BIDSPath(root=bids_dir, subject=subject, task=task, run=run, datatype="eeg")


def load_raw(bids_dir: Path, subject: str, run: str, task: str, eog: list[str]) -> mne.io.BaseRaw:
    raw = read_raw_bids(bids_path(bids_dir, subject, run, task), verbose="error")
    raw.load_data(verbose="error")
    # channels.tsv lists every type as n/a, so set them here
    types = {ch: ("eog" if ch in eog else "eeg") for ch in raw.ch_names}
    raw.set_channel_types(types, on_unit_change="ignore")
    return raw


def read_events(bids_dir: Path, subject: str, run: str, task: str) -> pd.DataFrame:
    bp = bids_path(bids_dir, subject, run, task).update(suffix="events", extension=".tsv")
    return pd.read_csv(bp.fpath, sep="\t")


def crop_to_blocks(raw: mne.io.BaseRaw, blocks: pd.DataFrame, pad_s: float) -> mne.io.BaseRaw:
    # drops setup time before the first block and end-of-recording junk
    tmax = min(float(blocks["end_s"].max()) + pad_s, raw.times[-1])
    return raw.crop(tmin=float(blocks["start_s"].min()), tmax=tmax)