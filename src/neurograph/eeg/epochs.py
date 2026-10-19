import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject, RejectLog

EVENT_ID = {"closed": 1, "open": 2}


def block_epochs(raw: mne.io.BaseRaw, blocks: pd.DataFrame, length_s: float,
                 pad_s: float) -> mne.Epochs:
    sf = raw.info["sfreq"]
    onsets, codes = [], []
    for _, b in blocks.iterrows():
        starts = np.arange(b["start_s"], b["end_s"] + pad_s - length_s + 1e-9, length_s)
        onsets.extend(starts)
        codes.extend([EVENT_ID[b["eyes"]]] * len(starts))
    samples = np.round(np.asarray(onsets) * sf).astype(int)  # absolute samples (file starts at 0)
    events = np.column_stack([samples, np.zeros_like(samples), np.asarray(codes)])
    return mne.Epochs(raw, events, event_id=EVENT_ID, tmin=0.0, tmax=length_s - 1.0 / sf,
                      baseline=None, preload=True, reject_by_annotation=False, verbose="error")


def run_autoreject(epochs: mne.Epochs, cfg: dict) -> tuple[mne.Epochs, RejectLog]:
    ar = AutoReject(n_interpolate=np.array(cfg["n_interpolate"]),
                    consensus=np.array(cfg["consensus"]), cv=cfg["cv"],
                    random_state=cfg["random_state"], n_jobs=cfg["n_jobs"], verbose=False)
    return ar.fit_transform(epochs, return_log=True)


def count_conditions(epochs: mne.Epochs) -> dict[str, int]:
    codes = epochs.events[:, 2]
    return {name: int((codes == code).sum()) for name, code in EVENT_ID.items()}


def n_repaired(log: RejectLog) -> int:
    labels = np.nan_to_num(log.labels[~log.bad_epochs])
    return int((labels == 2).any(axis=1).sum())  # 2 = channel interpolated in that epoch


def repair_by_channel(log: RejectLog, top: int = 5) -> dict[str, float]:
    labels = np.nan_to_num(log.labels[~log.bad_epochs])
    pct = (labels == 2).mean(axis=0) * 100  # % of kept epochs where the channel was rebuilt
    order = np.argsort(pct)[::-1][:top]
    return {log.ch_names[i]: round(float(pct[i]), 1) for i in order if pct[i] > 0}


def alpha_ratio(epochs: mne.Epochs, ch: str = "O1", band: tuple[float, float] = (8.0, 13.0)) -> float:
    spec = epochs.compute_psd(method="welch", fmin=1.0, fmax=45.0, picks=[ch], verbose="error")
    psd, freqs = spec.get_data(return_freqs=True)
    power = psd[:, 0, (freqs >= band[0]) & (freqs <= band[1])].mean(axis=1)
    codes = epochs.events[:, 2]
    return float(power[codes == EVENT_ID["closed"]].mean() / power[codes == EVENT_ID["open"]].mean())
