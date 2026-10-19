import glob
from pathlib import Path

import mne
import numpy as np

from neurograph.eeg.epochs import EVENT_ID

CONDITIONS = list(EVENT_ID)  # ["closed", "open"]


def find_input_derivatives(pattern: str) -> Path:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no Phase 2 output matches {pattern}; attach it as notebook input")
    return Path(hits[0]).parent


def subject_psd(epochs: mne.Epochs, fmin: float, fmax: float,
                n_fft: int) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for cond in CONDITIONS:
        spec = epochs[cond].compute_psd(method="welch", fmin=fmin, fmax=fmax, n_fft=n_fft,
                                        n_overlap=0, verbose="error")
        psd, freqs = spec.get_data(return_freqs=True)
        out.append(psd.mean(axis=0))  # mean over epochs = Welch over the whole condition
    return np.stack(out), freqs       # (condition, channel, freq) in V²/Hz


def band_power(psd: np.ndarray, freqs: np.ndarray,
               bands: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    df = freqs[1] - freqs[0]
    absolute = np.stack([psd[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) * df * 1e12
                         for lo, hi in bands.values()], axis=-1)  # µV², (..., channel, band)
    total = psd.sum(axis=-1, keepdims=True) * df * 1e12
    return absolute, absolute / total


def feature_row(absolute: np.ndarray, relative: np.ndarray, ch_names: list[str],
                bands: list[str], exclude: list[str]) -> dict[str, float]:
    row = {}
    for c, cond in enumerate(CONDITIONS):
        for b, band in enumerate(bands):
            for i, ch in enumerate(ch_names):
                if ch in exclude:
                    continue
                row[f"{cond}_{band}_abs_{ch}"] = float(np.log10(absolute[c, i, b]))  # log µV²
                row[f"{cond}_{band}_rel_{ch}"] = float(relative[c, i, b])
    return row


def alpha_peak(psd_ch: np.ndarray, freqs: np.ndarray, rng: list[float]) -> tuple[float, float]:
    # peak frequency, and peak height above the flanks (dB) as a "is there a real peak" check
    m = (freqs >= rng[0]) & (freqs <= rng[1])
    k = np.argmax(psd_ch[m])
    flank = psd_ch[((freqs >= rng[0] - 2) & (freqs < rng[0])) | ((freqs > rng[1]) & (freqs <= rng[1] + 2))]
    return float(freqs[m][k]), float(10 * np.log10(psd_ch[m][k] / flank.mean()))
