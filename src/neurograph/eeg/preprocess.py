import mne
import numpy as np
import pandas as pd

from neurograph.eeg.qc import eog_swap_check, eog_verdict


def swap_eog(raw: mne.io.BaseRaw) -> None:
    raw.rename_channels({"HEOG": "VEOG", "VEOG": "HEOG"})


def fix_eog_labels(raw: mne.io.BaseRaw) -> str:
    verdict = eog_verdict(eog_swap_check(raw))
    if verdict == "looks SWAPPED":
        swap_eog(raw)
    return verdict


def set_template_montage(raw: mne.io.BaseRaw, montage: str, drop: list[str]) -> list[str]:
    dropped = [ch for ch in drop if ch in raw.ch_names]
    raw.drop_channels(dropped)
    raw.set_montage(mne.channels.make_standard_montage(montage), match_case=False)
    return dropped


def filter_raw(raw: mne.io.BaseRaw, l_freq: float, h_freq: float, notch: list[float]) -> None:
    raw.filter(l_freq, h_freq, picks=["eeg", "eog"], verbose="error")
    raw.notch_filter(notch, picks=["eeg", "eog"], verbose="error")


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    return (x - med) / (np.median(np.abs(x - med)) * 1.4826)


def channel_metrics(raw: mne.io.BaseRaw, lf: list[float], hf: list[float]) -> pd.DataFrame:
    eeg = raw.copy().pick("eeg")
    x = eeg.get_data()
    x = x - np.median(x, axis=0)  # robust reference so near-reference channels don't look flat
    psd, freqs = mne.time_frequency.psd_array_welch(x, eeg.info["sfreq"], fmin=lf[0], fmax=hf[1],
                                                    n_fft=1000, verbose="error")
    low = (freqs >= lf[0]) & (freqs < lf[1])
    high = (freqs >= hf[0]) & (freqs <= hf[1])
    hf_pow = psd[:, high].mean(axis=1)
    ratio = np.log(psd[:, high].sum(axis=1) / psd[:, low].sum(axis=1))
    corr = np.abs(np.corrcoef(x))
    np.fill_diagonal(corr, 0)
    sd = x.std(axis=1)
    return pd.DataFrame({"sd_uv": sd * 1e6, "z_sd": _robust_z(np.log(sd)),
                         "z_hf": _robust_z(ratio), "hf_db": 10 * np.log10(hf_pow * 1e12),
                         "z_hfpow": _robust_z(np.log(hf_pow)), "max_corr": corr.max(axis=1)},
                        index=eeg.ch_names)


def find_bad_channels(metrics: pd.DataFrame, z_thresh: float, min_corr: float) -> dict[str, str]:
    bads = {}
    for ch, m in metrics.iterrows():
        reasons = []
        if abs(m["z_sd"]) > z_thresh:
            reasons.append(f"amplitude z={m['z_sd']:+.1f}")
        if m["z_hf"] > z_thresh:
            reasons.append(f"high-freq ratio z={m['z_hf']:+.1f}")
        if m["z_hfpow"] > z_thresh:
            reasons.append(f"high-freq power z={m['z_hfpow']:+.1f}")
        if m["max_corr"] < min_corr:
            reasons.append(f"low corr {m['max_corr']:.2f}")
        if reasons:
            bads[ch] = ", ".join(reasons)
    return bads

def average_reference(raw: mne.io.BaseRaw) -> None:
    raw.set_eeg_reference("average", projection=False, verbose="error")  # skips info['bads']