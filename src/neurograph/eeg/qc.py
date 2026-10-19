from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd


def channel_std_uv(raw: mne.io.BaseRaw) -> pd.Series:
    eeg = raw.copy().pick("eeg").filter(1.0, 40.0, verbose="error")  # drop slow drift first
    return pd.Series(eeg.get_data().std(axis=1) * 1e6, index=eeg.ch_names)


def eyes_blocks(events: pd.DataFrame) -> pd.DataFrame:
    ev = events[events["value"].astype(str).isin([str(c) for c in range(11, 17)])].copy()
    ev["code"] = ev["value"].astype(int)
    ev["block"] = (ev["code"] != ev["code"].shift()).cumsum()
    out = ev.groupby("block").agg(code=("code", "first"), start_s=("onset", "min"),
                                  end_s=("onset", "max"), n=("onset", "size"))
    out["eyes"] = np.where(out["code"] % 2 == 1, "closed", "open")  # odd codes = closed
    return out.reset_index(drop=True)


def eog_swap_check(raw: mne.io.BaseRaw) -> dict[str, float]:
    names = ["FP1", "FP2", "F7", "F8", "HEOG", "VEOG"]
    filt = raw.copy().pick(names).filter(1.0, 10.0, picks="all", verbose="error")  # blink band
    x = {n: filt.get_data(picks=[n])[0] for n in names}
    fp, lr = (x["FP1"] + x["FP2"]) / 2, x["F7"] - x["F8"]
    r = lambda a, b: round(abs(float(np.corrcoef(a, b)[0, 1])), 2)
    return {"VEOG~FP": r(x["VEOG"], fp), "HEOG~FP": r(x["HEOG"], fp),
            "HEOG~F7-F8": r(x["HEOG"], lr), "VEOG~F7-F8": r(x["VEOG"], lr)}


def psd_summary(raw: mne.io.BaseRaw, line_freq: float) -> dict:
    spec = raw.compute_psd(method="welch", fmin=1.0, fmax=100.0, picks="eeg", exclude="bads",
                           n_fft=1000, verbose="error")  # 2 s windows -> 0.5 Hz steps
    psd, freqs = spec.get_data(return_freqs=True)
    o1 = psd[spec.ch_names.index("O1")]
    alpha = (freqs >= 7) & (freqs <= 14)
    mean = psd.mean(axis=0)
    side = ((freqs >= line_freq - 5) & (freqs <= line_freq - 2)) | ((freqs >= line_freq + 2) & (freqs <= line_freq + 5))
    line_db = 10 * np.log10(mean[np.argmin(abs(freqs - line_freq))] / mean[side].mean())
    return {"psd": psd, "freqs": freqs, "ch_names": spec.ch_names,
            "alpha_peak_o1": float(freqs[alpha][np.argmax(o1[alpha])]), "line_db": float(line_db)}


def plot_psd(psd: np.ndarray, freqs: np.ndarray, ch_names: list[str], out_path: Path,
             title: str, highlight: tuple[str, ...] = ("O1", "F3")) -> None:
    db = 10 * np.log10(psd * 1e12)  # V²/Hz -> dB µV²/Hz
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(freqs, db.T, color="0.8", lw=0.5)
    for ch in highlight:
        ax.plot(freqs, db[ch_names.index(ch)], lw=1.5, label=ch)
    ax.axvline(10, ls="--", c="green", lw=0.8)
    ax.axvline(60, ls="--", c="red", lw=0.8)
    ax.set(xlabel="Frequency (Hz)", ylabel="Power (dB µV²/Hz)", title=title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)



def eog_verdict(eog: dict[str, float], margin: float = 0.1) -> str:
    blink = eog["VEOG~FP"] - eog["HEOG~FP"]          # VEOG should win on blinks
    side = eog["HEOG~F7-F8"] - eog["VEOG~F7-F8"]     # HEOG should win on left-right moves
    if blink > margin and side > margin:
        return "labels look correct"
    if blink < -margin and side < -margin:
        return "looks SWAPPED"
    return "unclear"