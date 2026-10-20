import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.features import (CONDITIONS, alpha_peak, band_power, feature_row,
                                     find_input_derivatives, subject_psd)


def load_psds(in_dir: Path, subs: list[str], run: str, p: dict):
    psds, counts, chs, info = [], [], None, None
    t0 = time.perf_counter()
    for k, s in enumerate(subs, 1):
        ep = mne.read_epochs(in_dir / "epochs" / f"sub-{s}_run-{run}_epo.fif", verbose="error")
        if chs is None:
            chs, info = ep.ch_names, ep.info
        ep.reorder_channels(chs)  # same channel order for everyone (raises if one is missing)
        psd, freqs = subject_psd(ep, p["fmin"], p["fmax"], p["n_fft"])
        psds.append(psd)
        counts.append({f"n_{c}": len(ep[c]) for c in CONDITIONS})
        if k % 20 == 0 or k == len(subs):
            print(f"  loaded {k}/{len(subs)} | {time.perf_counter() - t0:.0f} s", flush=True)
    return np.stack(psds), freqs, chs, info, counts


def db(x: np.ndarray) -> np.ndarray:
    return 10 * np.log10(x * 1e12)  # V²/Hz -> dB µV²/Hz


def plot_group_psd(psd, freqs, chs, groups, chans, out_path):
    fig, axes = plt.subplots(1, len(chans), figsize=(4.2 * len(chans), 3.4), sharey=True)
    for ax, ch in zip(axes, chans):
        for g, color in [("CTL", "tab:blue"), ("DEP", "tab:red")]:
            x = db(psd[groups == g, 0, chs.index(ch)])  # eyes closed
            m, se = x.mean(axis=0), x.std(axis=0) / np.sqrt(len(x))
            ax.plot(freqs, m, c=color, label=f"{g} (n={len(x)})")
            ax.fill_between(freqs, m - se, m + se, color=color, alpha=0.2)
        ax.axvspan(8, 13, color="green", alpha=0.08)
        ax.set(title=f"{ch} eyes closed", xlabel="Frequency (Hz)")
    axes[0].set_ylabel("Power (dB µV²/Hz)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_alpha_sanity(psd, freqs, chs, info, rel_alpha_closed, iaf, subs, inspect, out_path):
    n = 2 + len(inspect)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4))
    axes[0].hist(iaf, bins=np.arange(6.75, 14.5, 0.5), color="tab:blue")
    axes[0].set(title="alpha peak (O1/OZ/O2, closed)", xlabel="Hz", ylabel="subjects")
    mne.viz.plot_topomap(rel_alpha_closed.mean(axis=0), info, axes=axes[1], show=False)
    axes[1].set_title("relative alpha, closed (mean)")
    oz = chs.index("OZ")
    for ax, s in zip(axes[2:], inspect):
        i = subs.index(s)
        for c, color in [(0, "tab:blue"), (1, "tab:orange")]:
            ax.plot(freqs, db(psd[i, c, oz]), c=color, label=f"eyes {CONDITIONS[c]}")
        ax.axvspan(8, 13, color="green", alpha=0.08)
        ax.set(title=f"sub-{s} OZ", xlabel="Frequency (Hz)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    t0 = time.perf_counter()
    paths, eeg, f = load_config("paths"), load_config("eeg"), load_config("features")
    in_dir = find_input_derivatives(f["input_glob"])
    out = Path(paths["paths"]["derivatives_dir"])
    (out / "features").mkdir(parents=True, exist_ok=True)
    if (in_dir / "labels.csv").resolve() != (out / "labels.csv").resolve():  # local runs: input = output folder
        shutil.copy(in_dir / "labels.csv", out / "labels.csv")  # keep labels next to the features
    labels = pd.read_csv(out / "labels.csv", dtype={"sub": str})
    subs = labels["sub"].tolist()

    psd, freqs, chs, info, counts = load_psds(in_dir, subs, eeg["main_run"], f["psd"])
    absolute, relative = band_power(psd, freqs, f["bands"])  # (subject, condition, channel, band)
    bands = list(f["bands"])
    iaf_idx = [chs.index(c) for c in f["iaf"]["channels"]]
    peaks = [alpha_peak(psd[i, 0, iaf_idx].mean(axis=0), freqs, f["iaf"]["range"]) for i in range(len(subs))]

    rows = []
    for i, s in enumerate(subs):
        row = {"sub": s, "group": labels.loc[i, "group"], **counts[i],
               "iaf_hz": peaks[i][0], "iaf_peak_db": round(peaks[i][1], 2)}
        row.update(feature_row(absolute[i], relative[i], chs, bands, f["exclude_channels"]))
        rows.append(row)
    feats = pd.DataFrame(rows)
    feats.to_csv(out / "features" / "band_power.csv", index=False)
    np.savez_compressed(out / "features" / "psd.npz", psd=psd.astype(np.float32), freqs=freqs,
                        ch_names=np.array(chs), subjects=np.array(subs), conditions=np.array(CONDITIONS))

    fig_dir = REPO_ROOT / paths["paths"]["figures_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    groups = labels["group"].to_numpy()
    plot_group_psd(psd, freqs, chs, groups, ["F3", "F4", "CZ", "OZ"], fig_dir / "psd_group_closed.png")
    a = bands.index("alpha")
    iaf = np.array([p[0] for p in peaks])
    plot_alpha_sanity(psd, freqs, chs, info, relative[:, 0, :, a], iaf, subs, f["sanity"]["inspect"],
                      fig_dir / "alpha_sanity.png")

    # sanity numbers
    num = feats.select_dtypes("number")
    post = [chs.index(c) for c in f["sanity"]["posterior"]]
    front = [chs.index(c) for c in f["sanity"]["frontal"]]
    alpha_c, alpha_o = absolute[:, 0, :, a], absolute[:, 1, :, a]
    n_feat = sum(c.startswith(tuple(CONDITIONS)) for c in feats.columns)
    print(f"== step 3.1 PSD + band power | commit {git_hash()} | input {in_dir} ==")
    print(f"subjects {len(subs)} | channels {len(chs)} (features use {len(chs) - len(f['exclude_channels'])}, "
          f"excluded {f['exclude_channels']}) | freqs {freqs[0]}-{freqs[-1]} Hz step {freqs[1] - freqs[0]}")
    print(f"feature matrix: {feats.shape} | EEG features {n_feat} | NaN {int(num.isna().sum().sum())} | "
          f"inf {int(np.isinf(num.to_numpy()).sum())}")
    print(f"epochs per subject: closed median {feats['n_closed'].median():.0f} (min {feats['n_closed'].min()}) | "
          f"open median {feats['n_open'].median():.0f} (min {feats['n_open'].min()})")
    print(f"alpha peak (IAF): median {np.median(iaf):.1f} Hz | IQR {np.percentile(iaf, 25):.1f}-"
          f"{np.percentile(iaf, 75):.1f} | in 8-12 Hz: {np.mean((iaf >= 8) & (iaf <= 12)):.0%} | "
          f"weak peak (< 1 dB above flanks): {int((feats['iaf_peak_db'] < 1).sum())}")
    print(f"alpha back > front (closed): {np.mean(alpha_c[:, post].mean(1) > alpha_c[:, front].mean(1)):.0%} of subjects")
    print(f"alpha closed > open at OZ (Berger): {np.mean(alpha_c[:, chs.index('OZ')] > alpha_o[:, chs.index('OZ')]):.0%}")
    rel = relative[:, 0, chs.index("OZ")]
    print(f"relative power at OZ, closed (median): " + ", ".join(f"{b} {np.median(rel[:, k]):.2f}" for k, b in enumerate(bands)))
    for s in f["sanity"]["inspect"]:
        i = subs.index(s)
        po = alpha_peak(psd[i, 1, iaf_idx].mean(axis=0), freqs, f["iaf"]["range"])
        oz = chs.index("OZ")
        print(f"  sub-{s}: peak closed {peaks[i][0]:.1f} Hz ({peaks[i][1]:+.1f} dB) | open {po[0]:.1f} Hz "
              f"({po[1]:+.1f} dB) | alpha OZ closed/open {alpha_c[i, oz] / alpha_o[i, oz]:.2f}x")
    print(f"time {time.perf_counter() - t0:.0f} s")
    for name in ["features/band_power.csv", "features/psd.npz"]:
        print(f"saved: {out / name} ({(out / name).stat().st_size / 1e6:.1f} MB)")
    print("figures: reports/figures/psd_group_closed.png | reports/figures/alpha_sanity.png")


if __name__ == "__main__":
    main()
