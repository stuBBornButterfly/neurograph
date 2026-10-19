import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne_icalabel import label_components


def fit_ica(raw: mne.io.BaseRaw, n_components: int, method: str, decim: int,
            seed: int) -> tuple[ICA, float]:
    fit_params = dict(ortho=False, extended=True) if method == "picard" else dict(extended=True)
    ica = ICA(n_components=n_components, method=method, fit_params=fit_params,
              random_state=seed, max_iter="auto")
    t0 = time.perf_counter()
    ica.fit(raw, picks="eeg", decim=decim, verbose="error")
    return ica, time.perf_counter() - t0


def label_ica(raw: mne.io.BaseRaw, ica: ICA) -> pd.DataFrame:
    out = label_components(raw, ica, method="iclabel")
    return pd.DataFrame({"label": out["labels"], "prob": out["y_pred_proba"]},
                        index=[f"IC{i:02d}" for i in range(ica.n_components_)])


def pick_exclude(labels: pd.DataFrame, reject: list[str], min_prob: float) -> list[int]:
    mask = labels["label"].isin(reject) & (labels["prob"] >= min_prob)
    return [int(i) for i in np.where(mask.to_numpy())[0]]


def eog_check(raw: mne.io.BaseRaw, ica: ICA, ch: str = "VEOG") -> tuple[list[int], int, float]:
    inds, scores = ica.find_bads_eog(raw, ch_name=ch, verbose="error")
    scores = np.abs(np.asarray(scores)).ravel()
    top = int(np.argmax(scores))
    return [int(i) for i in inds], top, float(scores[top])


def plot_ica_topos(ica: ICA, labels: pd.DataFrame, exclude: list[int], out_path: Path,
                   ncols: int = 6) -> None:
    comps = ica.get_components()
    n = comps.shape[1]
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2.2 * nrows))
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.axis("off")
            continue
        mne.viz.plot_topomap(comps[:, k], ica.info, axes=ax, show=False)
        lab = labels.iloc[k]
        ax.set_title(f"IC{k:02d} {lab['label'][:11]} {lab['prob']:.2f}", fontsize=7,
                     color="red" if k in exclude else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)