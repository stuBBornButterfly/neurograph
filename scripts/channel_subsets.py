import glob
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.features import (alpha_peak, band_power, faa, find_input_derivatives, region_map,
                                     regional_row, subject_psd)
from neurograph.models.evaluate import cross_validate, make_splits
from scripts.run_baselines import design


def layout_row(ep: mne.Epochs, chs: list[str], reref: bool, f: dict) -> dict[str, float]:
    ep = ep.copy().pick(chs)
    if reref:  # a smaller cap's average reference only averages its own channels
        ep.set_eeg_reference("average", verbose="error")
    p = f["psd"]
    psd, freqs = subject_psd(ep, p["fmin"], p["fmax"], p["n_fft"])  # (condition, channel, freq)
    absolute, relative = band_power(psd, freqs, f["bands"])  # (condition, channel, band)
    bands = list(f["bands"])
    iaf_ch = [chs.index(c) for c in f["iaf"]["channels"] if c in chs]
    hz, db = alpha_peak(psd[0, iaf_ch].mean(axis=0), freqs, f["iaf"]["range"])
    pairs = [pr for pr in f["faa_pairs"] if set(pr) <= set(chs)]
    row = {"iaf_hz": np.nan if db < f["iaf_min_peak_db"] else hz}
    row.update(faa(absolute[:, :, bands.index("alpha")], chs, pairs))
    row.update(regional_row(absolute, relative, chs, region_map(chs, f["regions"]), bands))
    return row


def plot_channels(infos: dict, summary: pd.DataFrame, models: list[str], out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(infos) + 1, figsize=(4 * len(infos) + 6, 4))
    for ax, (name, info) in zip(axes, infos.items()):
        mne.viz.plot_sensors(info, axes=ax, show=False, show_names=len(info.ch_names) <= 32)
        ax.set_title(f"{name}: {len(info.ch_names)} channels")
    x = np.arange(len(infos))
    w = 0.8 / len(models)
    for k, mod in enumerate(models):
        s = summary[summary["model"] == mod].set_index("layout").loc[list(infos)]
        axes[-1].bar(x + k * w, s["auc"], w, yerr=s["sd"], capsize=3, label=mod)
    axes[-1].axhline(0.5, c="grey", ls="--", lw=1)
    axes[-1].set_xticks(x + w * (len(models) - 1) / 2, list(infos))
    axes[-1].set(ylim=(0.3, 0.9), ylabel="AUC (mean ± SD, 50 folds)", title="AUC by channel layout")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    t0 = time.perf_counter()
    paths, m, f = load_config("paths"), load_config("models"), load_config("features")
    c, cv = m["channels"], m["cv"]
    in_dir = find_input_derivatives(m["input_glob"]).parent
    out = Path(paths["paths"]["derivatives_dir"])
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(in_dir / "labels.csv", dtype={"sub": str})
    subs = labels["sub"].tolist()
    files = {Path(p).name.split("_")[0][4:]: p for p in glob.glob(c["epochs_glob"], recursive=True)}
    missing = [s for s in subs if s not in files]
    if missing:
        raise FileNotFoundError(f"no epochs for {len(missing)} people {missing[:5]} | {len(files)} epoch files match "
                                f"{c['epochs_glob']}; attach neurograph-20-preprocess Version 2")

    rows = {name: [] for name in c["layouts"]}
    infos = {}
    for k, s in enumerate(subs, 1):
        ep = mne.read_epochs(files[s], verbose="error")
        for name, chs in c["layouts"].items():
            use = ep.ch_names if chs is None else chs
            absent = sorted(set(use) - set(ep.ch_names))
            if absent:
                raise ValueError(f"layout {name}: channels not in the recording: {absent}")
            rows[name].append({"sub": s, **layout_row(ep, list(use), chs is not None, f)})
            infos.setdefault(name, ep.copy().pick(list(use)).info)
        if k % 30 == 0 or k == len(subs):
            print(f"  features {k}/{len(subs)} | {time.perf_counter() - t0:.0f} s", flush=True)

    print(f"== step 4.5 channel subsets | commit {git_hash()} ==")
    ref = pd.read_csv(in_dir / "features" / "features.csv", dtype={"sub": str}).set_index("sub").loc[subs]
    feats = {}
    for name, r in rows.items():
        feats[name] = pd.DataFrame(r)
        feats[name].to_csv(out / "features" / f"features_{name}.csv", index=False)
    full = feats["full"].set_index("sub")
    common = [col for col in ref.columns if col in full.columns]
    diff = float(np.nanmax(np.abs(full[common].to_numpy() - ref[common].to_numpy())))
    print(f"sanity: 'full' layout vs step 3 features.csv: {len(common)} columns, max |diff| {diff:.2e} "
          f"(should be ~0)")

    y = labels["y_dep"].to_numpy()
    splits = make_splits(y, labels["sub"].to_numpy(), cv["n_splits"], cv["n_repeats"], cv["random_state"])
    folds, summary = [], []
    for name, chs in c["layouts"].items():
        df = labels[["sub", "group", "y_dep", "sex"]].merge(feats[name], on="sub", validate="one_to_one")
        df = df.assign(female=(df["sex"] == 1).astype(int))
        n_feat_ch = sum(len(v) for v in region_map(list(infos[name].ch_names), f["regions"]).values())
        for mod, spec in c["models"].items():
            X, cols = design(df, spec, m["feature_sets"])
            res, _ = cross_validate(X, y, splits, spec["kind"], m, m["threshold"])
            folds.append(res.assign(layout=name, model=mod))
            summary.append({"layout": name, "model": mod, "channels": len(infos[name].ch_names),
                            "feature_channels": n_feat_ch, "inputs": len(cols), "auc": res["auc"].mean(),
                            "sd": res["auc"].std()})
    folds, summary = pd.concat(folds), pd.DataFrame(summary)
    folds.to_csv(out / "models" / "channels_folds.csv", index=False)

    print(f"\n{'layout':6s} {'chans':>5s} {'in features':>11s}  " + "  ".join(f"{mod:>18s}" for mod in c["models"]))
    for name in c["layouts"]:
        s = summary[summary.layout == name].set_index("model")
        first = s.iloc[0]
        print(f"{name:6s} {first['channels']:5d} {first['feature_channels']:11d}  "
              + "  ".join(f"{s.loc[mod, 'auc']:.3f} ± {s.loc[mod, 'sd']:.3f}".rjust(18) for mod in c["models"]))
    print("(in features: channels left after dropping frontal-pole and mastoid sites; AUC over the same 50 folds)")

    print("\n-- paired vs full cap, same 50 folds --")
    key = ["repeat", "fold"]
    for name in [n for n in c["layouts"] if n != "full"]:
        for mod in c["models"]:
            a = folds[(folds.layout == name) & (folds.model == mod)].set_index(key)["auc"]
            b = folds[(folds.layout == "full") & (folds.model == mod)].set_index(key)["auc"]
            d = a - b
            print(f"  {name} - full | {mod:18s} ΔAUC {d.mean():+.3f} ± {d.std():.3f} | "
                  f"smaller cap better in {(d > 0).mean():.0%} of folds")

    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "channels_auc.png"
    plot_channels(infos, summary, list(c["models"]), fig_path)
    print(f"\ntime {time.perf_counter() - t0:.0f} s")
    print(f"saved: {out}/features/features_<layout>.csv | models/channels_folds.csv | "
          f"figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
