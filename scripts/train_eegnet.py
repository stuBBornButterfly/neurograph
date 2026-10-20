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
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from neurograph.config import REPO_ROOT, git_hash, load_config
from neurograph.eeg.features import find_input_derivatives
from neurograph.models.eegnet import predict_epochs, subject_scores, train_fold
from neurograph.models.evaluate import cross_validate, make_splits
from scripts.run_baselines import design, load_table


def load_epochs(files: dict[str, str], subs: list[str], cond: str,
                exclude: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X, owner, chs = [], [], None
    t0 = time.perf_counter()
    for k, s in enumerate(subs, 1):
        ep = mne.read_epochs(files[s], verbose="error")[cond]
        if chs is None:
            chs = [c for c in ep.ch_names if c not in exclude]
        X.append((ep.get_data(picks=chs) * 1e6).astype(np.float32))  # V -> µV, same channel order
        owner.extend([s] * len(ep))
        if k % 30 == 0 or k == len(subs):
            print(f"  loaded {k}/{len(subs)} | {time.perf_counter() - t0:.0f} s", flush=True)
    return np.concatenate(X), np.array(owner), chs


def plot_eegnet(curves: list[pd.DataFrame], folds: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for h in curves:
        axes[0].plot(h["epoch"], h["train_loss"], c="tab:blue", alpha=0.5, lw=1)
        axes[0].plot(h["epoch"], h["val_loss"], c="tab:orange", alpha=0.7, lw=1)
    axes[0].plot([], [], c="tab:blue", label="train loss")
    axes[0].plot([], [], c="tab:orange", label="validation loss (held-out people)")
    axes[0].set(title="EEGNet learning curves (repeat 0)", xlabel="training epoch", ylabel="weighted BCE")
    axes[0].legend(fontsize=8)
    axes[1].scatter(folds["auc_lr"], folds["auc"], s=18)
    axes[1].plot([0.2, 1], [0.2, 1], c="grey", ls="--", lw=1)
    axes[1].set(title="per-fold AUC: EEGNet vs band-power LR", xlabel="logistic regression (eyes closed)",
                ylabel="EEGNet (eyes closed)", xlim=(0.2, 1), ylim=(0.2, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    t0 = time.perf_counter()
    paths, m = load_config("paths"), load_config("models")
    e, cv = m["eegnet"], m["cv"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    in_dir = find_input_derivatives(m["input_glob"]).parent
    out = Path(paths["paths"]["derivatives_dir"]) / "models"
    out.mkdir(parents=True, exist_ok=True)
    df = load_table(in_dir)  # same people, same order as steps 4.1-4.3
    subs, y = df["sub"].to_numpy(), df["y_dep"].to_numpy()
    files = {Path(f).name.split("_")[0][4:]: f for f in glob.glob(e["epochs_glob"], recursive=True)}
    missing = [s for s in subs if s not in files]
    if missing:
        raise FileNotFoundError(f"no epochs for {missing[:5]}...; attach neurograph-20-preprocess Version 2")

    print(f"== step 4.4 EEGNet | commit {git_hash()} | device {dev}"
          f"{' ' + torch.cuda.get_device_name(0) if dev.type == 'cuda' else ''} | torch {torch.__version__} ==")
    X_np, owner, chs = load_epochs(files, list(subs), e["condition"], e["exclude_channels"])
    y_of = dict(zip(subs, y))
    y_ep = np.array([y_of[s] for s in owner], dtype=np.float32)
    print(f"eyes-{e['condition']} epochs {X_np.shape[0]} | channels {len(chs)} | samples {X_np.shape[2]} "
          f"| per person median {int(np.median(pd.Series(owner).value_counts()))} | {X_np.nbytes / 1e9:.2f} GB")
    X = torch.from_numpy(X_np).to(dev)
    y_t = torch.from_numpy(y_ep).to(dev)
    del X_np

    splits = [s for s in make_splits(y, subs, cv["n_splits"], cv["n_repeats"], cv["random_state"])
              if s[0] < e["n_repeats"]]  # identical to the first repeats of steps 4.1-4.3
    spec = m["explain"]["models"][e["compare"]]
    Xf, _ = design(df, spec, m["feature_sets"])
    lr_res, _ = cross_validate(Xf, y, splits, spec["kind"], m, m["threshold"])
    lr_auc = lr_res.set_index(["repeat", "fold"])["auc"]

    rows, oof, curves = [], [], []
    for r, f, tr, te in splits:
        t1 = time.perf_counter()
        tr_sub, va_sub = train_test_split(subs[tr], test_size=e["val_frac"], stratify=y[tr],
                                          random_state=e["seed"] + 10 * r + f)
        ep_tr, ep_va = np.where(np.isin(owner, tr_sub))[0], np.where(np.isin(owner, va_sub))[0]
        ep_te = np.where(np.isin(owner, subs[te]))[0]
        model, mu, sd, hist = train_fold(X, y_t, owner, ep_tr, ep_va, e, e["seed"] + 10 * r + f)
        p_ep = predict_epochs(model, X, torch.as_tensor(ep_te, device=dev), mu, sd)
        s_te, p_te = subject_scores(p_ep, owner[ep_te])
        y_te = np.array([y_of[s] for s in s_te])
        h = pd.DataFrame(hist)
        best = int(h["val_loss"].idxmin())
        rows.append({"repeat": r, "fold": f, "auc": roc_auc_score(y_te, p_te),
                     "auc_epoch": roc_auc_score(y_ep[ep_te], p_ep), "auc_lr": lr_auc[(r, f)],
                     "best_epoch": best, "epochs_run": len(h), "seconds": time.perf_counter() - t1})
        oof.append(pd.DataFrame({"repeat": r, "sub": s_te, "y": y_te, "p_dep": p_te}))
        if r == 0:
            curves.append(h)
        x = rows[-1]
        print(f"  r{r} f{f} | train {len(tr_sub)} people ({len(ep_tr)} epochs), val {len(va_sub)}, test {len(te)} "
              f"| best epoch {best}/{len(h)} | AUC person {x['auc']:.3f} (LR {x['auc_lr']:.3f}) "
              f"epoch-level {x['auc_epoch']:.3f} | {x['seconds']:.0f} s", flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(out / "eegnet_folds.csv", index=False)
    pd.concat(oof).to_csv(out / "eegnet_oof.csv", index=False)

    d = folds["auc"] - folds["auc_lr"]
    print(f"\n-- {len(folds)} folds ({e['n_repeats']} repeats x {cv['n_splits']}), split by person --")
    print(f"EEGNet (raw eyes-{e['condition']} EEG)   AUC {folds['auc'].mean():.3f} ± {folds['auc'].std():.3f} "
          f"| epoch-level {folds['auc_epoch'].mean():.3f}")
    print(f"{e['compare']} (band power)  AUC {folds['auc_lr'].mean():.3f} ± {folds['auc_lr'].std():.3f}")
    print(f"EEGNet - LR: ΔAUC {d.mean():+.3f} ± {d.std():.3f} | EEGNet better in {(d > 0).mean():.0%} of folds")
    print(f"best epoch median {folds['best_epoch'].median():.0f} | stopped early in "
          f"{(folds['epochs_run'] < e['max_epochs']).mean():.0%} of folds")
    print("leakage checklist: folds by person ✓ | channel scaling from training epochs only ✓ | "
          "early stopping on held-out training people ✓ | test people never seen ✓")
    fig_path = REPO_ROOT / paths["paths"]["figures_dir"] / "eegnet.png"
    plot_eegnet(curves, folds, fig_path)
    print(f"time {time.perf_counter() - t0:.0f} s")
    print(f"saved: {out}/eegnet_folds.csv | eegnet_oof.csv | figure: {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
