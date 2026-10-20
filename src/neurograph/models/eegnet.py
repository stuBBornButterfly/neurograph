import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class EEGNet(nn.Module):
    # Lawhern et al. 2018: temporal filters -> spatial filter per temporal filter -> separable conv
    def __init__(self, n_chans: int, n_times: int, f1: int = 8, d: int = 2, f2: int = 16,
                 kern_len: int = 125, dropout: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, kern_len), padding=(0, kern_len // 2), bias=False),  # frequency filters
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (n_chans, 1), groups=f1, bias=False),  # scalp maps, one set per filter
            nn.BatchNorm2d(f1 * d), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, (1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, 1, bias=False),  # mix the filtered maps
            nn.BatchNorm2d(f2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout))
        with torch.no_grad():
            n_out = self.block2(self.block1(torch.zeros(1, 1, n_chans, n_times))).numel()
        self.head = nn.Linear(n_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channel, time)
        z = self.block2(self.block1(x.unsqueeze(1)))
        return self.head(z.flatten(1)).squeeze(1)  # logit of DEP


def channel_stats(X: torch.Tensor, idx: torch.Tensor, chunk: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    # per-channel mean/SD over the training epochs only (in chunks to save GPU memory)
    s = torch.zeros(X.shape[1], device=X.device, dtype=torch.float64)
    ss = torch.zeros_like(s)
    for k in range(0, len(idx), chunk):
        xb = X[idx[k:k + chunk]].double()
        s += xb.sum(dim=(0, 2))
        ss += (xb ** 2).sum(dim=(0, 2))
    n = len(idx) * X.shape[2]
    mu = s / n
    sd = (ss / n - mu ** 2).clamp_min(1e-12).sqrt()
    return mu.float()[None, :, None], sd.float()[None, :, None]


def predict_epochs(model: nn.Module, X: torch.Tensor, idx: torch.Tensor, mu: torch.Tensor,
                   sd: torch.Tensor, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(idx), batch):
            out.append(torch.sigmoid(model((X[idx[k:k + batch]] - mu) / sd)).cpu())
    return torch.cat(out).numpy()


def subject_scores(p_epoch: np.ndarray, subj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # one P(DEP) per person = mean over that person's epochs
    subs = np.unique(subj)
    return subs, np.array([p_epoch[subj == s].mean() for s in subs])


def train_fold(X: torch.Tensor, y_ep: torch.Tensor, subj_ep: np.ndarray, tr: np.ndarray, va: np.ndarray,
               cfg: dict, seed: int) -> tuple[nn.Module, torch.Tensor, torch.Tensor, list[dict]]:
    # tr / va: epoch indices of training and validation people; early stopping on validation loss
    torch.manual_seed(seed)
    dev = X.device
    tr_t, va_t = torch.as_tensor(tr, device=dev), torch.as_tensor(va, device=dev)
    mu, sd = channel_stats(X, tr_t)
    model = EEGNet(X.shape[1], X.shape[2], cfg["f1"], cfg["d"], cfg["f2"], cfg["kern_len"],
                   cfg["dropout"]).to(dev)
    y_np = y_ep.cpu().numpy()
    pos = float(y_np[tr].mean())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))  # balance DEP/CTL
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, wait, history = np.inf, None, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        order = tr_t[torch.randperm(len(tr_t), generator=gen).to(dev)]
        total = 0.0
        for k in range(0, len(order), cfg["batch_size"]):
            b = order[k:k + cfg["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model((X[b] - mu) / sd), y_ep[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            va_loss = sum(loss_fn(model((X[va_t[k:k + 256]] - mu) / sd), y_ep[va_t[k:k + 256]]).item()
                          * len(va_t[k:k + 256]) for k in range(0, len(va_t), 256)) / len(va_t)
        p_va = predict_epochs(model, X, va_t, mu, sd)
        s, ps = subject_scores(p_va, subj_ep[va])
        yv = np.array([y_np[subj_ep == q][0] for q in s])
        auc = roc_auc_score(yv, ps) if 0 < yv.sum() < len(yv) else np.nan
        history.append({"epoch": ep, "train_loss": total / len(tr_t), "val_loss": va_loss, "val_auc": auc})
        if va_loss < best - 1e-4:
            best, best_state, wait = va_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, mu, sd, history
