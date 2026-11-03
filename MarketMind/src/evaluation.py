import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score
from .models import make_model, predict


def walk_forward(data, feats, name, task, params, start, end, refit_every):
    """Expanding-window walk-forward. For a block starting at row s, train on rows < s only
    (their targets are known by the close of s). Returns DataFrame indexed like `data` rows in [start,end]."""
    ycol = "y_dir" if task == "clf" else "y_ret"
    d = data.reset_index(drop=True)
    idx = d.index[(d["date"] >= start) & (d["date"] <= end)].to_numpy()
    preds, trainstat = [], []
    for b in range(0, len(idx), refit_every):
        blk = idx[b:b + refit_every]
        tr = d.iloc[: blk[0]]
        if task == "clf" and tr[ycol].nunique() < 2:
            continue
        m = make_model(name, task, **params).fit(tr[feats], tr[ycol])
        p = predict(m, task, d.loc[blk, feats])
        base = tr[ycol].mean()
        preds.append(pd.DataFrame({"row": blk, "pred": p, "train_mean": base, "n_train": len(tr)}))
    out = pd.concat(preds).set_index("row")
    out["date"] = d.loc[out.index, "date"].values
    out["y"] = d.loc[out.index, ycol].values
    out["ret_0"] = d.loc[out.index, "ret_0"].values
    out["y_ret"] = d.loc[out.index, "y_ret"].values
    return out


def clf_metrics(y, p, base_p):
    yhat = (p >= 0.5).astype(int)
    brier = np.mean((p - y) ** 2); brier_b = np.mean((base_p - y) ** 2)
    return {
        "accuracy": accuracy_score(y, yhat), "balanced_acc": balanced_accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat, zero_division=0),
        "brier": brier, "brier_skill": 1 - brier / brier_b, "share_pred_up": float(yhat.mean()),
        "confusion": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def reg_metrics(y, p, base_p):
    e = y - p
    return {
        "mae": np.mean(np.abs(e)), "rmse": np.sqrt(np.mean(e ** 2)),
        "oos_r2_vs_histmean": 1 - np.sum(e ** 2) / np.sum((y - base_p) ** 2),
        "oos_r2_vs_zero": 1 - np.sum(e ** 2) / np.sum(y ** 2), "mean_error": np.mean(e),
    }


def block_bootstrap_diff(loss_model, loss_base, block=10, n_boot=2000, seed=0):
    """Circular block bootstrap of mean(loss_model - loss_base). Negative = model better.
    Assumes weak dependence at the block scale; one-sided p = P(boot mean >= 0) after centring on 0 is NOT used --
    we report the share of resampled means >= 0 (uncentred), a conservative 'is the model really better?' p-value."""
    d = np.asarray(loss_model) - np.asarray(loss_base)
    n = len(d); rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, nb))
    ix = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = d[ix.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    return {"mean_diff": d.mean(), "ci_lo": np.percentile(means, 2.5), "ci_hi": np.percentile(means, 97.5), "p_not_better": float((means >= 0).mean())}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1, run)
    return adj
