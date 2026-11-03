"""Paired test: does the random forest beat a FAIR 'yesterday's direction' baseline on the same test dates?
Baseline probability = P(up | yesterday up) or P(up | yesterday not up), estimated from training rows only
(same expanding window / refit schedule as the models). Planned after the main test, so it is labelled post-hoc."""
import json, numpy as np, pandas as pd
from .run import prepare, CFG, log
from .features import FEATURES
from .evaluation import walk_forward, block_bootstrap_diff

def main():
    sel = json.load(open("results/tables/selected_hyperparams.json"))
    _, _, d = prepare()
    s, k = CFG["splits"]["test_start"], CFG["splits"]["refit_every"]; end = str(d["date"].max().date()); bs = CFG["bootstrap"]
    rf = walk_forward(d, FEATURES, "rf", "clf", sel["rf_clf"], s, end, k)
    hg = walk_forward(d, FEATURES, "hgb", "clf", sel["hgb_clf"], s, end, k)
    dd = d.reset_index(drop=True); prev_up = (dd["ret_0"] > 0).to_numpy(); y_all = dd["y_dir"].to_numpy()
    idx = rf.index.to_numpy(); p = np.empty(len(idx))
    for b in range(0, len(idx), k):
        blk = idx[b:b + k]; tr = np.arange(0, blk[0])
        pu = y_all[tr][prev_up[tr]].mean(); pd_ = y_all[tr][~prev_up[tr]].mean()
        p[b:b + k] = np.where(prev_up[blk], pu, pd_)
    y = rf["y"].to_numpy(); base = rf["train_mean"].to_numpy()
    L = lambda q: (q - y) ** 2
    out = []
    for name, q in [("prev-direction (probabilistic)", p), ("rf", rf["pred"].to_numpy()), ("hgb", hg["pred"].to_numpy())]:
        out.append(dict(model=name, brier=L(q).mean(), brier_skill_vs_base=1 - L(q).mean() / L(base).mean(), accuracy=((q >= .5) == y).mean()))
    t = []
    for name, q in [("rf vs prev-direction", rf["pred"].to_numpy()), ("hgb vs prev-direction", hg["pred"].to_numpy()), ("prev-direction vs base rate", None)]:
        a, b_ = (L(q), L(p)) if q is not None else (L(p), L(base))
        t.append(dict(comparison=name, **block_bootstrap_diff(a, b_, bs["block_length"], bs["n_boot"], bs["seed"])))
    acc = (((rf["pred"].to_numpy() >= .5) == y).astype(float) - ((p >= .5) == y).astype(float))
    t.append(dict(comparison="accuracy diff rf - prev-direction (neg = rf better in loss form)", **block_bootstrap_diff(-acc, np.zeros(len(acc)), bs["block_length"], bs["n_boot"], bs["seed"])))
    pd.DataFrame(out).to_csv("results/tables/paired_test_models.csv", index=False)
    pd.DataFrame(t).to_csv("results/tables/paired_test_bootstrap.csv", index=False)
    log("posthoc_paired", "ALL", {}, "NOTE", "", "Post-hoc paired test on test period (3rd look); no model/hyperparameter changed")
    pd.set_option("display.width", 200); print(pd.DataFrame(out).round(4).to_string(index=False)); print(pd.DataFrame(t).round(4).to_string(index=False))

if __name__ == "__main__":
    main()
