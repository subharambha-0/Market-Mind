"""CLI:  python -m src.run audit | validate | test
validate -> picks hyperparameters on the validation block (never sees test rows).
test     -> evaluates the FROZEN choices on the final test period once; refuses to run without validate."""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
from .data_loader import load_clean_index
from .targets import add_targets
from .features import build_features, FEATURES
from .evaluation import walk_forward, clf_metrics, reg_metrics, block_bootstrap_diff, holm
from . import baselines as B

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config/experiment.yaml"))
TAB, FIG, LOG = ROOT / "results/tables", ROOT / "results/figures", ROOT / "results/experiment_log.csv"


def log(stage, model, params, metric, value, extra=""):
    row = pd.DataFrame([{"time_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "stage": stage,
                         "model": model, "params": json.dumps(params), "metric": metric, "value": value, "note": extra}])
    row.to_csv(LOG, mode="a", header=not LOG.exists(), index=False)


def prepare():
    clean, rep = load_clean_index(ROOT / CFG["data"]["raw_path"], CFG["data"]["max_gap_days"], CFG["data"]["suspicious_abs_return"])
    d = add_targets(clean)
    d = pd.concat([d, build_features(d, CFG["features"]["min_warmup_sessions"])], axis=1)
    d = d[d["date"] >= CFG["sample"]["primary_start"]]
    d = d.dropna(subset=FEATURES + ["y_ret"]).reset_index(drop=True)
    return clean, rep, d


def audit():
    clean, rep, d = prepare()
    for p in (TAB, FIG):
        p.mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed").mkdir(exist_ok=True)
    clean.to_csv(ROOT / "data/processed/nepse_index_clean.csv", index=False)
    d.to_csv(ROOT / "data/processed/model_table.csv", index=False)
    per_year = clean.groupby(clean["date"].dt.year).size()
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].semilogy(clean["date"], clean["close"]); ax[0].set_title("NEPSE index close (log scale)")
    for g in rep["gaps_over_limit"]:
        ax[0].axvline(pd.Timestamp(g["date"]), color="r", alpha=.3)
    ax[1].plot(clean["date"], clean["ret"], lw=.4); ax[1].set_title("Daily returns")
    ax[1].plot(clean["date"], clean["ret"].rolling(20).std(), color="k", lw=.8, label="20-session vol"); ax[1].legend()
    ax[2].bar(per_year.index, per_year.values); ax[2].set_title("Recorded sessions per year (2020 = real closure)")
    plt.tight_layout(); plt.savefig(FIG / "data_audit.png", dpi=110); plt.close()
    info = dict(rep); info.pop("exclusions")
    info.update(model_rows=len(d), model_first=str(d["date"].min().date()), model_last=str(d["date"].max().date()),
                dev_rows=int((d["date"] <= CFG["splits"]["dev_end"]).sum()),
                val_rows=int(((d["date"] >= CFG["splits"]["val_start"]) & (d["date"] <= CFG["splits"]["dev_end"])).sum()),
                test_rows=int((d["date"] >= CFG["splits"]["test_start"]).sum()),
                up_share_dev=float(d.loc[d["date"] <= CFG["splits"]["dev_end"], "y_dir"].mean()),
                up_share_test=float(d.loc[d["date"] >= CFG["splits"]["test_start"], "y_dir"].mean()),
                zero_returns_in_model_table=int((d["y_ret"] == 0).sum()))
    json.dump(info, open(TAB / "data_audit.json", "w"), indent=1, default=str)
    print(json.dumps(info, indent=1, default=str))


def validate():
    _, _, d = prepare()
    dev = d[d["date"] <= CFG["splits"]["dev_end"]].reset_index(drop=True)   # test rows physically removed
    s, e, k = CFG["splits"]["val_start"], CFG["splits"]["dev_end"], CFG["splits"]["refit_every"]
    chosen, rows = {}, []
    for name, task in [("logistic", "clf"), ("rf", "clf"), ("hgb", "clf"), ("ridge", "reg"), ("rf", "reg"), ("hgb", "reg")]:
        grid = CFG["grids"][name]
        best = None
        for vals in product(*grid.values()):
            params = dict(zip(grid.keys(), vals))
            o = walk_forward(dev, FEATURES, name, task, params, s, e, k)
            score = np.mean((o["pred"] - o["y"]) ** 2)                     # Brier (clf) or MSE (reg)
            base = np.mean((o["train_mean"] - o["y"]) ** 2)
            skill = 1 - score / base
            rows.append(dict(model=name, task=task, **params, val_loss=score, val_skill_vs_base=skill))
            log("validation", f"{name}_{task}", params, "val_skill_vs_base", skill, "primary-metric proxy: 1-loss/loss_baseline")
            if best is None or score < best[0]:
                best = (score, params, skill)
        chosen[f"{name}_{task}"] = best[1]
    pd.DataFrame(rows).to_csv(TAB / "validation_grid.csv", index=False)
    json.dump(chosen, open(TAB / "selected_hyperparams.json", "w"), indent=1)
    print(pd.DataFrame(rows).to_string()); print(chosen)


def test():
    sel_path = TAB / "selected_hyperparams.json"
    if not sel_path.exists():
        sys.exit("Run `validate` first: hyperparameters must be frozen before the test period is used.")
    sel = json.load(open(sel_path))
    _, _, d = prepare()
    s, k = CFG["splits"]["test_start"], CFG["splits"]["refit_every"]
    end = str(d["date"].max().date()); bs = CFG["bootstrap"]
    preds = {m: walk_forward(d, FEATURES, m.rsplit("_", 1)[0], m.rsplit("_", 1)[1], p, s, end, k) for m, p in sel.items()}
    ref = preds["logistic_clf"]
    tmr = preds["ridge_reg"]["train_mean"].to_numpy()   # expanding mean of RETURNS (training rows only)
    y, yr, tm, r0 = ref["y"].to_numpy(), ref["y_ret"].to_numpy(), ref["train_mean"].to_numpy(), ref["ret_0"].to_numpy()
    dates = ref["date"].to_numpy()
    # sanity: all models evaluated on identical dates
    assert all((o["date"].to_numpy() == dates).all() for o in preds.values())

    clf_rows, losses = [], {}
    cand = {"BASE majority (train)": (tm >= .5).astype(float), "BASE previous-direction": B.previous_direction(r0),
            "BASE base-rate prob": tm}
    for m in ["logistic_clf", "rf_clf", "hgb_clf"]:
        cand[m] = preds[m]["pred"].to_numpy()
    for name, p in cand.items():
        met = clf_metrics(y, p if name.startswith(("logistic", "rf", "hgb", "BASE base")) else p, tm)
        clf_rows.append(dict(model=name, **{kk: v for kk, v in met.items() if kk != "confusion"}, confusion=json.dumps(met["confusion"])))
        losses[name] = (p - y) ** 2
    clf = pd.DataFrame(clf_rows); clf.to_csv(TAB / "test_direction_results.csv", index=False)

    reg_rows = []
    rc = {"BASE zero": np.zeros(len(y)), "BASE hist-mean": tmr}
    for m in ["ridge_reg", "rf_reg", "hgb_reg"]:
        rc[m] = preds[m]["pred"].to_numpy()
    rlosses = {}
    for name, p in rc.items():
        reg_rows.append(dict(model=name, **reg_metrics(yr, p, tmr))); rlosses[name] = (yr - p) ** 2
    reg = pd.DataFrame(reg_rows); reg.to_csv(TAB / "test_return_results.csv", index=False)

    tests = []
    for m in ["logistic_clf", "rf_clf", "hgb_clf"]:
        tests.append(dict(comparison=f"{m} vs base-rate (Brier)", **block_bootstrap_diff(losses[m], losses["BASE base-rate prob"], bs["block_length"], bs["n_boot"], bs["seed"])))
    for m in ["ridge_reg", "rf_reg", "hgb_reg"]:
        tests.append(dict(comparison=f"{m} vs hist-mean (sq. error)", **block_bootstrap_diff(rlosses[m], rlosses["BASE hist-mean"], bs["block_length"], bs["n_boot"], bs["seed"])))
    t = pd.DataFrame(tests); t["p_holm"] = holm(t["p_not_better"].to_numpy()); t.to_csv(TAB / "test_bootstrap_holm.csv", index=False)

    # --- audit / robustness
    rob = []
    yr_ = pd.DatetimeIndex(dates).year
    ext = np.abs(yr) < 0.04
    for label, mask in [("all", np.ones(len(y), bool)), *[(f"year {u}", yr_ == u) for u in sorted(set(yr_))], ("excl |r|>=4%", ext)]:
        for m in ["logistic_clf", "rf_clf", "hgb_clf"]:
            rob.append(dict(subset=label, n=int(mask.sum()), model=m, brier_skill=1 - np.mean((preds[m]["pred"].to_numpy()[mask] - y[mask]) ** 2) / np.mean((tm[mask] - y[mask]) ** 2),
                            share_pred_up=float((preds[m]["pred"].to_numpy()[mask] >= .5).mean())))
        for m in ["ridge_reg", "rf_reg", "hgb_reg"]:
            rob.append(dict(subset=label, n=int(mask.sum()), model=m, oos_r2_vs_histmean=1 - np.sum((yr[mask] - rc[m][mask]) ** 2) / np.sum((yr[mask] - tmr[mask]) ** 2)))
    pd.DataFrame(rob).to_csv(TAB / "test_robustness.csv", index=False)

    for _, r in clf.iterrows():
        log("final_test", r["model"], {}, "brier_skill", r["brier_skill"])
    for _, r in reg.iterrows():
        log("final_test", r["model"], {}, "oos_r2_vs_histmean", r["oos_r2_vs_histmean"])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m in ["logistic_clf", "rf_clf", "hgb_clf"]:
        ax[0].plot(dates, np.cumsum(losses["BASE base-rate prob"] - losses[m]), label=m)
    ax[0].axhline(0, color="k", lw=.5); ax[0].set_title("Cumulative Brier gain vs base rate (>0 = better)"); ax[0].legend(fontsize=7)
    p = preds["logistic_clf"]["pred"].to_numpy(); bins = pd.cut(p, np.linspace(p.min(), p.max(), 6), include_lowest=True)
    cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ax[1].plot(cal["p"], cal["y"], "o-"); ax[1].plot([.4, .7], [.4, .7], "k--", lw=.5); ax[1].set_title("Logistic calibration (test)")
    plt.tight_layout(); plt.savefig(FIG / "test_results.png", dpi=110); plt.close()
    pd.set_option("display.width", 200); pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(clf.drop(columns="confusion").to_string()); print(reg.to_string()); print(t.to_string())
    print(pd.DataFrame(rob).to_string())
    print("test period", dates.min(), dates.max(), "n =", len(y))


if __name__ == "__main__":
    {"audit": audit, "validate": validate, "test": test}[sys.argv[1]]()
