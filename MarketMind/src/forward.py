"""Forward test of frozen model v1 (random forest, hyperparameters from results/tables/selected_hyperparams.json).
  python -m src.forward predict   -> trains on ALL rows with known targets, predicts the session after the latest row, appends to results/forward_log.csv
  python -m src.forward score     -> after you refresh the raw CSV, fills in what actually happened and prints running results
Predictions are logged BEFORE the outcome exists, so they cannot be edited in hindsight."""
import json, sys, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from .run import CFG, ROOT
from .data_loader import load_clean_index
from .targets import add_targets
from .features import build_features, FEATURES
from .models import make_model, predict

LOG = ROOT / "results/forward_log.csv"


def _table():
    clean, _ = load_clean_index(ROOT / CFG["data"]["raw_path"], CFG["data"]["max_gap_days"], CFG["data"]["suspicious_abs_return"])
    d = add_targets(clean)
    d = pd.concat([d, build_features(d, CFG["features"]["min_warmup_sessions"])], axis=1)
    return d[d["date"] >= CFG["sample"]["primary_start"]].reset_index(drop=True)


def do_predict():
    d = _table(); p = json.load(open(ROOT / "results/tables/selected_hyperparams.json"))["rf_clf"]
    last = d.iloc[[-1]]
    if last[FEATURES].isna().any(axis=1).iloc[0]:
        sys.exit("Latest row has missing features.")
    tr = d.dropna(subset=FEATURES + ["y_dir"])
    m = make_model("rf", "clf", **p).fit(tr[FEATURES], tr["y_dir"])
    prob = float(predict(m, "clf", last[FEATURES])[0])
    row = dict(made_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), based_on_close_date=str(last["date"].iloc[0].date()),
               close=float(last["close"].iloc[0]), prob_up=round(prob, 4), call="UP" if prob >= .5 else "NOT UP",
               train_rows=len(tr), base_rate=round(float(tr["y_dir"].mean()), 4), actual_next_ret="", actual_dir="")
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    if len(old) and str(old["based_on_close_date"].iloc[-1]) == row["based_on_close_date"]:
        print("Already predicted for", row["based_on_close_date"]); print(old.tail(1).T); return
    pd.concat([old, pd.DataFrame([row])]).to_csv(LOG, index=False)
    print(f"Based on close of {row['based_on_close_date']}: P(next session up) = {prob:.1%} -> {row['call']}  (training base rate {row['base_rate']:.1%})")


def do_score():
    d = add_targets(load_clean_index(ROOT / CFG["data"]["raw_path"], CFG["data"]["max_gap_days"])[0])
    lg = pd.read_csv(LOG, dtype={"actual_next_ret": object, "actual_dir": object}); look = d.assign(k=d["date"].dt.strftime("%Y-%m-%d")).set_index("k")
    for i, r in lg.iterrows():
        k = str(r["based_on_close_date"])
        if k in look.index and not np.isnan(look.loc[k, "y_ret"]):
            lg.loc[i, "actual_next_ret"] = look.loc[k, "y_ret"]; lg.loc[i, "actual_dir"] = int(look.loc[k, "y_ret"] > 0)
    lg.to_csv(LOG, index=False)
    lg["actual_dir"] = pd.to_numeric(lg["actual_dir"], errors="coerce")
    s = lg[lg["actual_dir"].notna()].copy(); s["actual_dir"] = s["actual_dir"].astype(int)
    if s.empty: print("No scored predictions yet."); return
    hit = ((s["prob_up"] >= .5).astype(int) == s["actual_dir"]); br = ((s["prob_up"] - s["actual_dir"]) ** 2).mean(); bb = ((s["base_rate"] - s["actual_dir"]) ** 2).mean()
    print(f"scored {len(s)} | hits {int(hit.sum())} ({hit.mean():.1%}) | Brier {br:.4f} vs base-rate {bb:.4f} | skill {1 - br / bb:+.1%}")
    print("Need ~300+ scored sessions before this means anything.")

if __name__ == "__main__":
    {"predict": do_predict, "score": do_score}[sys.argv[1]]()
