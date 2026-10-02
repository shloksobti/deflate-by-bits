"""For aggressive E6 episodes: would ledger DSR have certified the best-so-far portfolio
had the agent stopped after q queries?  (Fragility of DSR's empirical variance term.)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from analyze_e6 import S0, NAMES, vec, stationary_bootstrap_null
from core import deflated_sharpe, sharpe, tstat
RES = Path(__file__).resolve().parents[1] / "results" / "exp6"
rows = []
for p in sorted(RES.glob("claude_opus_optimize_naive_*_ledger.jsonl")):
    L = [json.loads(l) for l in open(p)]
    end = next(x for x in L if x.get("end"))
    rng = np.random.default_rng(end["seed"])
    S = {s: stationary_bootstrap_null(S0[s], rng) for s in ("train", "val", "test")}
    va = S["val"]
    Q = [vec(x["w"]) for x in L if "q" in x]
    if len(Q) < 100:
        continue
    W = np.array(Q).T
    srs = sharpe(va @ W)
    t = srs * np.sqrt(va.shape[0])
    for q in sorted(set([50, 100, 200, 300, 423, 424, 430, 440, 450, 500, 600, 800, 1000, 1500, 2000, 3000, len(Q)])):
        if q > len(Q):
            continue
        b = int(np.argmax(t[:q]))
        d = deflated_sharpe(va @ W[:, b], n_trials=q, var_trial_sr=float(np.var(srs[:q], ddof=1)))
        rows.append(dict(ep=p.name[:-14], q=q, best_t=float(t[b]), nV=float(np.var(srs[:q], ddof=1) * va.shape[0]),
                         dsr=d, cert=d > 0.95, test_t=float(tstat(S["test"] @ W[:, b]))))
d = pd.DataFrame(rows)
print(d.round(3).to_string())
d.to_csv(RES / "prefix_dsr.csv", index=False)
