"""Exp 3: Workflows on real Ken French industry strategies (k=423).

(a) actual data: one realisation, report validation t, ledger DSR, test t.
(b) stationary-bootstrap null replicates (all edges removed): false-certification rate.
Outputs results/exp3_actual.csv, results/exp3_null.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import (NaiveOracle, greedy_forward, sign_flip_ensemble, rank_long_short,
                  ledger_dsr, span_threshold, tstat, sharpe)
from realdata import base_library, split, stationary_bootstrap_null

ALPHA = 0.05
JS = [1, 2, 5, 10, 20, 50]
ROOT = Path(__file__).resolve().parents[1]


def topj(R):
    order = np.argsort(-sharpe(R))
    cands = []
    for j in JS:
        w = np.zeros(R.shape[1]); w[order[:j]] = 1.0 / j
        cands.append(w)
    return sorted(cands, key=lambda c: -float(sharpe(R @ c)))


def workflows(tr, va, te, rng):
    k = va.shape[1]
    z = stats.norm.ppf(1 - ALPHA)
    res = {}
    for name, atk in [("naive_signflip", sign_flip_ensemble), ("naive_rankls", rank_long_short),
                      ("naive_greedy", greedy_forward)]:
        o = NaiveOracle(va)
        w = atk(o, k, rng=rng)
        d = ledger_dsr(o, w, va)
        res[name] = (w, d > 0.95, d)
    o = NaiveOracle(va)
    for i in range(k):
        o.query(np.eye(k)[i])
    cs = topj(va)
    for c in cs:
        o.query(c)
    d = ledger_dsr(o, cs[0], va)
    res["naive_topj"] = (cs[0], d > 0.95, d)
    w = topj(tr)[0]
    res["train_vault"] = (w, tstat(va @ w) > z, np.nan)
    cs = topj(tr)
    cK = stats.t.isf(ALPHA / (len(cs) + 1), df=va.shape[0] - 1)
    w, ok = cs[0], False
    for c in cs:
        if tstat(va @ c) > cK:
            w, ok = c, True
            break
    res["verdict"] = (w, ok, np.nan)
    wg = res["naive_greedy"][0]
    res["naive_span"] = (wg, tstat(va @ wg) > span_threshold(k, va.shape[0], ALPHA), np.nan)
    h = va.shape[0] // 2
    A, B = va[:h], va[h:]
    cs = topj(A)
    cB = stats.t.isf(ALPHA / (len(cs) + 1), df=B.shape[0] - 1)
    w, ok = cs[0], False
    for c in cs:
        if tstat(B @ c) > cB:
            w, ok = c, True
            break
    res["split_verdict"] = (w, ok, np.nan)
    return [dict(workflow=n, certified=bool(c), dsr=d, t_val=float(tstat(va @ w)),
                 t_test=float(tstat(te @ w)), sr_test_ann=float(sharpe(te @ w) * np.sqrt(252)))
            for n, (w, c, d) in res.items()]


def main(reps):
    rng = np.random.default_rng(11)
    S = split(base_library())
    act = pd.DataFrame(workflows(S["train"], S["val"], S["test"], rng))
    act.to_csv(ROOT / "results/exp3_actual.csv", index=False)
    print(act.round(3).to_string())
    rows = []
    for r in range(reps):
        tr, va, te = (stationary_bootstrap_null(S[s], rng) for s in ("train", "val", "test"))
        rows += [dict(x, rep=r) for x in workflows(tr, va, te, rng)]
        if r % 25 == 0:
            print("rep", r, flush=True)
    nul = pd.DataFrame(rows)
    nul.to_csv(ROOT / "results/exp3_null.csv", index=False)
    print(nul.groupby("workflow")[["certified", "t_val", "t_test"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
