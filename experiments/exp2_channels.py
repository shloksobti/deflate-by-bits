"""Exp 2: Validity and power of holdout feedback channels for an adaptive agent.

Universe: k base strategies.  m of them carry true per-period Sharpe mu in the
training regime (set S_tr) and m in the holdout/test regime (set S_ho); a fraction
phi of S_ho overlaps S_tr (phi=1: stationary, phi=0: full rotation).
The test period follows the holdout regime.

Channels (feedback the agent receives about the holdout) and certification rule:
  naive_dsr   full-precision holdout SR, greedy search on holdout, certify DSR>0.95
              computed from the complete query ledger (status quo).
  naive_topj_dsr  rank all k on holdout, report best equal-weight top-j, ledger DSR
  train_vault top-j on train only (j chosen on train), one-shot holdout test t > z_{1-a}   (0 bits)
  split_vault top-j on holdout half A (full precision), one-shot test on half B
  verdict     agent queries top-j-on-train candidates (K=6); holdout answers only PASS/FAIL
              at t > c_K = t^{-1}(a/(K+1)); stop at first PASS       (log(K+1) nats)
  ladder3     greedy on holdout with sealed bit-ladder, budget B=3; certify if final
              holdout t > ladder_threshold(#queries, 3)
Outputs results/exp2.csv
"""
import math, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import (NaiveOracle, SealedLadderOracle, ThresholdoutOracle, greedy_forward, ledger_dsr,
                  ladder_threshold, span_threshold, tstat, sharpe)

N = 2520
K_BASE = 100
M = 10
ALPHA = 0.05
MUS_ANNUAL = [0.0, 0.2, 0.4, 0.6]
PHIS = [1.0, 0.5, 0.0]
REPS = 200


def make(rng, mu, phi):
    k = K_BASE
    S_tr = rng.choice(k, M, replace=False)
    n_keep = int(round(phi * M))
    rest = np.setdiff1d(np.arange(k), S_tr)
    S_ho = np.concatenate([S_tr[:n_keep], rng.choice(rest, M - n_keep, replace=False)])
    def block(S):
        R = rng.standard_normal((N, k))
        R[:, S] += mu
        return R
    return block(S_tr), block(S_ho), block(S_ho), S_ho


JS = [1, 2, 5, 10, 15, 20]


def topj(R, js=JS):
    """Honest-quant candidates: equal-weight portfolios of the top-j base strategies
    ranked by in-sample SR on R, ordered by their in-sample SR."""
    order = np.argsort(-sharpe(R))
    cands = []
    for j in js:
        w = np.zeros(R.shape[1]); w[order[:j]] = 1.0 / j
        cands.append(w)
    return sorted(cands, key=lambda c: -float(sharpe(R @ c)))


def run_one(rng, mu_ann, phi):
    mu = mu_ann / math.sqrt(252)
    Rtr, Rho, Rte, _ = make(rng, mu, phi)
    z = stats.norm.ppf(1 - ALPHA)
    out = {}

    o = NaiveOracle(Rho)
    w = greedy_forward(o, K_BASE, rng=rng)
    out["naive_dsr"] = (ledger_dsr(o, w, Rho) > 0.95, w)

    # status-quo agent #2: rank on holdout, report best top-j, ledger DSR
    o = NaiveOracle(Rho)
    for i in range(K_BASE):
        o.query(np.eye(K_BASE)[i])
    for c in topj(Rho):
        o.query(c)
    w = topj(Rho)[0]
    out["naive_topj_dsr"] = (ledger_dsr(o, w, Rho) > 0.95, w)

    w = topj(Rtr)[0]
    out["train_vault"] = (tstat(Rho @ w) > z, w)

    A, B = Rho[: N // 2], Rho[N // 2:]
    w = topj(A)[0]
    out["split_vault"] = (tstat(B @ w) > z, w)

    cands = topj(Rtr)
    cK = stats.t.isf(ALPHA / (len(cands) + 1), df=N - 1)
    passed, w = False, cands[0]
    for c in cands:
        if tstat(Rho @ c) > cK:
            passed, w = True, c
            break
    out["verdict"] = (passed, w)

    # span bound (Scheffe/GRS): valid for any analyst in the span; applied to the greedy report
    out["naive_span"] = (tstat(Rho @ out["naive_dsr"][1]) > span_threshold(K_BASE, N, ALPHA), out["naive_dsr"][1])

    # adaptive stress test of the sealed verdict: K=200 queries; after the train-ranked
    # candidates, keep perturbing the incumbent (add / flip one random signal) on every FAIL
    K_ADA = 200
    cA = stats.t.isf(ALPHA / (K_ADA + 1), df=N - 1)
    queue = topj(Rtr)
    inc = queue[0].copy()
    passed, w = False, inc
    for q in range(K_ADA):
        if q < len(queue):
            c = queue[q]
        else:
            c = inc.copy(); i = rng.integers(K_BASE)
            c[i] = c[i] + rng.choice([-1.0, 1.0]) * (np.abs(inc).max() if np.abs(inc).max() > 0 else 1.0)
            if np.abs(c).sum() == 0:
                continue
            c = c / np.abs(c).sum()
            if sharpe(Rtr @ c) >= sharpe(Rtr @ inc):  # adapt the incumbent using free train info
                inc = c
        if tstat(Rho @ c) > cA:
            passed, w = True, c
            break
    out["verdict_adaptive"] = (passed, w)

    # split + verdict: search with full precision on holdout half A, sealed verdict on half B
    cs = topj(A)
    cB = stats.t.isf(ALPHA / (len(cs) + 1), df=N // 2 - 1)
    passed, w = False, cs[0]
    for c in cs:
        if tstat(B @ c) > cB:
            passed, w = True, c
            break
    out["split_verdict"] = (passed, w)

    # Thresholdout feedback (Dwork et al. 2015) to a greedy analyst; certify final at z_{1-a}
    sd = 1 / math.sqrt(N)
    o = ThresholdoutOracle(Rho, Rtr, threshold=2 * sd, sigma=0.5 * sd, budget=50, rng=rng)
    w = greedy_forward(o, K_BASE, rng=rng)
    out["thresholdout"] = (tstat(Rho @ w) > z, w)

    o = SealedLadderOracle(Rho, eta=1.0, budget=3)
    w = greedy_forward(o, K_BASE, rng=rng)
    thr = ladder_threshold(o.n_queries, 3, ALPHA, n=N)
    out["ladder3"] = (tstat(Rho @ w) > thr, w)

    return [dict(mu=mu_ann, phi=phi, channel=ch, certified=bool(c),
                 t_test=float(tstat(Rte @ w))) for ch, (c, w) in out.items()]


def main():
    rng = np.random.default_rng(7)
    rows, t0 = [], time.time()
    for mu in MUS_ANNUAL:
        for phi in (PHIS if mu > 0 else [1.0]):
            for rep in range(REPS):
                rows += [dict(r, rep=rep) for r in run_one(rng, mu, phi)]
            print(f"mu={mu} phi={phi} {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).resolve().parents[1] / "results" / "exp2.csv", index=False)
    df["t_test_cert"] = df.t_test.where(df.certified)
    print(df.groupby(["mu", "phi", "channel"])[["certified", "t_test_cert"]].mean().round(3).to_string())


if __name__ == "__main__":
    if len(sys.argv) > 1:
        REPS = int(sys.argv[1])
    main()
