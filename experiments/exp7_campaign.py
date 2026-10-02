"""E7: multi-round research campaigns sharing one holdout.

A programme runs R rounds against ONE holdout (n=2520 periods). Each round adds k=50 new
base strategies to a growing universe; with probability 0.3 a round's batch contains m=5
genuine signals (annualised Sharpe MU in train, holdout and test). All protocols consider
4 candidates per round and spend alpha=5% over the campaign.

Scenario "independent": each round's candidates are built from that round's new batch,
using training data only (no information flows between rounds).
  split_vault  holdout cut into R chunks; round r tests its 4 candidates on chunk r at
               t > t^{-1}(a/(4R))
  bonf         4 candidates on the full holdout at t > t^{-1}(a/(4R)) (trusted reuse)
  sealed       same bar, enforced by a sealed verdict (numerically identical to bonf here)

Scenario "adaptive": the researcher sees earlier holdout results and builds on them.
  naive_dsr    full-precision holdout Sharpe of every strategy queried so far; candidates
               are top-j by HOLDOUT Sharpe over the accumulated universe; certify with
               ledger DSR over the cumulative ledger
  bonf         same full-precision, holdout-ranked candidates, certified at the Bonferroni
               bar t^{-1}(a/(4R)) (what "reuse with a multiplicity correction" does)
  sealed       only PASS/FAIL bits flow back; candidates are top-j by TRAIN Sharpe over the
               accumulated universe, plus the previous PASSed portfolio merged with the
               round's best training candidate; bar t^{-1}(a/(4R)) (weak FWER, Cor. 4(ii))
Metrics: P(any false certification) where false = certified portfolio with no positive
exposure to genuine signals, under the global null (mu=0) and under partial nulls
(mu=0.6); power = share of genuine rounds with a true certification.
Outputs results/exp7.csv
"""
import math, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import deflated_sharpe, sharpe, tstat

N, K_BASE, M, P_SIGNAL, ALPHA = 2520, 50, 5, 0.3, 0.05
JS = [1, 2, 5, 10]


def topj_by(score, cols):
    order = cols[np.argsort(-score[cols])]
    return [order[:j] for j in JS]


def ew(idx, k):
    w = np.zeros(k); w[idx] = 1.0 / len(idx)
    return w


def campaign(rng, R, mu_ann, scenario):
    mu = mu_ann / math.sqrt(252)
    K = K_BASE * R
    sig_round = (rng.random(R) < P_SIGNAL) & (mu > 0)
    true_mu = np.zeros(K)
    for r in range(R):
        if sig_round[r]:
            true_mu[r * K_BASE + rng.choice(K_BASE, M, replace=False)] = mu
    tr, ho = (rng.standard_normal((N, K)) + true_mu for _ in range(2))
    sr_tr, sr_ho = sharpe(tr), sharpe(ho)
    bar = stats.t.isf(ALPHA / (4 * R), df=N - 1)
    edges = np.linspace(0, N, R + 1).astype(int)
    out = []
    ledger_srs = []  # cumulative ledger for naive_dsr
    passed_w = None

    def record(r, proto, w, ok):
        true = bool(ok and float(true_mu @ w) > 0)
        out.append(dict(R=R, mu=mu_ann, scenario=scenario, round=r, signal=bool(sig_round[r]),
                        protocol=proto, certified=bool(ok), true_cert=true, false_cert=bool(ok and not true)))

    for r in range(R):
        new = np.arange(r * K_BASE, (r + 1) * K_BASE)
        seen = np.arange(0, (r + 1) * K_BASE)
        if scenario == "independent":
            cands = [ew(i, K) for i in topj_by(sr_tr, new)]
            chunk = ho[edges[r]:edges[r + 1]]
            cb = stats.t.isf(ALPHA / (4 * R), df=chunk.shape[0] - 1)
            w = next((c for c in cands if tstat(chunk @ c) > cb), cands[0])
            record(r, "split_vault", w, tstat(chunk @ w) > cb)
            w = next((c for c in cands if tstat(ho @ c) > bar), cands[0])
            ok = tstat(ho @ w) > bar
            record(r, "bonf", w, ok)
            record(r, "sealed", w, ok)
        else:
            # full-precision adaptive researcher
            ledger_srs.extend(sr_ho[new].tolist())
            cands = [ew(i, K) for i in topj_by(sr_ho, seen)]
            c_srs = [float(sharpe(ho @ c)) for c in cands]
            ledger_srs.extend(c_srs)
            b = int(np.argmax(c_srs))
            d = deflated_sharpe(ho @ cands[b], n_trials=len(ledger_srs), var_trial_sr=float(np.var(ledger_srs, ddof=1)))
            record(r, "naive_dsr", cands[b], d > 0.95)
            record(r, "bonf", cands[b], tstat(ho @ cands[b]) > bar)
            # sealed researcher: train info + verdict bits only
            cands = [ew(i, K) for i in topj_by(sr_tr, seen)][:3]
            if passed_w is not None:
                m_ = passed_w + ew(topj_by(sr_tr, new)[1], K)
                cands.append(m_ / np.abs(m_).sum())
            else:
                cands.append(ew(topj_by(sr_tr, new)[2], K))
            w = next((c for c in cands if tstat(ho @ c) > bar), None)
            if w is not None:
                passed_w = w
            record(r, "sealed", w if w is not None else cands[0], w is not None)
    return out


def main(reps=500):
    rng = np.random.default_rng(77)
    rows = []
    for scen in ("independent", "adaptive"):
        for R in [1, 2, 5, 10, 20]:
            for mu in [0.0, 0.6]:
                for rep in range(reps):
                    rows += [dict(x, rep=rep) for x in campaign(rng, R, mu, scen)]
            print(scen, "R", R, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(Path(__file__).resolve().parents[1] / "results" / "exp7.csv", index=False)
    g = d.groupby(["scenario", "R", "mu", "protocol", "rep"]).false_cert.any()
    fw = g.groupby(["scenario", "R", "mu", "protocol"]).mean().unstack("mu")
    pw = d[(d.mu > 0) & d.signal].groupby(["scenario", "R", "protocol"]).true_cert.mean()
    t = fw.rename(columns={0.0: "FWER_null", 0.6: "FWER_partial"})
    t["power"] = pw
    print(t.round(3).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)
