"""Exp 1: Under a pure-noise universe, adaptive ensemble attacks defeat the
Deflated Sharpe Ratio even when DSR is computed from a complete trial ledger.

Sweeps the number of base strategies k and the cross-strategy correlation rho.
Outputs results/exp1.csv.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import (NaiveOracle, best_of_k, sign_select_ensemble, sign_flip_ensemble,
                  greedy_forward, rank_long_short, ledger_dsr, tstat)

N_PERIODS = 2520          # ~10 years of daily data per split
KS = [5, 10, 20, 50, 100, 200, 500, 1000]
RHOS = [0.0, 0.3, 0.6]
REPS = 200
ATTACKS = {"best_of_k": best_of_k, "sign_select": sign_select_ensemble,
           "sign_flip": sign_flip_ensemble, "rank_ls": rank_long_short, "greedy": greedy_forward}
GREEDY_MAX_K = 200        # greedy is O(k) full-matrix queries; cap for runtime


def universe(rng, n, k, rho):
    f = rng.standard_normal((n, 1))
    return np.sqrt(rho) * f + np.sqrt(1 - rho) * rng.standard_normal((n, k))


def main():
    rng = np.random.default_rng(20261002)
    rows = []
    t0 = time.time()
    for rho in RHOS:
        for rep in range(REPS):
            kmax = max(KS)
            Rv_full = universe(rng, N_PERIODS, kmax, rho)
            Rt_full = universe(rng, N_PERIODS, kmax, rho)
            for k in KS:
                Rv, Rt = Rv_full[:, :k], Rt_full[:, :k]
                for name, atk in ATTACKS.items():
                    if name == "greedy" and k > GREEDY_MAX_K:
                        continue
                    o = NaiveOracle(Rv)
                    w = atk(o, k, rng=rng)
                    rows.append(dict(rho=rho, rep=rep, k=k, attack=name,
                                     n_queries=o.n_queries,
                                     t_val=float(tstat(Rv @ w)),
                                     t_test=float(tstat(Rt @ w)),
                                     dsr=ledger_dsr(o, w, Rv)))
        print(f"rho={rho} done {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = Path(__file__).resolve().parents[1] / "results" / "exp1.csv"
    df.to_csv(out, index=False)
    summ = (df.assign(certified=df.dsr > 0.95)
              .groupby(["rho", "attack", "k"])[["t_val", "t_test", "dsr", "certified"]]
              .mean().round(3))
    print(summ.to_string())


if __name__ == "__main__":
    main()
