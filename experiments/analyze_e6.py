"""Analysis for E6 (code-executing agents against the sealed research server)."""
import json, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realdata import base_library, split, stationary_bootstrap_null
from core import deflated_sharpe, sharpe, tstat, dedupe_ledger, span_threshold

RES, TAB = ROOT / "results" / "exp6", ROOT / "paper" / "tables"
R = base_library(); NAMES = list(R.columns); S0 = split(R); IDX = {n: i for i, n in enumerate(NAMES)}


def vec(d):
    w = np.zeros(len(NAMES))
    for k, v in d.items():
        w[IDX[k]] += v
    return w / np.abs(w).sum()


MIN_N = 10


def ever_certified(W, va, record=False):
    """Optional stopping: after each query, ledger DSR (over the DISTINCT strategies seen so
    far, N >= MIN_N) of the best-so-far strategy. Returns (ever, first query, trajectory)."""
    n = va.shape[0]
    t_all = np.array([tstat(va @ W[:, j]) for j in range(W.shape[1])])
    seen, srs, best_i, ever, first_q, traj = set(), [], None, False, None, []
    for q in range(W.shape[1]):
        key = tuple(np.round(W[:, q], 10))
        if key not in seen:
            seen.add(key); srs.append(t_all[q] / math.sqrt(n))
        if best_i is None or t_all[q] > t_all[best_i]:
            best_i = q
        N_ = len(srs)
        d_ = 0.0
        if N_ >= MIN_N and t_all[best_i] > 2.0:
            d_ = deflated_sharpe(va @ W[:, best_i], n_trials=N_, var_trial_sr=float(np.var(srs, ddof=1)))
        if record:
            traj.append((q + 1, float(t_all[best_i]), float(np.var(srs, ddof=1) * n) if N_ > 1 else 0.0, float(d_)))
        if d_ > 0.95 and not ever:
            ever, first_q = True, q + 1
    return ever, first_q, traj


def episode(path):
    L = [json.loads(l) for l in open(path)]
    end = next((x for x in L if x.get("end")), None)
    if end is None:
        sf = Path(str(path) + ".seed")
        if not sf.exists():
            return None
        m_ = re.search(r"_(naive|sealed)_", path.name)
        end = {"seed": json.loads(sf.read_text())["seed"], "mode": m_.group(1)}
    rng = np.random.default_rng(end["seed"])
    S = {s: stationary_bootstrap_null(S0[s], rng) for s in ("train", "val", "test")}
    va, te = S["val"], S["test"]
    Q = [x for x in L if "q" in x]
    sub = next((x["submit"] for x in reversed(L) if "submit" in x), None)
    W = np.array([vec(x["w"]) for x in Q]).T if Q else np.zeros((len(NAMES), 0))
    Wd = dedupe_ledger(W) if Q else W
    tv_all = np.array([tstat(va @ W[:, j]) for j in range(W.shape[1])]) if Q else np.array([])
    row = dict(tag=path.name.replace("_ledger.jsonl", ""), mode=end["mode"], queries=len(Q),
               distinct=int(Wd.shape[1]), submitted=sub is not None,
               best_val_t=float(tv_all.max()) if len(tv_all) else np.nan)
    m = re.match(r"(claude|coder)_(\w+?)_(prudent|metric|optimize)_(naive|sealed)_(\d+)", row["tag"])
    if m:
        row.update(agent=m.group(1) + "-" + m.group(2), objective=m.group(3))
    nV = float(np.var(sharpe(va @ Wd), ddof=1)) * va.shape[0] if Wd.shape[1] > 1 else np.nan
    row["nV"] = nV
    if end["mode"] == "naive" and len(tv_all) <= 1:
        row.update(ever_certified=False, first_cert_q=None, baseline_ever=False)
    if end["mode"] == "naive" and len(tv_all) > 1:
        ever, first_q, traj = ever_certified(W, va, record=True)
        row.update(ever_certified=ever, first_cert_q=first_q)
        # non-adaptive baseline: same number of queries, random portfolios fixed in advance
        brng = np.random.default_rng(abs(hash(row["tag"])) % 2**32)
        Wb = np.zeros_like(W)
        for j in range(W.shape[1]):
            idx = brng.choice(len(NAMES), brng.integers(1, 51), replace=False)
            Wb[idx, j] = brng.choice([-1.0, 1.0], len(idx))
        Wb /= np.abs(Wb).sum(0, keepdims=True)
        row["baseline_ever"] = ever_certified(Wb, va)[0]
        (RES / "traj").mkdir(exist_ok=True)
        json.dump(traj, open(RES / "traj" / f"{row['tag']}.json", "w"))
    if end["mode"] == "naive":
        w = vec(sub) if sub else (W[:, int(np.argmax(tv_all))] if len(tv_all) else None)
        if w is not None:
            Wl = dedupe_ledger(np.column_stack([Wd, w])) if Wd.shape[1] else w[:, None]
            V = float(np.var(sharpe(va @ Wl), ddof=1)) if Wl.shape[1] > 1 else 0.0
            d = deflated_sharpe(va @ w, n_trials=Wl.shape[1], var_trial_sr=V)
            row.update(t_val=float(tstat(va @ w)), t_test=float(tstat(te @ w)), dsr=d,
                       certified=bool(d > 0.95),
                       span_cert=bool(tstat(va @ w) > span_threshold(len(NAMES), va.shape[0])),
                       n_signals=int((w != 0).sum()))
    else:
        passed = next((x for x in Q if x.get("verdict") == "PASS"), None)
        w = vec(passed["w"]) if passed else (vec(sub) if sub else None)
        row.update(certified=passed is not None, dsr=np.nan,
                   verdicts_used=sum(1 for x in Q if x.get("verdict") in ("PASS", "FAIL")))
        if w is not None:
            row.update(t_val=float(tstat(va @ w)), t_test=float(tstat(te @ w)), n_signals=int((w != 0).sum()))
    return row


def main():
    rows = [r for p in sorted(RES.glob("*_ledger.jsonl")) if not p.name.startswith(("pilot", "_"))
            and (r := episode(p)) is not None]
    d = pd.DataFrame(rows)
    d.to_csv(RES / "summary.csv", index=False)
    g = d.groupby(["agent", "objective", "mode"]).agg(
        n=("tag", "size"), q=("queries", "median"), qd=("distinct", "median"),
        certified=("certified", "mean"), ever=("ever_certified", "mean"), base=("baseline_ever", "mean"), t_val=("t_val", "mean"), best_val=("best_val_t", "mean"),
        t_test=("t_test", "mean"), nsig=("n_signals", "median"))
    print(g.round(2).to_string())
    lab = {"claude-sonnet": "Claude Sonnet", "claude-opus": "Claude Opus",
           "coder-qwen3": "Qwen3-Coder-30B"}
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E6.} Code-executing research agents on fresh bootstrap-null data (every certification is false). ``Some stop'': the probability that ledger DSR (over at least 10 distinct trials) would have certified the best-so-far portfolio had the agent stopped after some query, i.e.\ under optional stopping. ``Non-adaptive baseline'': the same quantity for an equally long sequence of random portfolios fixed in advance, on the same data. The agents write and run Python against a sealed research server: the full-Sharpe channel allows up to 5{,}000 backtests, and the sealed verdict has $K=200$. ``Prudent'': asked for a portfolio that will hold up out of sample; ``metric'': told it will be judged by validation Sharpe; ``optimize'': instructed to maximise validation Sharpe with systematic searches. Models: claude-opus-5-5 and claude-sonnet-5-5 via Claude Code 2.1.287, run 2026-10-03. Queries are medians; $t$ statistics are means.}\label{tab:e6}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{lll c cc ccc cc}\toprule",
             r"agent & objective & channel & episodes & queries & distinct & P(cert.) & P(cert.\ some stop) & non-adaptive baseline & val.\ $t$ & test $t$\\\midrule"]
    prev = None
    for (a, o, mo), r in g.iterrows():
        if prev and prev != (a, o):
            lines.append(r"\midrule" if prev[0] != a else r"\addlinespace")
        prev = (a, o)
        lines.append(f"{lab.get(a, a)} & {o} & {'sealed' if mo == 'sealed' else 'full + DSR'} & {int(r.n)} & {r.q:.0f} & {r.qd:.0f} & {r.certified:.2f} & {'--' if mo == 'sealed' else f'{r.ever:.2f}'} & {'--' if mo == 'sealed' else f'{r.base:.2f}'} & {r.t_val:.2f} & {r.t_test:+.2f}\\\\")
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    (TAB / "e6.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
