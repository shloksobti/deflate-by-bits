"""Analysis + figures + tables for E4 (LLM agents) and E5 (GRPO)."""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
from realdata import base_library, split, stationary_bootstrap_null
from agentenv import SignalUniverse, parse_batch, Episode
from core import deflated_sharpe, sharpe, tstat, expected_max_z
from figures import C, INK2, GRID, plt
from scipy import stats

RES, FIG, TAB = ROOT / "results", ROOT / "figures", ROOT / "paper" / "tables"
R = base_library(); S = split(R); NAMES = list(R.columns)
_null_cache = {}


def universe(data, seed):
    if data == "actual":
        return SignalUniverse(NAMES, S)
    if seed not in _null_cache:
        r = np.random.default_rng(1000 + seed)
        _null_cache[seed] = SignalUniverse(NAMES, {s: stationary_bootstrap_null(S[s], r) for s in ("train", "val", "test")})
    return _null_cache[seed]


def load_e4(tag):
    rows, curves = [], []
    for l in open(RES / f"exp4_{tag}.jsonl"):
        e = json.loads(l)
        u = universe(e["data"], e["seed"])
        if "batch" in tag:
            W = [w for x in e["log"] for w in parse_batch(u, x["text"], 20)][: (e["final"] or {}).get("n_queries", 10**9)]
        else:
            W = [w for w, _ in (u.parse(x["text"]) for x in e["log"]) if w is not None]
        # replay through the current environment (distinct-trial ledger convention)
        ep = Episode(u, e["channel"], K=300 if "batch" in tag else 40)
        for w in W:
            if ep.done:
                break
            ep.step(w)
        f = ep.final() or {}
        rows.append(dict(model=tag, data=e["data"], channel=e["channel"], seed=e["seed"], **f,
                         parse_fail=sum(x["n_signals"] == 0 for x in e["log"])))
        if W:
            va = u.mats["val"]
            tv = np.array([tstat(va @ w) for w in W])
            curves.append(dict(model=tag, data=e["data"], channel=e["channel"], seed=e["seed"],
                               run_max=np.maximum.accumulate(tv), n=len(W),
                               sig=[int((w != 0).sum()) for w in W]))
    return pd.DataFrame(rows), curves


def dsr_required_t(N, nV=1.0):
    return math.sqrt(nV) * expected_max_z(max(N, 2)) + stats.norm.ppf(0.95)


def e4(tags):
    dfs, allc = [], []
    for t in tags:
        if (RES / f"exp4_{t}.jsonl").exists():
            d, c = load_e4(t); dfs.append(d); allc += c
    d = pd.concat(dfs)
    d.to_csv(RES / "exp4_summary.csv", index=False)
    g = d.groupby(["model", "data", "channel"]).agg(
        n=("certified", "size"), q=("n_queries", "mean"), qd=("n_distinct", "mean"), nV=("nV", "median"),
        certified=("certified", "mean"), t_val=("t_val", "mean"),
        t_test=("t_test", "mean"), n_signals=("n_signals", "mean"), parse_fail=("parse_fail", "mean"))
    print(g.round(3).to_string())
    lab = {"qwen7b": "Qwen2.5-7B", "qwen72b": "Qwen2.5-72B", "qwen72b_batch": "Qwen2.5-72B, batch tool (20/turn)"}
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E4.} LLM research agents (40 turns of one proposal, or 15 turns of up to 20 proposals with the batch tool; ``queries'' is the mean number of holdout queries actually made). Null rows: each episode uses its own bootstrap-null replicate, so every certification is false. Naive channel: report the best-validation proposal, certified by ledger DSR$>0.95$. Verdict channel: sealed PASS/FAIL with budget $K$ equal to the query allowance. Means over episodes.}\label{tab:e4}",
             r"\begin{tabular}{ll l cc ccc}\toprule",
             r"agent & data & channel & episodes & queries (distinct) & P(certify) & val.\ $t$ & test $t$\\\midrule"]
    lab2 = {"qwen7b": "Qwen2.5-7B", "qwen72b": "Qwen2.5-72B", "qwen72b_batch": "72B + batch tool"}
    prev = None
    for m in ["qwen7b", "qwen72b", "qwen72b_batch"]:
        for (mm, da, ch), r in g.iterrows():
            if mm != m:
                continue
            if prev is not None and prev != m:
                lines.append(r"\midrule")
            prev = m
            lines.append(f"{lab2[m]} & {da} & {'sealed verdict' if ch == 'verdict' else 'full Sharpe + DSR'} & {int(r.n)} & {r.q:.0f} ({r.qd:.0f}) & {r.certified:.3f} & {r.t_val:.2f} & {r.t_test:+.2f}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (TAB / "e4.tex").write_text("\n".join(lines) + "\n")

    # figure: running max validation t over turns (null, naive) vs DSR-required t
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.7))
    panels = [("null", "(a) null data, 1 proposal/turn", [t for t in tags if "batch" not in t]),
              ("actual", "(b) actual data, 1 proposal/turn", [t for t in tags if "batch" not in t]),
              ("null", "(c) null data, batch tool", [t for t in tags if "batch" in t])]
    for j, (data, title, ptags) in enumerate(panels):
        for i, t in enumerate(ptags):
            i = tags.index(t)
            cs = [c for c in allc if c["model"] == t and c["data"] == data and c["channel"] == "naive"]
            if not cs:
                continue
            L = max(len(c["run_max"]) for c in cs)
            M = np.array([np.pad(c["run_max"], (0, L - len(c["run_max"])), mode="edge") for c in cs])
            x = np.arange(1, L + 1)
            ax[j].plot(x, M.mean(0), color=C[i], label=lab.get(t, t))
            ax[j].fill_between(x, np.percentile(M, 10, 0), np.percentile(M, 90, 0), color=C[i], alpha=0.15, lw=0)
        x = np.arange(1, 41 if j < 2 else 301)
        ax[j].plot(x, [dsr_required_t(n) for n in x], "--", color=C[7], lw=1, label="DSR-required $t$")
        ax[j].plot(x, [stats.norm.isf(0.05 / n) for n in x], ":", color=INK2, lw=1, label="sealed-verdict bar")
        ax[j].set(xlabel="holdout queries", title=title)
        ax[j].legend(fontsize=6.5, loc="lower right")
    ax[0].set_ylabel("best validation $t$ so far")
    fig.tight_layout(); fig.savefig(FIG / "fig4_agents.pdf")
    return d


def e5(tags):
    out = {}
    fig, ax = plt.subplots(1, len(tags), figsize=(2.6 * len(tags), 2.6), sharey=True)
    rows = []
    for p, tag in enumerate(tags):
        path = RES / f"exp5_{tag}_ledger.jsonl"
        if not path.exists():
            continue
        L = [json.loads(l) for l in open(path)]
        df = pd.DataFrame([x for x in L if x["valid"]])
        df["step"] = df.call
        b = df.groupby(df.step // 10)[["sr_train", "sr_val", "sr_test"]].mean()
        x = b.index * 10
        poolf = RES / f"exp5_{tag}_pool.jsonl"
        if tag.endswith("valpool") and poolf.exists():
            P = pd.DataFrame([json.loads(l) for l in open(poolf)])
            b = P.groupby(P.call // 10)[["sr_val", "sr_test"]].last()
            b["sr_train"] = np.nan
            x = b.index * 10
        for i, (col, lab_) in enumerate([("sr_train", "train"), ("sr_val", "validation"), ("sr_test", "test")]):
            ax[p].plot(x, b[col], color=[INK2, C[0], C[1]][i], label=lab_)
        ax[p].axhline(0, color=GRID, lw=1)
        _, data, rew = tag.split("_")
        ax[p].set(title=f"{data} data, reward = {'val. pool' if rew == 'valpool' else rew}", xlabel="GRPO step")
        # reported strategy: best by the reward split among distinct portfolios
        df["key"] = df.w.apply(lambda w: json.dumps(w, sort_keys=True))
        dd = df.drop_duplicates("key")
        best = dd.loc[dd["sr_" + ("val" if rew == "valpool" else rew)].idxmax()]
        u = universe(data, 0) if data == "actual" else None
        if data == "null":
            r = np.random.default_rng(1000)
            u = SignalUniverse(NAMES, {s: stationary_bootstrap_null(S[s], r) for s in ("train", "val", "test")})
        def vec(wd):
            v = np.zeros(len(NAMES))
            for k_, val in wd.items():
                v[NAMES.index(k_)] = val
            return v
        va = u.mats["val"]
        wb = vec(best.w)
        if rew == "valpool":
            P = [json.loads(l) for l in open(poolf)]
            wb = vec(P[-1]["w"])
            Wl = np.array([vec(w) for w in dd.w] + [vec(x["w"]) for x in P]).T
            from core import dedupe_ledger
            Wl = dedupe_ledger(Wl)
            dsr = deflated_sharpe(va @ wb, n_trials=Wl.shape[1], var_trial_sr=float(np.var(sharpe(va @ Wl), ddof=1)))
            cert = dsr > 0.95
        elif rew == "val":
            Wl = np.array([vec(w) for w in dd.w]).T
            dsr = deflated_sharpe(va @ wb, n_trials=Wl.shape[1], var_trial_sr=float(np.var(sharpe(va @ Wl), ddof=1)))
            cert = dsr > 0.95
        else:
            # sealed verdict over the K=40 distinct portfolios with highest train SR
            top = dd.sort_values("sr_train", ascending=False).head(40)
            bar = stats.t.isf(0.05 / 41, df=va.shape[0] - 1)
            dsr, cert = float("nan"), False
            for _, rr in top.iterrows():
                if tstat(va @ vec(rr.w)) > bar:
                    wb, cert = vec(rr.w), True
                    break
            if not cert:
                wb = vec(top.iloc[0].w)
        rows.append(dict(run=tag, data=data, reward=rew, completions=len(L), valid=len(df),
                         distinct=len(dd), t_val=float(tstat(va @ wb)), t_test=float(tstat(u.mats["test"] @ wb)),
                         dsr=dsr, certified=bool(cert),
                         final_val=float(b.sr_val.iloc[-3:].mean()), final_test=float(b.sr_test.iloc[-3:].mean())))
    ax[0].set_ylabel("mean annualised Sharpe of samples")
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "fig5_grpo.pdf")
    t = pd.DataFrame(rows)
    print(t.round(3).to_string())
    t.to_csv(RES / "exp5_summary.csv", index=False)
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E5.} GRPO on Qwen2.5-1.5B, 300 steps $\times$ 32 completions. Reported strategy: the best distinct portfolio by the reward split. Validation-reward runs are certified by ledger DSR over all distinct portfolios; the pool run reports the final pool, with the ledger comprising all distinct portfolios and pool states; train-reward runs by the sealed verdict over the 40 best-by-train portfolios.}\label{tab:e5}",
             r"\begin{tabular}{ll ccc cc}\toprule",
             r"data & reward & distinct portfolios & val.\ $t$ & test $t$ & DSR & certified\\\midrule"]
    for _, r in t.iterrows():
        dsr = "--" if np.isnan(r.dsr) else f"{r.dsr:.3f}"
        rw = {"val": "validation", "train": "train (sealed)", "valpool": "validation pool"}[r.reward]
        lines.append(f"{r.data} & {rw} & {r.distinct} & {r.t_val:.2f} & {r.t_test:+.2f} & {dsr} & {'yes' if r.certified else 'no'}\\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (TAB / "e5.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "both"
    if what in ("e4", "both"):
        e4(["qwen7b", "qwen72b", "qwen72b_batch"])
    if what in ("e5", "both"):
        e5([t for t in ["grpo_null_val", "grpo_null_train", "grpo_null_valpool", "grpo_actual_val", "grpo_actual_train"]
            if (RES / f"exp5_{t}_ledger.jsonl").exists()])
