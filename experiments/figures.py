"""Paper figures for exp1-3 (and exp4/5 when their results exist)."""
import json, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 200,
    "savefig.bbox": "tight"})
EULER = 0.5772156649015329


def emax(N):
    return (1 - EULER) * stats.norm.ppf(1 - 1 / N) + EULER * stats.norm.ppf(1 - 1 / (N * math.e))


def fig1():
    d = pd.read_csv(RES / "exp1.csv")
    d["cert"] = d.dsr > 0.95
    names = {"best_of_k": "best-of-$k$ (non-adaptive)", "sign_select": "keep winners",
             "rank_ls": "rank long-short", "greedy": "greedy forward"}
    # (a) mean validation t vs k, rho = 0, with theory
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 2.7))
    g = d[d.rho == 0].groupby(["attack", "k"]).t_val.mean()
    ks = np.array(sorted(d.k.unique()))
    kk = np.geomspace(5, 1000, 200)
    for i, a in enumerate(names):
        s = g.loc[a]
        ax[0].plot(s.index, s.values, "o-", color=C[i], ms=3, label=names[a])
    ax[0].plot(kk, np.sqrt(2 * kk / np.pi), "--", color=INK2, lw=1, label=r"$\sqrt{2k/\pi}$ (Thm 1)")
    ax[0].plot(kk, np.sqrt(kk / np.pi), ":", color=INK2, lw=1, label=r"$\sqrt{k/\pi}$")
    ax[0].plot(kk, [emax(x + 1) for x in kk], "-.", color=C[7], lw=1, label="DSR bar $E[\\max_{k+1}]$")
    ax[0].set(xscale="log", xlabel="base strategies $k$", ylabel="validation $t$-stat",
              title="(a) validation $t$, $\\rho=0$")
    ax[0].legend(fontsize=6.2, loc="upper left")
    for j, rho in enumerate([0.0, 0.3, 0.6]):
        gg = d[d.rho == rho].groupby(["attack", "k"]).cert.mean()
        for i, a in enumerate(names):
            s = gg.loc[a]
            ax[j + 1].plot(s.index, s.values, "o-", color=C[i], ms=3)
        ax[j + 1].axhline(0.05, color=INK2, lw=0.8, ls="--")
        ax[j + 1].set(xscale="log", ylim=(-0.03, 1.03), xlabel="base strategies $k$",
                      title=f"({'bcd'[j]}) DSR certifies, $\\rho={rho}$")
    ax[1].set_ylabel("P(ledger DSR > 0.95)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_attack.pdf")


def fig2():
    d = pd.read_csv(RES / "exp2.csv")
    chans = ["naive_dsr", "naive_topj_dsr", "thresholdout", "naive_span", "train_vault", "split_vault",
             "verdict", "verdict_adaptive", "split_verdict", "ladder3"]
    lab = ["greedy+DSR", "top-$j$+DSR", "Thresholdout fb.", "span bound", "train vault", "split vault",
           "verdict (ours)", "verdict adaptive", "split+verdict", "ladder $B$=3"]
    fig, ax = plt.subplots(1, 4, figsize=(10.5, 3.2), sharey=True)
    panels = [(0.0, 1.0, "(a) null: false certification"), (0.6, 1.0, "(b) SR 0.6, stationary"),
              (0.6, 0.5, "(c) SR 0.6, half rotated"), (0.6, 0.0, "(d) SR 0.6, fully rotated")]
    for p, (mu, phi, title) in enumerate(panels):
        s = d[(d.mu == mu) & (d.phi == phi)].groupby("channel").certified.mean()
        vals = [s.get(c, np.nan) for c in chans]
        cols = [C[7], C[7], C[7], C[0], C[0], C[0], C[2], C[2], C[2], C[0]]
        alphas = [0.85, 0.55, 0.35, 0.3, 0.85, 0.55, 0.95, 0.7, 0.5, 0.2]
        for i, v in enumerate(vals):
            ax[p].barh(i, v, color=cols[i], alpha=alphas[i], height=0.7)
            ax[p].text(min(v + 0.02, 0.84), i, f"{v:.2f}", va="center", fontsize=7, color=INK)
        if mu == 0:
            ax[p].axvline(0.05, color=INK2, ls="--", lw=0.8)
        ax[p].set(xlim=(0, 1.05), title=title, xlabel="P(certified)")
        ax[p].set_yticks(range(len(chans)), lab)
    ax[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_channels.pdf")


def fig3():
    nul = pd.read_csv(RES / "exp3_null.csv")
    act = pd.read_csv(RES / "exp3_actual.csv")
    order = ["naive_greedy", "naive_rankls", "naive_signflip", "naive_topj", "naive_span", "train_vault", "verdict", "split_verdict"]
    lab = ["greedy+DSR", "rank L/S+DSR", "sign-flip+DSR", "top-$j$+DSR", "span bound", "train vault", "verdict (ours)", "split+verdict"]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0))
    s = nul.groupby("workflow").certified.mean()
    for i, w in enumerate(order):
        col = C[7] if w.startswith("naive") and w != "naive_span" else (C[2] if "verdict" in w else C[0])
        ax[0].barh(i, s[w], color=col, height=0.7)
        ax[0].text(s[w] + 0.02, i, f"{s[w]:.3f}", va="center", fontsize=7)
    ax[0].axvline(0.05, color=INK2, ls="--", lw=0.8)
    ax[0].set_yticks(range(len(order)), lab); ax[0].invert_yaxis()
    ax[0].set(xlim=(0, 1.15), xlabel="P(certified)", title="(a) bootstrap null, 300 replicates")
    a = act.set_index("workflow").loc[order]
    y = np.arange(len(order))
    ax[1].scatter(a.t_val, y, color=INK2, s=18, label="validation 2000–12", zorder=3)
    ax[1].scatter(a.t_test, y, color=C[1], s=18, label="test 2013–26", zorder=3)
    for i in range(len(order)):
        ax[1].plot([a.t_val.iloc[i], a.t_test.iloc[i]], [i, i], color=GRID, lw=2, zorder=1)
    ax[1].set_yticks(y, lab); ax[1].invert_yaxis()
    ax[1].axvline(0, color=INK2, lw=0.6)
    ax[1].set(xlabel="$t$-statistic", title="(b) actual data: validation vs. test")
    ax[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_real.pdf")





def fig6():
    """E6: optional stopping. Best-so-far validation t and ledger DSR along each Opus
    'optimize' search that made >100 queries."""
    tr = sorted((RES / "exp6" / "traj").glob("claude_opus_optimize_naive_*.json"))
    tr = [(p, json.load(open(p))) for p in tr]
    tr = [(p, t) for p, t in tr if len(t) > 100]
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.8))
    for i, (p, t) in enumerate(tr):
        q = np.array([x[0] for x in t]); bt_ = np.array([x[1] for x in t]); d = np.array([x[3] for x in t])
        col = C[i % 8]
        ax[0].plot(q, bt_, color=col, lw=1.2)
        ax[1].plot(q, d, color=col, lw=1.2)
    ax[0].axhline(23.38, color=INK2, ls=":", lw=1); ax[0].text(3.2, 21.6, "span bound 23.4", fontsize=7, color=INK2)
    ax[0].axvline(423, color=GRID, lw=1); ax[1].axvline(423, color=GRID, lw=1)
    ax[1].axhline(0.95, color=C[7], ls="--", lw=1); ax[1].text(3.2, 0.86, "certification bar 0.95", fontsize=7, color=C[7])
    ax[0].set(xscale="log", xlabel="holdout queries made so far", ylabel="best validation $t$ so far",
              title="(a) Opus search on pure noise")
    ax[1].set(xscale="log", xlabel="holdout queries made so far", ylabel="ledger DSR of best so far",
              title="(b) DSR if the agent stopped here", ylim=(-0.03, 1.03))
    fig.tight_layout(); fig.savefig(FIG / "fig6_opus.pdf")


def fig7():
    d = pd.read_csv(RES / "exp7.csv")
    g = d.groupby(["scenario", "R", "mu", "protocol", "rep"]).false_cert.any().groupby(["scenario", "R", "mu", "protocol"]).mean()
    pw = d[(d.mu > 0) & d.signal].groupby(["scenario", "R", "protocol"]).true_cert.mean()
    curves = [("adaptive", "naive_dsr", "adaptive: full Sharpe + cum. DSR", C[7], "-"),
              ("adaptive", "bonf", "adaptive: full Sharpe + Bonferroni", C[1], "-"),
              ("adaptive", "sealed", "adaptive: sealed verdict (ours)", C[2], "-"),
              ("independent", "split_vault", "independent: split vault", C[0], "--"),
              ("independent", "bonf", "independent: Bonferroni = sealed", INK2, "--")]
    Rs = sorted(d.R.unique())
    fig, ax = plt.subplots(1, 2, figsize=(8.6, 2.9))
    for sc, k, l, c, ls in curves:
        ax[0].plot(Rs, [g.loc[(sc, r, 0.0, k)] for r in Rs], marker="o", ms=3, color=c, ls=ls, label=l)
        ax[1].plot(Rs, [pw.loc[(sc, r, k)] for r in Rs], marker="o", ms=3, color=c, ls=ls, label=l)
    ax[0].axhline(0.05, color=INK2, lw=0.8, ls=":")
    ax[0].set(xscale="log", xlabel="research rounds sharing one holdout ($R$)", ylabel="P(any false certification)",
              title="(a) campaign family-wise error (global null)", ylim=(-0.03, 1.03))
    ax[1].set(xscale="log", xlabel="research rounds sharing one holdout ($R$)", ylabel="power per genuine round",
              title="(b) campaign power", ylim=(-0.03, 1.03))
    for a in ax:
        a.set_xticks(Rs, [str(r) for r in Rs])
    ax[1].legend(fontsize=6.3, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig7_campaign.pdf")


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1(); fig2(); fig3(); fig6(); fig7()
    print("ok")
