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


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig1(); fig2(); fig3()
    print("ok")
