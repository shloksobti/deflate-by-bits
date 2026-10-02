"""Generate LaTeX tables / inline numbers for the paper from results/*.csv|jsonl."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, TAB = ROOT / "results", ROOT / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)


def w(name, s):
    (TAB / name).write_text(s.strip() + "\n")


def e1():
    d = pd.read_csv(RES / "exp1.csv")
    d["cert"] = d.dsr > 0.95
    v = d[(d.rho == 0) & (d.attack == "rank_ls") & (d.k == 1000)].t_val.mean()
    w("e1_rankls_1000.tex", f"${v:.2f}$")
    lab = {"best_of_k": "best-of-$k$", "sign_select": "keep winners", "sign_flip": "sign flip",
           "rank_ls": "rank long--short", "greedy": "greedy forward"}
    ks = [10, 50, 100, 1000]
    lines = [r"\begin{table}[h]\centering\small",
             r"\caption{E1 details ($\rho=0$, 200 replicates): mean validation $t$ / mean test $t$ / P(ledger DSR$>0.95$).}\label{tab:e1}",
             r"\begin{tabular}{l" + "c" * len(ks) + "}\\toprule",
             "analyst & " + " & ".join(f"$k={k}$" for k in ks) + r"\\\midrule"]
    g = d[d.rho == 0].groupby(["attack", "k"])[["t_val", "t_test", "cert"]].mean()
    for a, l in lab.items():
        cells = []
        for k in ks:
            if (a, k) in g.index:
                r = g.loc[(a, k)]
                cells.append(f"{r.t_val:.1f} / {r.t_test:+.2f} / {r.cert:.2f}")
            else:
                cells.append("--")
        lines.append(l + " & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    w("e1.tex", "\n".join(lines))


def e2():
    d = pd.read_csv(RES / "exp2.csv")
    chans = ["naive_dsr", "naive_topj_dsr", "thresholdout", "naive_span", "train_vault", "split_vault",
             "verdict", "verdict_adaptive", "split_verdict", "ladder3"]
    lab = ["greedy + ledger DSR", "top-$j$ + ledger DSR", "greedy + Thresholdout feedback", "greedy + span bound",
           "train vault", "split vault", "\\textbf{sealed verdict} ($K=6$)", "\\textbf{sealed verdict}, adaptive ($K=200$)",
           "\\textbf{split + sealed verdict}", "bit-ladder ($B=3$)"]
    cols = [(0.0, 1.0)] + [(mu, phi) for mu in (0.4, 0.6) for phi in (1.0, 0.5, 0.0)]
    g = d.groupby(["mu", "phi", "channel"]).certified.mean()
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E2.} Probability of certification (500 replicates per cell). Column 1 is the null, where the rate is the false-certification rate and the target is $\le0.05$. The other columns give power at annualised signal Sharpe $\mu$ and train/holdout overlap $\phi$.}\label{tab:e2}",
             r"\begin{tabular}{l c ccc ccc}\toprule",
             r" & null & \multicolumn{3}{c}{$\mu=0.4$} & \multicolumn{3}{c}{$\mu=0.6$}\\\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"channel & & $\phi{=}1$ & $0.5$ & $0$ & $\phi{=}1$ & $0.5$ & $0$\\\midrule"]
    for c, l in zip(chans, lab):
        cells = []
        for mu, phi in cols:
            v = g.get((mu, phi, c), np.nan)
            cell = f"{v:.2f}"
            if mu == 0 and v > 0.1:
                cell = r"\textcolor{red!70!black}{" + cell + "}"
            cells.append(cell)
        lines.append(l + " & " + " & ".join(cells) + r"\\")
        if c in ("thresholdout",):
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    w("e2.tex", "\n".join(lines))


def e3():
    nul = pd.read_csv(RES / "exp3_null.csv")
    act = pd.read_csv(RES / "exp3_actual.csv").set_index("workflow")
    order = ["naive_greedy", "naive_rankls", "naive_signflip", "naive_topj", "naive_span", "train_vault", "verdict", "split_verdict"]
    lab = ["greedy + ledger DSR", "rank long--short + ledger DSR", "sign flip + ledger DSR",
           "top-$j$ + ledger DSR", "greedy + span bound", "train vault", "\\textbf{sealed verdict}", "\\textbf{split + sealed verdict}"]
    g = nul.groupby("workflow").certified.agg(["mean", "count"])
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E3.} 423 real hedged industry strategies. Null: false-certification rate over 300 stationary-bootstrap replicates (binomial s.e.\ in parentheses). Actual: validation (2000--2012) and test (2013--2026) $t$ of the reported strategy, and whether it was certified.}\label{tab:e3}",
             r"\begin{tabular}{l c ccc}\toprule",
             r" & null & \multicolumn{3}{c}{actual data}\\\cmidrule(lr){3-5}",
             r"workflow & P(certify) & val.\ $t$ & test $t$ & certified\\\midrule"]
    for o, l in zip(order, lab):
        p, n = g.loc[o, "mean"], g.loc[o, "count"]
        se = np.sqrt(p * (1 - p) / n)
        a = act.loc[o]
        lines.append(f"{l} & {p:.3f} ({se:.3f}) & {a.t_val:.2f} & {a.t_test:+.2f} & {'yes' if a.certified else 'no'}\\\\")
        if o == "naive_topj":
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    w("e3.tex", "\n".join(lines))




def e7():
    d = pd.read_csv(RES / "exp7.csv")
    g = d.groupby(["scenario", "R", "mu", "protocol", "rep"]).false_cert.any().groupby(["scenario", "R", "mu", "protocol"]).mean()
    pw = d[(d.mu > 0) & d.signal].groupby(["scenario", "R", "protocol"]).true_cert.mean()
    Rs = sorted(d.R.unique())
    rows = [("independent", "split_vault", "split vault"), ("independent", "bonf", "Bonferroni reuse = sealed verdict"),
            ("adaptive", "naive_dsr", "full Sharpe + cumulative ledger DSR"), ("adaptive", "bonf", "full Sharpe + Bonferroni"),
            ("adaptive", "sealed", "\\textbf{sealed verdict}")]
    lines = [r"\begin{table}[t]\centering\small",
             r"\caption{\textbf{E7.} Research campaigns of $R$ rounds sharing one holdout (500 campaigns per cell, 4 candidates per round for every protocol). Each cell gives the probability of at least one false certification under the global null, then the power per genuine round ($\mu=0.6$). \emph{Independent}: candidates use training data only, so Bonferroni reuse and the sealed verdict coincide. \emph{Adaptive}: researchers build on earlier holdout results; full-precision channels pick candidates by holdout Sharpe, while the sealed researcher sees only PASS/FAIL.}\label{tab:e7}",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{ll" + "c" * len(Rs) + r"}\toprule",
             "campaign & protocol & " + " & ".join(f"$R={r}$" for r in Rs) + r"\\\midrule"]
    prev = None
    for sc, k, l in rows:
        if prev and prev != sc:
            lines.append(r"\midrule")
        prev = sc
        cells = []
        for r in Rs:
            f = g.loc[(sc, r, 0.0, k)]
            fs = f"{f:.2f}" if f <= 0.1 else r"\textcolor{red!70!black}{" + f"{f:.2f}" + "}"
            cells.append(f"{fs} / {pw.loc[(sc, r, k)]:.2f}")
        lines.append(f"{sc} & {l} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule\end{tabular}}\end{table}"]
    w("e7.tex", "\n".join(lines))


if __name__ == "__main__":
    e1(); e2(); e3(); e7()
    print(sorted(p.name for p in TAB.iterdir()))
