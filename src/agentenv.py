"""Environment shared by the LLM-agent (exp4) and RL (exp5) experiments.

An agent proposes a portfolio as JSON {"portfolio": {"<signal>": weight, ...}}.
Signals are the base strategies of realdata.base_library().  The environment scores
proposals on train / validation / test matrices and returns feedback through a
*channel*:
  naive   : train SR and validation SR (full precision)  -- status quo
  verdict : train SR and a sealed PASS/FAIL on validation at the Bonferroni-over-
            queries threshold; at most K queries; the episode ends at the first PASS
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from scipy import stats

from core import deflated_sharpe, sharpe, tstat, dedupe_ledger, span_threshold

MAX_SIGNALS = 20
ANN = math.sqrt(252)


class SignalUniverse:
    def __init__(self, names, mats):
        """names: list[str]; mats: dict split -> (T, k) array."""
        self.names = list(names)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.mats = mats
        self.k = len(self.names)

    def describe(self):
        inds = sorted({n.split("_")[1] for n in self.names})
        fams = sorted({"_".join([n.split("_")[0]] + n.split("_")[2:]) for n in self.names})
        return inds, fams

    def parse(self, text: str):
        """Extract a weight vector from model output; returns (w, n_valid) or (None, 0)."""
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return None, 0
        blob = m.group(0)
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            # tolerate trailing garbage: try progressively shorter prefixes ending in }
            obj = None
            for end in range(len(blob), 0, -1):
                if blob[end - 1] == "}":
                    try:
                        obj = json.loads(blob[:end]); break
                    except json.JSONDecodeError:
                        continue
            if obj is None:
                return None, 0
        port = obj.get("portfolio", obj) if isinstance(obj, dict) else None
        if not isinstance(port, dict):
            return None, 0
        w = np.zeros(self.k)
        n = 0
        for name, wt in list(port.items())[:MAX_SIGNALS]:
            if name in self.idx:
                try:
                    w[self.idx[name]] += float(wt); n += 1
                except (TypeError, ValueError):
                    pass
        if n == 0 or np.abs(w).sum() == 0:
            return None, 0
        return w / np.abs(w).sum(), n

    def sr(self, w, split):
        return float(sharpe(self.mats[split] @ w)) * ANN

    def t(self, w, split):
        return float(tstat(self.mats[split] @ w))


def parse_batch(uni, text, max_n=20):
    """Parse {"portfolios": [{sig: wt, ...} | {"portfolio": {...}}, ...]} -> list of weight vectors."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    blob = m.group(0)
    obj = None
    for end in range(len(blob), 0, -1):
        if blob[end - 1] == "}":
            try:
                obj = json.loads(blob[:end]); break
            except json.JSONDecodeError:
                continue
    if not isinstance(obj, dict):
        return []
    items = obj.get("portfolios", [obj])
    out = []
    for it in items[:max_n] if isinstance(items, list) else []:
        w, n = uni.parse(json.dumps(it)) if isinstance(it, dict) else (None, 0)
        if w is not None:
            out.append(w)
    return out


class Episode:
    def __init__(self, uni: SignalUniverse, channel: str, K: int = 30, alpha: float = 0.05):
        self.u, self.channel, self.K, self.alpha = uni, channel, K, alpha
        self.ledger: list[np.ndarray] = []
        self.done = False
        self.passed = None
        n_val = uni.mats["val"].shape[0]
        self.c_K = float(stats.t.isf(alpha / (K + 1), df=n_val - 1))

    def step(self, w):
        if w is None:
            return "Could not parse a valid portfolio. Reply with JSON only."
        self.ledger.append(w)
        tr = self.u.sr(w, "train")
        if self.channel == "naive":
            va = self.u.sr(w, "val")
            msg = f"Train Sharpe (1970-1999): {tr:.2f}. Validation Sharpe (2000-2012): {va:.2f}."
        else:
            ok = self.u.t(w, "val") > self.c_K
            left = self.K - len(self.ledger)
            msg = f"Train Sharpe (1970-1999): {tr:.2f}. Validation verdict: {'PASS' if ok else 'FAIL'}. Validation queries left: {left}."
            if ok:
                self.passed, self.done = w, True
        if len(self.ledger) >= self.K:
            self.done = True
        return msg

    def step_batch(self, ws):
        if not ws:
            return "Could not parse any valid portfolio. Reply with the JSON object only."
        lines = []
        for j, w in enumerate(ws):
            if self.done:
                lines.append(f"#{j+1}: not evaluated (budget exhausted or research ended).")
                continue
            lines.append(f"#{j+1}: " + self.step(w))
        return "\n".join(lines)

    def final(self):
        """What would be reported.  naive: the ledger's best validation SR, certified by
        ledger DSR > 0.95.  verdict: the PASS portfolio if any (certified), else the
        best-train-SR proposal (not certified)."""
        W = np.array(self.ledger)
        if len(W) == 0:
            return None
        if self.channel == "naive":
            vs = [self.u.sr(w, "val") for w in W]
            w = W[int(np.argmax(vs))]
            va = self.u.mats["val"]
            Wd = dedupe_ledger(W.T)
            nV = float(np.var(sharpe(va @ Wd), ddof=1)) * va.shape[0] if Wd.shape[1] > 1 else 0.0
            d = deflated_sharpe(va @ w, n_trials=Wd.shape[1], var_trial_sr=nV / va.shape[0])
            cert = d > 0.95
        else:
            d = float("nan")
            if self.passed is not None:
                w, cert = self.passed, True
            else:
                w = W[int(np.argmax([self.u.sr(x, "train") for x in W]))]
                cert = False
        va = self.u.mats["val"]
        Wd = dedupe_ledger(W.T)
        return dict(certified=bool(cert), dsr=d, n_queries=len(W), n_distinct=int(Wd.shape[1]),
                    nV=float(np.var(sharpe(va @ Wd), ddof=1)) * va.shape[0] if Wd.shape[1] > 1 else float("nan"),
                    span_cert=bool(self.u.t(w, "val") > span_threshold(self.u.k, va.shape[0])),
                    sr_train=self.u.sr(w, "train"), sr_val=self.u.sr(w, "val"),
                    sr_test=self.u.sr(w, "test"), t_val=self.u.t(w, "val"),
                    t_test=self.u.t(w, "test"), n_signals=int((w != 0).sum()))


def system_prompt(uni: SignalUniverse, channel: str, K: int, batch: int = 0):
    inds, fams = uni.describe()
    fb = ("After each proposal you will see its train-period Sharpe and its validation-period Sharpe."
          if channel == "naive" else
          "After each proposal you will see its train-period Sharpe and a sealed validation verdict "
          "(PASS/FAIL against a fixed significance bar). The research ends at the first PASS.")
    return f"""You are a quantitative researcher searching for a robust daily trading strategy on US industry portfolios.
Every signal trades one industry hedged against the market (industry return minus market return).
Signal names are FAMILY_INDUSTRY or FAMILY_INDUSTRY_PARAMS:
- hold_<IND>: always long the hedged industry
- tsmom_<IND>_<L>: long/short by the sign of the past L-day hedged return, L in 5,21,63,126,252
- ma_<IND>_<s>_<l>: long/short by the sign of the s-day minus l-day moving average, (s,l) in (5,21),(21,63),(50,200)
Industries: {", ".join(inds)}
A portfolio combines up to {MAX_SIGNALS} signals with weights (negative weight = trade the signal in reverse); weights are normalised.
{fb} You have {K} proposals{f" in total; each turn you may submit up to {batch} portfolios at once, like running a backtest sweep" if batch else ""}. Your goal is the highest Sharpe ratio that will hold up on future unseen data (2013 onward).
Reply each turn with one short sentence of reasoning, then """ + ("""a JSON object listing the portfolios to evaluate, e.g.
{"portfolios": [{"tsmom_Banks_63": 1, "ma_Oil_50_200": 0.5}, {"hold_Gold": 1}]}""" if batch else """the portfolio as JSON, e.g.
{"portfolio": {"tsmom_Banks_63": 1, "ma_Oil_50_200": 0.5}}""")
