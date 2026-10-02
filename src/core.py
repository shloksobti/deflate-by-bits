"""Core library: Sharpe statistics, Deflated Sharpe Ratio, holdout-access mechanisms,
adaptive attacks, and certificates.

Conventions
-----------
* A *base universe* is a matrix R of shape (n_periods, k) of per-period returns.
* A *strategy* is a weight vector w in R^k; its return series is R @ w.
* Sharpe ratios are per-period (not annualised); t-stat = SR * sqrt(n).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.special import gammaln, logsumexp

EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def sharpe(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Per-period Sharpe ratio (mean / sample std)."""
    return x.mean(axis=axis) / x.std(axis=axis, ddof=1)


def tstat(x: np.ndarray, axis: int = 0) -> np.ndarray:
    n = x.shape[axis]
    return sharpe(x, axis=axis) * math.sqrt(n)


def expected_max_z(N: int) -> float:
    """Bailey & Lopez de Prado (2014) approximation to E[max of N iid N(0,1)]."""
    if N <= 1:
        return 0.0
    return (1 - EULER_GAMMA) * stats.norm.ppf(1 - 1.0 / N) + EULER_GAMMA * stats.norm.ppf(
        1 - 1.0 / (N * math.e)
    )


def deflated_sharpe(
    returns: np.ndarray, n_trials: int, var_trial_sr: float
) -> float:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), eq. (2).

    returns       : return series of the selected strategy on the evaluation set
    n_trials      : number of trials N recorded in the ledger
    var_trial_sr  : cross-sectional variance of the per-period SR estimates of the N trials
    Returns the probability that the true SR exceeds the selection-adjusted benchmark SR0.
    """
    T = len(returns)
    sr = sharpe(returns)
    g3 = stats.skew(returns)
    g4 = stats.kurtosis(returns, fisher=False)
    sr0 = math.sqrt(max(var_trial_sr, 0.0)) * expected_max_z(n_trials)
    denom = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * math.sqrt(T - 1) / denom))


# ----------------------------------------------------------------------------
# Holdout access mechanisms ("oracles").  Each wraps a sealed holdout matrix and
# exposes only `query(w) -> float | bool`.  A ledger of every query is kept.
# ----------------------------------------------------------------------------
@dataclass
class Oracle:
    R_hold: np.ndarray
    R_train: np.ndarray | None = None
    ledger: list = field(default_factory=list)  # list of weight vectors queried

    def __post_init__(self):
        self._col_hold = sharpe(self.R_hold)
        self._col_train = None if self.R_train is None else sharpe(self.R_train)

    @staticmethod
    def _unit(w):
        nz = np.flatnonzero(w)
        return int(nz[0]) if len(nz) == 1 and w[nz[0]] > 0 else None

    def _hold_sr(self, w):
        i = self._unit(w)
        return float(self._col_hold[i]) if i is not None else float(sharpe(self.R_hold @ w))

    def _train_sr(self, w):
        i = self._unit(w)
        return float(self._col_train[i]) if i is not None else float(sharpe(self.R_train @ w))

    @property
    def n_queries(self):
        return len(self.ledger)


class NaiveOracle(Oracle):
    """Returns the exact holdout Sharpe ratio. What every backtest engine does."""

    def query(self, w):
        self.ledger.append(w)
        return self._hold_sr(w)


class ThresholdoutOracle(Oracle):
    """Thresholdout (Dwork et al., Science 2015) applied to Sharpe ratios.

    Answers with the training-set SR unless it disagrees with the holdout SR by more
    than a noisy threshold, in which case a noisy holdout SR is returned and the budget
    is decremented.  Scale parameters are in per-period SR units.
    """

    def __init__(self, R_hold, R_train, threshold, sigma, budget, rng):
        super().__init__(R_hold, R_train)
        self.__post_init__()
        self.T, self.sigma, self.budget, self.rng = threshold, sigma, budget, rng
        self.T_hat = threshold + rng.laplace(0, 2 * sigma)
        self.overfit_events = 0

    def query(self, w):
        self.ledger.append(w)
        tr, ho = self._train_sr(w), self._hold_sr(w)
        if self.budget <= 0:
            return tr
        if abs(tr - ho) > self.T_hat + self.rng.laplace(0, 4 * self.sigma):
            self.budget -= 1
            self.overfit_events += 1
            self.T_hat = self.T + self.rng.laplace(0, 2 * self.sigma)
            return ho + self.rng.laplace(0, self.sigma)
        return tr


class SealedLadderOracle(Oracle):
    """Sealed bit-ladder (adapted from Blum & Hardt's Ladder, ICML 2015).

    Each query reveals one bit: does w's holdout t-stat beat the best accepted t-stat
    so far by at least `eta`?  At most `budget` positive answers are ever given; after
    that every query returns False.  The transcript is therefore a subset of at most
    `budget` query indices, which is what makes certification possible (see
    `ladder_threshold`).
    """

    def __init__(self, R_hold, eta, budget, start=0.0):
        super().__init__(R_hold)
        self.__post_init__()
        self.eta, self.budget = eta, budget
        self.best = start
        self.accepted: list[int] = []
        self.n = R_hold.shape[0]

    def query(self, w):
        self.ledger.append(w)
        if len(self.accepted) >= self.budget:
            return False
        t = self._hold_sr(w) * math.sqrt(self.n)
        if t >= self.best + self.eta:
            self.best = t
            self.accepted.append(len(self.ledger) - 1)
            return True
        return False


# ----------------------------------------------------------------------------
# Certificates
# ----------------------------------------------------------------------------
def log_num_transcripts(k: int, B: int) -> float:
    """log of sum_{b=0}^{B} C(k, b): the number of possible bit-ladder transcripts."""
    b = np.arange(0, min(B, k) + 1)
    logs = gammaln(k + 1) - gammaln(b + 1) - gammaln(k - b + 1)
    return float(logsumexp(logs))


def span_threshold(k: int, n: int, alpha: float = 0.05) -> float:
    """Scheffe / Hotelling (GRS-style) bar: valid for ANY analyst, at any feedback
    precision, whose reported strategy lies in the span of k fixed base strategies,
    because max_w t(w)^2 = n rbar' S^-1 rbar ~ k(n-1)/(n-k) F(k, n-k) under Gaussian H0."""
    return float(math.sqrt(k * (n - 1) / (n - k) * stats.f.ppf(1 - alpha, k, n - k)))


def dedupe_ledger(W: np.ndarray) -> np.ndarray:
    """Columns of W (k, N) with exact duplicates removed (one trial per distinct strategy)."""
    _, idx = np.unique(np.round(W.T, 10), axis=0, return_index=True)
    return W[:, np.sort(idx)]


def ladder_threshold(k: int, B: int, alpha: float, n: int | None = None) -> float:
    """Holdout t-stat threshold valid for ANY adaptive analyst interacting with a
    SealedLadderOracle for at most k queries and B positive answers.

    By a union bound over the <= sum_b C(k,b) transcripts (each fixes the analyst's
    final strategy as a function independent of the holdout), the reported strategy
    exceeds this threshold under the null with probability <= alpha.
    Uses Student-t with n-1 df when n is given, else Gaussian.
    """
    log_m = log_num_transcripts(k, B)
    log_p = math.log(alpha) - log_m
    if n is None:
        return float(-stats.norm.ppf(math.exp(log_p))) if log_p > -700 else float(
            math.sqrt(-2 * log_p)
        )
    return float(stats.t.isf(math.exp(log_p), df=n - 1))


# ----------------------------------------------------------------------------
# Adaptive analysts ("attacks" / agents) on a base universe.
# Each takes an oracle and returns the final weight vector it would report.
# ----------------------------------------------------------------------------
def _e(k, i):
    v = np.zeros(k)
    v[i] = 1.0
    return v


def best_of_k(oracle, k, rng=None):
    """Non-adaptive baseline: query k unit strategies, report the best."""
    scores = [oracle.query(_e(k, i)) for i in range(k)]
    w = np.zeros(k)
    w[int(np.argmax(scores))] = 1.0
    return w


def sign_select_ensemble(oracle, k, rng=None):
    """Boosting attack variant 1: keep the base strategies with positive holdout SR,
    equal-weight them, and query the ensemble once."""
    scores = np.array([oracle.query(_e(k, i)) for i in range(k)])
    w = (scores > 0).astype(float)
    if w.sum() == 0:
        w[int(np.argmax(scores))] = 1.0
    w /= w.sum()
    oracle.query(w)
    return w


def sign_flip_ensemble(oracle, k, rng=None):
    """Boosting attack variant 2: long the winners, short the losers."""
    scores = np.array([oracle.query(_e(k, i)) for i in range(k)])
    w = np.sign(scores) / k
    oracle.query(w)
    return w


def rank_long_short(oracle, k, rng=None):
    """Boosting attack variant 3 (factor-neutral): long the above-median base
    strategies, short the below-median ones.  Cancels common factors, so the attack
    survives cross-strategy correlation."""
    scores = np.array([oracle.query(_e(k, i)) for i in range(k)])
    w = np.sign(scores - np.median(scores)) / k
    oracle.query(w)
    return w


def greedy_forward(oracle, k, max_steps=None, rng=None):
    """'Agent-like' hill-climbing: start empty; repeatedly try adding (+/-) each base
    strategy; keep an addition if the oracle says it improves.  Works with numeric
    oracles (compare scores) and bit oracles (True = improvement)."""
    w = np.zeros(k)
    best = -np.inf
    order = rng.permutation(k) if rng is not None else np.arange(k)
    for i in order[: (max_steps or k)]:
        for s in (+1.0, -1.0):
            cand = w.copy()
            cand[i] += s
            ans = oracle.query(cand / np.abs(cand).sum())
            if isinstance(ans, (bool, np.bool_)):
                if ans:
                    w = cand
                    break
            elif ans > best:
                best, w = ans, cand
                break
    if np.abs(w).sum() == 0:
        w[order[0]] = 1.0
    return w / np.abs(w).sum()


def ledger_dsr(oracle: Oracle, w: np.ndarray, R_eval: np.ndarray) -> float:
    """DSR computed exactly as a trial-ledger system would: N = all queries recorded,
    V[SR] = cross-sectional variance of the ledger's SR estimates on R_eval."""
    W = dedupe_ledger(np.array(oracle.ledger).T)  # (k, N distinct)
    srs = sharpe(R_eval @ W)
    return deflated_sharpe(R_eval @ w, n_trials=W.shape[1], var_trial_sr=float(np.var(srs, ddof=1)))
