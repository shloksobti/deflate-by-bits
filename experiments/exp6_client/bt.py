"""Backtest API client. The validation data live on a separate server and cannot be
accessed except through backtest()."""
import json, os, urllib.request

import numpy as np

URL = os.environ.get("BT_URL", "http://127.0.0.1:8765")


def _get(path):
    with urllib.request.urlopen(URL + path) as r:
        return r.read()


def _post(path, obj):
    req = urllib.request.Request(URL + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def signals():
    """List of the 423 signal names."""
    return json.loads(_get("/signals"))


def train_returns():
    """Daily returns of every signal in the training period 1970-1999, shape (T, 423)."""
    shape = json.loads(_get("/shape"))
    return np.frombuffer(_get("/train.npy"), dtype=np.float32).reshape(shape)


def backtest(weights):
    """weights: dict {signal_name: weight}. Returns the portfolio's annualised training
    Sharpe and its validation result (2000-2012)."""
    return _post("/backtest", {"weights": weights})


def submit(weights):
    """Submit your final portfolio (dict {signal_name: weight})."""
    return _post("/submit", {"weights": weights})
