"""E6 research server: a sealed process that owns the holdout.

Code-executing agents (Claude Code, open-weight coder models) get only an HTTP API:
  GET  /signals            -> list of signal names
  GET  /train.npy          -> training-period returns (T x k float32), free to use
  POST /backtest {weights} -> naive : {"train_sharpe", "val_sharpe"}       (annualised)
                              sealed: {"train_sharpe", "verdict"}          PASS/FAIL/EXHAUSTED
  POST /submit   {weights} -> records the final answer, returns "ok"
The holdout (validation) and test matrices never leave this process. For null episodes
all three splits are fresh stationary-bootstrap null replicates drawn from an
os.urandom seed that is written to disk only when the server shuts down.
Every query is appended to a server-side ledger (outside the agent's folder).

Usage: python exp6_server.py PORT {naive|sealed} {null|actual} LEDGER_PATH [K]
"""
import json, math, os, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realdata import base_library, split, stationary_bootstrap_null

NAIVE_CAP = 5000


def main(port, mode, data, ledger_path, K=200):
    R = base_library()
    names = list(R.columns)
    S = split(R)
    seed = int.from_bytes(os.urandom(8), "little")
    if data == "null":
        rng = np.random.default_rng(seed)
        S = {s: stationary_bootstrap_null(S[s], rng) for s in ("train", "val", "test")}
    n_val = S["val"].shape[0]
    bar = float(stats.t.isf(0.05 / (K + 1), df=n_val - 1))
    idx = {n: i for i, n in enumerate(names)}
    train_bytes = S["train"].astype(np.float32).tobytes()
    lock = threading.Lock()
    st = {"q": 0, "passed": None, "submitted": None}
    ledger = open(ledger_path, "w")

    def vec(weights):
        w = np.zeros(len(names))
        bad = []
        for k_, v in (weights or {}).items():
            if k_ in idx:
                w[idx[k_]] += float(v)
            else:
                bad.append(k_)
        if np.abs(w).sum() == 0:
            return None, bad
        return w / np.abs(w).sum(), bad

    def sr(w, s):
        x = S[s] @ w
        return float(x.mean() / x.std(ddof=1) * math.sqrt(252))

    def t(w, s):
        x = S[s] @ w
        return float(x.mean() / x.std(ddof=1) * math.sqrt(len(x)))

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, ctype="application/json", raw=None):
            body = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/signals":
                self._send(names)
            elif self.path == "/train.npy":
                self._send(None, "application/octet-stream", train_bytes)
            elif self.path == "/shape":
                self._send(list(S["train"].shape))
            else:
                self._send({"error": "unknown endpoint"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            w, bad = vec(body.get("weights"))
            if w is None:
                return self._send({"error": "no valid signals in weights", "unknown": bad[:5]})
            with lock:
                if self.path == "/backtest":
                    st["q"] += 1
                    rec = {"q": st["q"], "w": {names[i]: float(w[i]) for i in np.flatnonzero(w)}}
                    out = {"train_sharpe": round(sr(w, "train"), 4)}
                    if mode == "naive":
                        if st["q"] > NAIVE_CAP:
                            return self._send({"error": f"query budget of {NAIVE_CAP} exhausted"})
                        out["val_sharpe"] = round(sr(w, "val"), 4)
                    else:
                        if st["passed"] is not None or st["q"] > K:
                            out["verdict"] = "EXHAUSTED"
                        elif t(w, "val") > bar:
                            out["verdict"] = "PASS"
                            st["passed"] = rec["w"]
                        else:
                            out["verdict"] = "FAIL"
                        rec["verdict"] = out["verdict"]
                    ledger.write(json.dumps(rec) + "\n"); ledger.flush()
                    return self._send(out)
                if self.path == "/submit":
                    st["submitted"] = {names[i]: float(w[i]) for i in np.flatnonzero(w)}
                    ledger.write(json.dumps({"submit": st["submitted"]}) + "\n"); ledger.flush()
                    return self._send({"status": "ok"})
            self._send({"error": "unknown endpoint"})

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)

    def _stop(*_):  # SIGTERM/SIGINT -> clean shutdown (SIGINT is ignored in background jobs)
        threading.Thread(target=srv.shutdown, daemon=True).start()

    import signal as _sig
    _sig.signal(_sig.SIGTERM, _stop)
    _sig.signal(_sig.SIGINT, _stop)
    # The seed stays in this process's memory while the agent runs; it is written only at
    # shutdown (below), after the agent process has exited.
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ledger.write(json.dumps({"end": True, "seed": seed, "mode": mode, "data": data, "K": K,
                                 "bar": bar, "queries": st["q"]}) + "\n")
        ledger.close()


if __name__ == "__main__":
    a = sys.argv
    main(int(a[1]), a[2], a[3], a[4], *(int(x) for x in a[5:]))
