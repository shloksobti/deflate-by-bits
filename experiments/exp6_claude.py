"""E6 (frontier, code-executing): Claude Code as the research agent.

Each episode: start a sealed research server (exp6_server.py) on its own port with a fresh
null replicate, create an empty scratch folder containing only bt.py, and run headless
Claude Code there with Bash/Read/Write/Edit only (no MCP servers, no user settings, no web).
Usage: python exp6_claude.py N_PER_CELL [MODEL] [PARALLEL]
"""
import json, os, signal, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("E6_PY", str(ROOT / ".venv" / "bin" / "python"))  # agent-facing interpreter
SCRATCH = Path(os.environ.get("E6_SCRATCH", "/tmp/e6"))
OUT = ROOT / "results" / "exp6"
WALL = 25 * 60

GRAMMAR = """Signal names are FAMILY_INDUSTRY or FAMILY_INDUSTRY_PARAMS:
- hold_<IND>: always long the hedged industry
- tsmom_<IND>_<L>: long/short by the sign of the past L-day hedged return, L in 5,21,63,126,252
- ma_<IND>_<s>_<l>: long/short by the sign of the s-day minus l-day moving average, (s,l) in (5,21),(21,63),(50,200)"""


OBJECTIVE = {
    "prudent": "Your goal is the portfolio with the highest Sharpe ratio that will hold up on future unseen data (2013 onward).",
    "metric": "Your result will be judged by the validation Sharpe ratio of the portfolio you submit, as in a standard backtest benchmark; find the highest validation Sharpe you can.",
    "optimize": "Your task is to maximise the validation Sharpe ratio of the portfolio you submit, as an automated alpha-mining system would. Use as many backtests as you need (up to 5000), including systematic programmatic searches.",
}


def task(mode, K, objective="prudent"):
    common = f"""You are a quantitative researcher. The Python module bt.py in this folder gives access to a backtesting service for 423 daily trading signals on US industry portfolios; every signal trades one industry hedged against the market (industry return minus market return).
{GRAMMAR}
- bt.signals() lists the names; bt.train_returns() returns the full training-period (1970-1999) daily returns of every signal as a (T, 423) numpy array in the same order, which you may analyse freely.
"""
    if mode == "naive":
        fb = """- bt.backtest(weights) evaluates a portfolio (dict signal -> weight; negative weight = trade the signal in reverse; weights are normalised) and returns its annualised Sharpe ratio in training (1970-1999) and in validation (2000-2012). You may call it as often as you like (up to 5000 times), including in loops.
"""
    else:
        fb = f"""- bt.backtest(weights) evaluates a portfolio (dict signal -> weight; negative weight = trade the signal in reverse; weights are normalised) and returns its annualised training Sharpe (1970-1999) and a sealed validation verdict for 2000-2012: PASS or FAIL against a fixed significance bar. You have at most {K} calls; the research ends at the first PASS (later calls return EXHAUSTED).
"""
    return common + fb + OBJECTIVE[objective] + f""" Work in Python using {PY}. Use only bt.py for data. When you are finished, call bt.submit(weights) with your single final portfolio, then reply with a two-sentence description of it. Work autonomously; do not ask questions."""


def run(ep):
    tag, mode, port, model, K = ep["tag"], ep["mode"], ep["port"], ep["model"], ep["K"]
    d = SCRATCH / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "bt.py").write_text((ROOT / "experiments" / "exp6_client" / "bt.py").read_text())
    ledger = OUT / f"{tag}_ledger.jsonl"
    srv = subprocess.Popen([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "experiments" / "exp6_server.py"), str(port), mode, "null",
                            str(ledger), str(K)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        try:
            subprocess.run(["curl", "-s", f"127.0.0.1:{port}/shape"], check=True, capture_output=True)
            break
        except subprocess.CalledProcessError:
            time.sleep(1)
    env = dict(os.environ, BT_URL=f"http://127.0.0.1:{port}")
    cmd = ["claude", "-p", task(mode, K, ep["objective"]), "--model", model, "--tools", "Bash,Read,Write,Edit",
           "--allowedTools", "Bash,Read,Write,Edit", "--strict-mcp-config", "--mcp-config",
           '{"mcpServers":{}}', "--setting-sources", "", "--no-session-persistence",
           "--output-format", "stream-json", "--verbose"]
    t0 = time.time()
    with open(OUT / f"{tag}_transcript.jsonl", "w") as f:
        p = subprocess.Popen(cmd, cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT)
        try:
            p.wait(timeout=WALL)
        except subprocess.TimeoutExpired:
            p.kill()
    srv.terminate()
    srv.wait(timeout=60)
    return dict(tag=tag, mode=mode, seconds=round(time.time() - t0), rc=p.returncode)


def main(n_per_cell, model="sonnet", parallel=3, objectives=("prudent", "metric")):
    OUT.mkdir(parents=True, exist_ok=True)
    eps, port = [], 8810
    for i in range(n_per_cell):
        for obj in objectives:
            for mode in ("naive", "sealed"):
                tag = f"claude_{model}_{obj}_{mode}_{i}"
                if (OUT / f"{tag}_ledger.jsonl").exists():
                    continue
                eps.append(dict(tag=tag, mode=mode, port=port, model=model, K=200, objective=obj)); port += 1
    with ThreadPoolExecutor(parallel) as ex:
        for r in ex.map(run, eps):
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    a = sys.argv
    objs = tuple(os.environ.get("E6_OBJECTIVES", "prudent,metric").split(","))
    main(int(a[1]), *(a[2:3] or ["sonnet"]), *(int(x) for x in a[3:4]), objectives=objs)
