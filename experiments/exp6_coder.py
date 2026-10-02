"""E6b: open-weight code-executing agent (vLLM) against the sealed research server.

The model writes ```python``` blocks; each block runs in a persistent per-episode Python
subprocess whose only data access is bt.py (HTTP to exp6_server.py). Episodes are batched
per turn. Usage: python exp6_coder.py MODEL TAG N_PER_CELL [TURNS]
"""
import json, os, re, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import exp6_claude as C
from exp6_claude import task, OUT

PY = sys.executable
SCRATCH = Path(os.environ.get("E6_SCRATCH", "/tmp/e6coder"))
REPL = r'''
import sys, io, json, contextlib, traceback
G = {"__name__": "__main__"}
for line in sys.stdin:
    code = json.loads(line)["code"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            exec(compile(code, "<cell>", "exec"), G)
        except Exception:
            traceback.print_exc()
    sys.__stdout__.write(json.dumps({"out": buf.getvalue()[-4000:]}) + "\n"); sys.__stdout__.flush()
'''


class Repl:
    def __init__(self, cwd, env):
        self.cwd, self.env = cwd, env
        self.start()

    def start(self):
        self.p = subprocess.Popen([PY, "-u", "-c", REPL], cwd=self.cwd, env=self.env, text=True,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def run(self, code, timeout=180):
        self.p.stdin.write(json.dumps({"code": code}) + "\n"); self.p.stdin.flush()
        import select
        r, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not r:
            self.p.kill(); self.start()
            return "TimeoutError: cell exceeded 180 s and the Python session was restarted."
        return json.loads(self.p.stdout.readline())["out"]


def main(model, tag, n_per_cell, turns=25):
    from vllm import LLM, SamplingParams
    OUT.mkdir(parents=True, exist_ok=True)
    eps, port = [], 8900
    for i in range(n_per_cell):
        for obj in os.environ.get("E6_OBJECTIVES", "prudent,metric").split(","):
            for mode in ("naive", "sealed"):
                t = f"coder_{tag}_{obj}_{mode}_{i}"
                d = SCRATCH / t; d.mkdir(parents=True, exist_ok=True)
                (d / "bt.py").write_text((ROOT / "experiments" / "exp6_client" / "bt.py").read_text())
                srv = subprocess.Popen([PY, str(ROOT / "experiments" / "exp6_server.py"), str(port), mode, "null",
                                        str(OUT / f"{t}_ledger.jsonl"), "200"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                env = dict(os.environ, BT_URL=f"http://127.0.0.1:{port}")
                prompt = task(mode, 200, obj).replace(f"Work in Python using {C.PY}.", "Work in Python.")
                prompt += ("\nEach turn, reply with brief reasoning and exactly one ```python``` code block; it will be "
                           "executed in a persistent Python session (variables persist) and you will see its printed output. "
                           f"You have {turns} turns. After calling bt.submit(...), reply DONE.")
                eps.append(dict(tag=t, srv=srv, env=env, cwd=d, done=False, log=[],
                                msgs=[{"role": "user", "content": prompt}]))
                port += 1
    time.sleep(25)  # servers build the library
    for e in eps:
        e["repl"] = Repl(e["cwd"], e["env"])
    llm = LLM(model=model, max_model_len=32768, gpu_memory_utilization=float(os.environ.get("VLLM_MEM", "0.5")), seed=0)
    for turn in range(turns):
        act = [e for e in eps if not e["done"]]
        if not act:
            break
        for e in act:  # keep context bounded: drop oldest exchanges beyond ~24k tokens
            while sum(len(m["content"]) for m in e["msgs"]) / 3.2 > 26000 and len(e["msgs"]) > 3:
                del e["msgs"][1:3]
        outs = llm.chat([e["msgs"] for e in act],
                        [SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2500, seed=97 * turn + j)
                         for j in range(len(act))], use_tqdm=False)
        for e, o in zip(act, outs):
            txt = o.outputs[0].text
            m = re.findall(r"```python\n(.*?)```", txt, re.S)
            res = e["repl"].run(m[0]) if m else "No ```python``` block found. Reply with one code block."
            e["msgs"] += [{"role": "assistant", "content": txt},
                          {"role": "user", "content": f"Output:\n{res[-3500:]}"}]
            e["log"].append(dict(turn=turn, text=txt, output=res[-3500:]))
            if "DONE" in txt and not m:
                e["done"] = True
        print(f"turn {turn} active={len(act)}", flush=True)
    for e in eps:
        e["srv"].terminate(); e["srv"].wait(timeout=60)
        e["repl"].p.kill()
        with open(OUT / f"{e['tag']}_transcript.json", "w") as f:
            json.dump(e["log"], f)
    print("done")


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], int(a[3]), *(int(x) for x in a[4:5]))
