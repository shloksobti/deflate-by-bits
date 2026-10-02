"""Exp 4: A real LLM research agent (open weights, vLLM) iterating on strategies.

Conditions: data in {actual, null} x channel in {naive, verdict}.  Each null episode
uses its own stationary-bootstrap null replicate, so every certification there is a
false discovery.  All active episodes are batched per turn.
Usage: python exp4_llm_agent.py MODEL_ID OUT_TAG [n_actual] [n_null] [turns]
"""
import json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realdata import base_library, split, stationary_bootstrap_null
from agentenv import SignalUniverse, Episode, system_prompt, parse_batch
import os
BATCH = int(os.environ.get("BATCH", "0"))


def main(model, tag, n_actual=8, n_null=16, turns=30):
    from vllm import LLM, SamplingParams
    R = base_library()
    S = split(R)
    names = list(R.columns)
    rng = np.random.default_rng(2026)
    eps = []
    for ch in ("naive", "verdict"):
        for i in range(n_actual):
            eps.append(dict(data="actual", channel=ch, seed=i, uni=SignalUniverse(names, S)))
        for i in range(n_null):
            r = np.random.default_rng(1000 + i)  # same null replicate i for both channels
            mats = {s: stationary_bootstrap_null(S[s], r) for s in ("train", "val", "test")}
            eps.append(dict(data="null", channel=ch, seed=i, uni=SignalUniverse(names, mats)))
    for e in eps:
        K = turns * BATCH if BATCH else turns
        e["env"] = Episode(e["uni"], e["channel"], K=K)
        e["msgs"] = [{"role": "system", "content": system_prompt(e["uni"], e["channel"], K, BATCH)},
                     {"role": "user", "content": "Propose your first portfolio."}]
        e["log"] = []

    llm = LLM(model=model, max_model_len=int(os.environ.get("MAXLEN", "24576")), gpu_memory_utilization=float(__import__("os").environ.get("VLLM_MEM", "0.9")), seed=0)
    t0 = time.time()
    for turn in range(turns):
        for e in eps:  # end episodes whose context would overflow (~3.2 chars/token)
            if sum(len(m["content"]) for m in e["msgs"]) / 3.2 > int(os.environ.get("MAXLEN", "24576")) - 2600:
                e["env"].done = True
        active = [e for e in eps if not e["env"].done]
        if not active:
            break
        sps = [SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2000 if BATCH else 400, seed=1000 * turn + j)
               for j in range(len(active))]
        outs = llm.chat([e["msgs"] for e in active], sps, use_tqdm=False)
        for e, o in zip(active, outs):
            txt = o.outputs[0].text
            if BATCH:
                ws = parse_batch(e["uni"], txt, BATCH)
                n = len(ws)
                fb = e["env"].step_batch(ws)
            else:
                w, n = e["uni"].parse(txt)
                fb = e["env"].step(w)
            e["msgs"] += [{"role": "assistant", "content": txt},
                          {"role": "user", "content": fb + (" Submit your next batch." if BATCH else " Propose your next portfolio.")}]
            e["log"].append(dict(turn=turn, text=txt, feedback=fb, n_signals=n))
        print(f"turn {turn} active={len(active)} {time.time()-t0:.0f}s", flush=True)

    out = ROOT / "results" / f"exp4_{tag}.jsonl"
    with out.open("w") as f:
        for e in eps:
            fin = e["env"].final()
            f.write(json.dumps(dict(data=e["data"], channel=e["channel"], seed=e["seed"],
                                    model=model, final=fin, log=e["log"])) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], *(int(x) for x in a[3:]))
