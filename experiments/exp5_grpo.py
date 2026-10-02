"""Exp 5: RL alpha mining.  GRPO-train a small LLM whose completions are portfolios;
reward = annualised Sharpe on the REWARD split ("val" = status-quo RL alpha miners that
optimise validation performance; "train" = honest, validation kept sealed).
Every scored completion is logged with its train/val/test Sharpe (the full ledger).
Usage: python exp5_grpo.py MODEL_ID {actual|null} {val|train|valpool} OUT_TAG [max_steps]
"""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from realdata import base_library, split, stationary_bootstrap_null
from agentenv import SignalUniverse, system_prompt


def main(model, data, reward_split, tag, max_steps=300):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    R = base_library()
    S = split(R)
    if data == "null":
        r = np.random.default_rng(int(__import__("os").environ.get("NULL_SEED", "1000")))
        S = {s: stationary_bootstrap_null(S[s], r) for s in ("train", "val", "test")}
    uni = SignalUniverse(list(R.columns), S)
    sysmsg = system_prompt(uni, "naive", 1).replace(
        "After each proposal you will see its train-period Sharpe and its validation-period Sharpe. You have 1 proposals. ", "")
    prompt = [{"role": "system", "content": sysmsg},
              {"role": "user", "content": "Propose one portfolio."}]
    ds = Dataset.from_dict({"prompt": [prompt] * (max_steps * 8)})
    ledger_path = ROOT / "results" / f"exp5_{tag}_ledger.jsonl"
    ledger = ledger_path.open("w")
    step = {"n": 0}
    # AlphaGen-style pool reward: reward = marginal improvement of the pooled
    # (combined) portfolio's validation Sharpe; the batch's best improver joins the pool.
    pool = {"w": np.zeros(uni.k)}
    pool_log = (ROOT / "results" / f"exp5_{tag}_pool.jsonl").open("w") if reward_split == "valpool" else None

    def pool_sr(v):
        return uni.sr(v, "val") if np.abs(v).sum() > 0 else 0.0

    def reward(prompts, completions, **kw):
        out, recs_batch = [], []
        for c in completions:
            txt = c[-1]["content"] if isinstance(c, list) else c
            w, n = uni.parse(txt)
            if w is None:
                out.append(-1.0)
                recs_batch.append((-1.0, None))
                ledger.write(json.dumps(dict(call=step["n"], valid=False)) + "\n")
                continue
            rec = dict(call=step["n"], valid=True, n_signals=n,
                       w={uni.names[i]: round(float(w[i]), 4) for i in np.flatnonzero(w)},
                       sr_train=uni.sr(w, "train"), sr_val=uni.sr(w, "val"), sr_test=uni.sr(w, "test"))
            ledger.write(json.dumps(rec) + "\n")
            if reward_split == "valpool":
                base = pool_sr(pool["w"])
                out.append(pool_sr(pool["w"] + w) - base)
            else:
                out.append(rec["sr_" + reward_split])
            recs_batch.append((out[-1], w))
        if reward_split == "valpool" and recs_batch:
            gain, wbest = max(recs_batch, key=lambda x: x[0])
            gain = gain if wbest is not None else 0.0
            if gain > 0:
                pool["w"] = pool["w"] + wbest
            pw = pool["w"]
            if np.abs(pw).sum() > 0:
                pool_log.write(json.dumps(dict(call=step["n"], n_signals=int((pw != 0).sum()),
                    w={uni.names[i]: round(float(pw[i]), 4) for i in np.flatnonzero(pw)},
                    t_val=uni.t(pw, "val"), t_test=uni.t(pw, "test"), sr_val=uni.sr(pw, "val"),
                    sr_test=uni.sr(pw, "test"))) + "\n"); pool_log.flush()
        step["n"] += 1
        ledger.flush()
        return out

    cfg = GRPOConfig(output_dir=str(ROOT / "ckpt" / tag), learning_rate=5e-5,
                     per_device_train_batch_size=32, num_generations=16,
                     gradient_accumulation_steps=1, max_completion_length=320,
                     max_steps=max_steps, logging_steps=5, bf16=True, beta=0.0,
                     temperature=1.0, report_to="none", save_strategy="no",
                     gradient_checkpointing=True, seed=int(__import__("os").environ.get("RL_SEED", "0")))
    lora = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear", task_type="CAUSAL_LM")
    tr = GRPOTrainer(model=model, reward_funcs=reward, args=cfg, train_dataset=ds,
                     peft_config=lora)
    tr.train()
    ledger.close()
    print("done", ledger_path)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4], *(int(x) for x in a[5:]))
