# Deflate by Bits, Not Trials

Code, data pipeline and raw results for *"Deflate by Bits, Not Trials: Why Query-Counted
Deflated Sharpe Ratios Fail Under Adaptive Search, and a Sealed-Holdout Remedy"*
(Shlok Sobti, 2026). **Paper:** [SSRN 7557458](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458) (also `paper/main.pdf`).

**Claim.** Counting backtests run (Deflated Sharpe Ratio on a query ledger) is invalid when
the researcher is adaptive: a "combine what worked" ensemble over k pure-noise strategies
reaches validation t ~ sqrt(2k/pi), versus a DSR bar of ~ sqrt(2 ln k). Certifying by the
*information the holdout reveals* (log |transcripts|) is valid for any analyst; a sealed
PASS/FAIL holdout with budget K costs only log K. Claude Opus, instructed to maximise validation
Sharpe, executes the attack autonomously (validation t up to 15.7 on pure noise) and is certified by
ledger DSR under optional stopping; a GRPO miner with a pool reward learns the same ensemble and is
DSR-certified. The sealed verdict stays valid throughout, including across adaptive campaigns.

## Layout
| path | what |
|---|---|
| `src/core.py` | Sharpe/DSR, holdout channels (naive, thresholdout, sealed ladder), attacks, certificates |
| `src/realdata.py` | 423 hedged industry strategies from Ken French daily data; stationary-bootstrap null |
| `src/agentenv.py` | environment for LLM agents and RL (JSON portfolios, naive vs sealed-verdict feedback) |
| `experiments/exp1_attack_vs_dsr.py` | E1: attacks vs ledger DSR on synthetic noise |
| `experiments/exp2_channels.py` | E2: validity/power of feedback channels, regime rotation |
| `experiments/exp3_real.py` | E3: real data + 300 bootstrap-null replicates |
| `experiments/exp4_llm_agent.py` | E4: Qwen2.5 7B/72B research agents via vLLM (GPU) |
| `experiments/exp5_grpo.py` | E5: GRPO alpha mining with validation / train / pool rewards (GPU) |
| `experiments/exp6_server.py`, `exp6_claude.py`, `exp6_coder.py` | E6: sealed research server; Claude Code and open-weight coder agents |
| `experiments/exp7_campaign.py` | E7: multi-round campaigns sharing one holdout (independent and adaptive) |
| `experiments/figures.py`, `tables.py`, `analyze_e4e5.py`, `analyze_e6.py` | figures and LaTeX tables |
| `experiments/gpu/` | GPU job scripts used for E4/E5 |
| `scripts/get_data.sh` | downloads the Ken French data into `data/` |
| `results/` | raw CSV/JSONL outputs and logs |

## Reproduce
```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python numpy scipy pandas matplotlib
scripts/get_data.sh   # Ken French daily data (paper used the CRSP 2026-08 vintage)
.venv/bin/python experiments/exp1_attack_vs_dsr.py
.venv/bin/python experiments/exp2_channels.py 500
.venv/bin/python experiments/exp3_real.py 300
.venv/bin/python experiments/exp7_campaign.py 500
E6_OBJECTIVES=prudent,metric,optimize .venv/bin/python experiments/exp6_claude.py 6 sonnet 3   # needs Claude Code
# GPU (vLLM + TRL): see experiments/gpu/; ran on 1x AMD MI325X, ROCm 7.14, vLLM 0.23, TRL 1.14
.venv/bin/python experiments/figures.py && .venv/bin/python experiments/tables.py && .venv/bin/python experiments/analyze_e4e5.py
cd paper && tectonic main.tex
```

## Citation
```
@misc{sobti2026deflate,
  title  = {Deflate by Bits, Not Trials: Why Query-Counted Deflated Sharpe Ratios Fail Under Adaptive Search, and a Sealed-Holdout Remedy},
  author = {Sobti, Shlok},
  year   = {2026},
  note   = {SSRN Working Paper 7557458},
  url    = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7557458}
}
```

Analysis, code and drafting were done with AI assistance (Claude, Anthropic); the author is responsible for all content.
