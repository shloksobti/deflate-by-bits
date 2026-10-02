cd /root/research
(VLLM_MEM=0.72 python experiments/exp4_llm_agent.py Qwen/Qwen2.5-72B-Instruct qwen72b 8 24 40 > results/exp4_qwen72b.log 2>&1; echo DONE72 >> results/queue.log) &
for cfg in "null val" "null train" "actual val" "actual train"; do set -- $cfg; python experiments/exp5_grpo.py Qwen/Qwen2.5-1.5B-Instruct $1 $2 grpo_$1_$2 300 > results/exp5_$1_$2.log 2>&1; echo "DONE grpo $1 $2 $?" >> results/queue.log; done
wait
echo ALLDONE >> results/queue.log
