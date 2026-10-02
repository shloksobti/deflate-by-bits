cd /root/research
until grep -q "DONE grpo actual train" results/queue.log 2>/dev/null; do sleep 30; done
python experiments/exp5_grpo.py Qwen/Qwen2.5-1.5B-Instruct null valpool grpo_null_valpool 300 > results/exp5_null_valpool.log 2>&1; echo "DONE grpo null valpool $?" >> results/queue.log
