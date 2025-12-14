cd /code/hongpaul-sandbox/temp/hierarchy_agent/

SESSION="alfworld_ppo_grpo_1_5b_seed2"
ENGINE=vllm
SCRIPT="run_scripts/grpo_ppo_1_5b/qwen_ppo_1_5b.sh"
SCRIPT2="run_scripts/grpo_ppo_1_5b/qwen_grpo_1_5b.sh"

SEED=2
GPUS="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION

tmux new-session -d -s "$SESSION" -n "seed${SEED}"

tmux send-keys -t "$SESSION:seed${SEED}" \
  "CUDA_VISIBLE_DEVICES=${GPUS} bash ${SCRIPT} ${ENGINE} ${SEED} && \
   CUDA_VISIBLE_DEVICES=${GPUS} bash ${SCRIPT2} ${ENGINE} ${SEED}" C-m

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"