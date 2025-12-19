set -euo pipefail

cd /code/hongpaul-sandbox/temp/agent-hgae/

SESSION="alfworld_hgae_3b_7b_seed1"
ENGINE=vllm
SCRIPT_HGAE_3B="run_scripts/hgae/run_hgae_3b.sh"
SCRIPT_HGAE_7B="run_scripts/hgae/run_hgae_7b.sh"

# Run A
SEED_A=1
GPUS_A="0,1,2,3"
GPUS_B="4,5,6,7"

tmux has-session -t $SESSION 2>/dev/null && tmux  kill-session -t $SESSION

# create session + first window
tmux new-session -d -s "$SESSION" -n "3B"
  tmux send-keys -t "$SESSION:3B" \
  "CUDA_VISIBLE_DEVICES=${GPUS_A} bash ${SCRIPT_HGAE_3B} ${ENGINE} ${SEED_A}" C-m

# second window
tmux new-window -t "$SESSION" -n "7B"

tmux send-keys -t "$SESSION:7B" \
  "CUDA_VISIBLE_DEVICES=${GPUS_B} bash ${SCRIPT_HGAE_7B} ${ENGINE} ${SEED_A}" C-m

# detach
# tmux detach -s "$SESSION"

echo "Launched tmux session: $SESSION"
echo "Attach with: tmux attach -t $SESSION"
