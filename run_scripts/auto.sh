NAME=$1

tmux new-session -s ${NAME} "bash auto_sub.sh" & clear && sleep 1
