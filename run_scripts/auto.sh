#!/usr/bin/env bash
NAME=$1

tmux new-session -d -s "${NAME}" "bash auto_sub.sh"
sleep 1
