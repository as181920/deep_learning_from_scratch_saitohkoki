#!/bin/bash
SESSION=ai
tmux -2 new-session -d -s $SESSION

tmux select-window -t $SESSION:0
tmux rename-window "dl"
tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m
tmux select-pane -t 0
# tmux split-window -t 0 -v
# tmux send-keys -t 0 "bin/dev" C-m
# tmux send-keys -t 1 "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m
# tmux select-pane -t 1

# tmux new-window -t $SESSION:1 -n "rails"
# tmux select-pane -t 0
# tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m
#
# tmux new-window -t $SESSION:2 -n "php"
# tmux select-pane -t 0
# tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m

tmux new-window -t $SESSION:3 -n "temp2"
tmux select-pane -t 0
tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m

tmux new-window -t $SESSION:4 -n "temp3"
tmux select-pane -t 0
tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m

tmux new-window -t $SESSION:5 -n "temp4"
tmux select-pane -t 0
tmux send-keys "cd /home/andersen/as_projects/AI/deep_learning_from_scratch_saitohkoki" C-m

tmux select-window -t $SESSION:0
tmux -2 attach-session -t $SESSION
