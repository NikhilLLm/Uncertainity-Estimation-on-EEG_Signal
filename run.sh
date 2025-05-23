# #!/bin/bash

# # Activate conda environment (adjust if using a different environment)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate eeg_atcnet_tta

# # Working directory
# WORKDIR="/home/teaching/Tushar/ATCNet-EMD-TTA"

# # Array of commands with session names and GPU assignments
# declare -A commands=(

#     ["run_fr10_im3"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 10 --num_runs 10 --imfs 3"
#     ["run_fr20_im3"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 20 --num_runs 10 --imfs 3"
#     ["run_fr30_im3"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 30 --num_runs 10 --imfs 3"
#     ["run_fr40_im3"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 40 --num_runs 10 --imfs 3"
#     ["run_fr50_im3"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 50 --num_runs 10 --imfs 3"

#     ["run_fr10_im5"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --num_arti_FRMs 10 --num_runs 10 --imfs 5"
#     ["run_fr20_im5"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --num_arti_FRMs 20 --num_runs 10 --imfs 5"
#     ["run_fr30_im5"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --num_arti_FRMs 30 --num_runs 10 --imfs 5"
#     ["run_fr40_im5"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 40 --num_runs 10 --imfs 5"
#     ["run_fr50_im5"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 50 --num_runs 10 --imfs 5"

#     ["run_fr10_im7"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 10 --num_runs 10 --imfs 7"
#     ["run_fr20_im7"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 20 --num_runs 10 --imfs 7"
#     ["run_fr30_im7"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 30 --num_runs 10 --imfs 7"
#     ["run_fr40_im7"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 40 --num_runs 10 --imfs 7"
#     ["run_fr50_im7"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 50 --num_runs 10 --imfs 7"
    
#     ["run_fr10_im9"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 10 --num_runs 10 --imfs 9"
#     ["run_fr20_im9"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 20 --num_runs 10 --imfs 9"
#     ["run_fr30_im9"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --num_arti_FRMs 30 --num_runs 10 --imfs 9"
#     ["run_fr40_im9"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 40 --num_runs 10 --imfs 9"
#     ["run_fr50_im9"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 50 --num_runs 10 --imfs 9"

# )

# # Create tmux sessions
# for session in "${!commands[@]}"; do
#     cmd="${commands[$session]}"
#     echo "Starting tmux session: $session with command: $cmd"
#     tmux new-session -d -s "$session" -c "$WORKDIR"
#     tmux send-keys -t "$session" "conda activate eeg_atcnet_tta" C-m
#     tmux send-keys -t "$session" "$cmd" C-m
# done

# echo "All tmux sessions started. List sessions with: tmux list-sessions"

# # to kill all tmux sessions, run:
# # tmux kill-server

#     # ["run_fr1_im1"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 1 --num_runs 2 --imfs 1"
#     # ["run_fr1_im2"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --num_arti_FRMs 1 --num_runs 2 --imfs 2"

#!/bin/bash

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eeg_atcnet_tta

# Working directory
WORKDIR="/home/teaching/Nikhil/ATCNet-EMD-TTA"

# Log file for debugging
LOGFILE="/home/teaching/Nikhil/ATCNet-EMD-TTA/tmux_setup.log"
echo "Starting run.sh at $(date)" > "$LOGFILE"

# Array of commands with session names and GPU assignments
# Test runs: epsilon=0.005, aug_counts=[[4,6,4,1], [8,12,8,1], [12,18,12,2]], num_runs=2
# Session format: test_eps0_005_aug<sum(aug_counts)>_run2
declare -A commands=(
    ["test_eps0_005_aug15_run2"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.005 --aug_counts 4,6,4,1 --num_runs 2"
    ["test_eps0_005_aug29_run2"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.005 --aug_counts 8,12,8,1 --num_runs 2"
    ["test_eps0_005_aug44_run2"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.005 --aug_counts 12,18,12,2 --num_runs 2"
    # Full runs (commented out for testing)
    #["run_eps0_025_aug15"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.025 --aug_counts 4,6,4,1 --num_runs 10"
    #["run_eps0_025_aug29"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.025 --aug_counts 8,12,8,1 --num_runs 10"
    #["run_eps0_025_aug44"]="CUDA_VISIBLE_DEVICES=0 python main_TTA.py --epsilon 0.025 --aug_counts 12,18,12,2 --num_runs 10"
    #["run_eps0_005_aug15"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --epsilon 0.005 --aug_counts 4,6,4,1 --num_runs 10"
    #["run_eps0_005_aug29"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --epsilon 0.005 --aug_counts 8,12,8,1 --num_runs 10"
    #["run_eps0_005_aug44"]="CUDA_VISIBLE_DEVICES=1 python main_TTA.py --epsilon 0.005 --aug_counts 12,18,12,2 --num_runs 10"
    #["run_eps0_01_aug15"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --epsilon 0.01 --aug_counts 4,6,4,1 --num_runs 10"
    #["run_eps0_01_aug29"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --epsilon 0.01 --aug_counts 8,12,8,1 --num_runs 10"
    #["run_eps0_01_aug44"]="CUDA_VISIBLE_DEVICES=2 python main_TTA.py --epsilon 0.01 --aug_counts 12,18,12,2 --num_runs 10"
)

# Create tmux sessions
for session in "${!commands[@]}"; do
    cmd="${commands[$session]}"
    echo "Starting tmux session: $session with command: $cmd" | tee -a "$LOGFILE"
    # Kill existing session if it exists
    tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session"
    echo "Checked for existing session $session; killed if it existed" | tee -a "$LOGFILE"
    # Create new session
    tmux new-session -d -s "$session" -c "$WORKDIR" || {
        echo "Failed to create tmux session: $session" | tee -a "$LOGFILE"
        continue
    }
    # Verify session exists
    tmux has-session -t "$session" 2>/dev/null || {
        echo "Session $session not found after creation" | tee -a "$LOGFILE"
        continue
    }
    # Send commands
    tmux send-keys -t "$session" "conda activate eeg_atcnet_tta" C-m
    tmux send-keys -t "$session" "$cmd" C-m
    echo "Commands sent to session: $session" | tee -a "$LOGFILE"
done

echo "All tmux sessions started. List sessions with: tmux list-sessions" | tee -a "$LOGFILE"
echo "To kill all tmux sessions, run: tmux kill-server" | tee -a "$LOGFILE"
echo "Monitor logs: tail -f /home/teaching/Nikhil/ATCNet-EMD-TTA/myresults/run_progress.log" | tee -a "$LOGFILE"