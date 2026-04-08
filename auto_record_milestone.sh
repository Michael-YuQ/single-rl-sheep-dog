#!/bin/bash
# Wait for milestone training to reach 1M steps, then record videos
TARGET_DIR="/inspire/hdd/global_user/yuqi-253114050256/sheep/milestonesheepv1"
LOG="/root/sheep-dog/results/train_milestone.log"
SCRIPT_DIR="/root/sheep-dog"

source /root/miniforge3/etc/profile.d/conda.sh
conda activate sheep

echo "Watching for 1M steps..."
while true; do
    STEP=$(grep "Agent Step" "$LOG" | tail -1 | grep -oP '\d[\d_]*' | tr -d '_')
    echo "Current step: $STEP"
    if [ "${STEP:-0}" -ge 990000 ]; then
        echo "Reached ~1M steps, recording videos..."
        break
    fi
    sleep 60
done

cd "$SCRIPT_DIR"
mkdir -p "$TARGET_DIR"

for SEED in 1 2 3 4 5 7 9 11 13 42; do
    echo "Recording seed=$SEED ..."
    python record_milestone_compare.py --n_sheep 100 --seed $SEED --max_steps 600 2>&1 | grep -E "Recording|steps|frac|Video|Done"
    cp results/milestone_compare_N100_seed${SEED}.mp4 "$TARGET_DIR/" 2>/dev/null || true
done

echo "All videos saved to $TARGET_DIR"
ls -lh "$TARGET_DIR/"
