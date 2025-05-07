#!/bin/bash
# memo


model_name=cogvideox_t2v
mode=fg
gpu=4

# Function to load text file into an array
load_txt() {
    mapfile -t "$1" < "$2"
}

# Load video IDs and prompts into arrays
load_txt video_ids "./dataset/cotracker/$model_name/${mode}_50.txt"
load_txt video_prompts "./dataset/txt_prompts/foreground.txt"

# load_txt video_ids "/scratch/slurm-user24-kaist/jisu/finetrainers_tracking/dataset/bg_50.txt"
# load_txt video_prompts "/scratch/slurm-user24-kaist/jisu/finetrainers_tracking/dataset/txt_prompts/augmented_scene.txt"

TOTAL_PROMPTS=50
OUTPUT_DIR="./original_debug"  # Define your output directory here

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

for id in "${video_ids[@]}"; do
    PROMPT="${video_prompts[$id]}"  # Get prompt based on video ID
    # Format filename as 000, 001, 002
    FILENAME="$(printf "%03d" $id)"
    python attention_statistics_debug.py \
        --prompt "$PROMPT" --prompt_idx $id \
        --output_dir ./original_debug \
        --model $model_name --video_mode $mode --num_inference_steps 50 \
        --pck --vis_track --vis_timesteps 49 --vis_layers 17 \
        --txt_path ./dataset/txt_prompts/foreground.txt \
        --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
        --track_path ./dataset/cotracker/$model_name/$mode/tracks \
        --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
        --video_dir ./dataset/videos \
        --device cuda:$gpu --qk_device cuda:$gpu
    echo "Generated: $FILENAME"
done