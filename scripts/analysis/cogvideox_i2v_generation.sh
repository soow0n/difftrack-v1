model=cogvideox_i2v
scene=fg # bg
python analyze_generation.py \
    --output_dir ./output \
    --model $model --video_mode $scene --num_inference_steps 50 \
    --pck --affinity_score \
    --vis_timesteps 49 --vis_layers 17 \
    --vis_attn_map --pos_h 16 24 --pos_w 16 36 --vis_track \
    --txt_path ./dataset/txt_prompts/$scene.txt \
    --idx_path ./dataset/cotracker/$model/${scene}_50.txt \
    --track_path ./dataset/cotracker/$model/$scene/tracks \
    --visibility_path ./dataset/cotracker/$model/$scene/visibility \
    --video_dir ./dataset/videos/$scene \
    --device cuda:0