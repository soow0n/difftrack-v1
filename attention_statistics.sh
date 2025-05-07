model_name=cogvideox_t2v
mode=fg
gpu=0

python attention_statistics.py \
    --output_dir ./output \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --affinity_score --affinity_mode max --pck \
    --vis_attn_map --pos_y 16 --pos_x 16 \
    --vis_track --vis_timesteps 49 --vis_layers 17 \
    --txt_path ./dataset/txt_prompts/foreground.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --track_path ./dataset/cotracker/$model_name/$mode/tracks \
    --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
    --video_dir ./dataset/videos \
    --device cuda:$gpu --qk_device cuda:$gpu