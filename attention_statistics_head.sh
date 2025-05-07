model_name=cogvideox_t2v
mode=fg
gpu=4

python attention_statistics_head.py \
    --output_dir ./head_pck \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --pck --vis_track --vis_timesteps 49 --vis_layers 17 \
    --txt_path ./dataset/txt_prompts/foreground.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --track_path ./dataset/cotracker/$model_name/$mode/tracks \
    --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
    --video_dir ./dataset/videos \
    --device cuda:$gpu --qk_device cuda:$gpu