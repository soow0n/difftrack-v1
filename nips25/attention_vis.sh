gpu=7
model_name=cogvideox_t2v # cogvideox_i2v, hunyuan_t2v

mode=fg
python attention_statistics.py \
    --output_dir /mnt/ssd5/soowon/diff-track/attn_cherrypick \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --vis_attn_map --pos_h 6 12 18 24 --pos_w 9 18 27 36 \
    --vis_timesteps 49 --vis_layers 17 \
    --txt_path ./dataset/txt_prompts/foreground.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --track_path /home/cvlab16/projects/jeeyoung/co-tracker/sam2/sam2/saved_predictions_0509_fg/fg_whole_grid/grid/tracks \
    --visibility_path /home/cvlab16/projects/jeeyoung/co-tracker/sam2/sam2/saved_predictions_0509_fg/fg_whole_grid/grid/visibility \
    --video_dir ./dataset/videos \
    --device cuda:$gpu --qk_device cuda:$gpu

# mode=bg
# python attention_statistics.py \
#     --output_dir ./output \
#     --model $model_name --video_mode $mode --num_inference_steps 50 \
#     --affinity_score --affinity_mode max --pck \
#     --vis_attn_map --pos_y 16 --pos_x 16 \
#     --vis_track --vis_timesteps 49 --vis_layers 17 \
#     --txt_path ./dataset/txt_prompts/background.txt \
#     --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
#     --track_path /home/cvlab16/projects/jeeyoung/co-tracker/sam2/sam2/saved_predictions_0509_bg/bg/grid/scene/tracks \
#     --visibility_path /home/cvlab16/projects/jeeyoung/co-tracker/sam2/sam2/saved_predictions_0509_bg/bg/grid/scene/visibility \
#     --video_dir ./dataset/videos_scene \
#     --device cuda:$gpu --qk_device cuda:$gpu