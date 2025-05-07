model_name=cogvideox_i2v
gpu=4

# mode=fg
# python attention_statistics_i2v.py \
#     --output_dir ./i2v_pck/fg \
#     --model $model_name --video_mode $mode --num_inference_steps 50 \
#     --pck \
#     --txt_path ./dataset/txt_prompts/foreground.txt \
#     --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
#     --track_path ./dataset/cotracker/$model_name/$mode/tracks \
#     --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
#     --video_dir ./dataset/videos \
#     --device cuda:$gpu --qk_device cuda:$gpu

mode=fg
matching_mode=feature
python attention_statistics_i2v.py \
    --output_dir ./i2v_pck/$matching_mode/$mode \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --pck --matching_mode $matching_mode \
    --txt_path ./dataset/txt_prompts/foreground.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --track_path ./dataset/cotracker/$model_name/$mode/tracks \
    --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
    --video_dir ./dataset/videos \
    --device cuda:$gpu --qk_device cuda:$gpu


# mode=bg
# matching_mode=qk
# python attention_statistics_i2v.py \
#     --output_dir ./i2v_pck/$matching_mode/$mode \
#     --model $model_name --video_mode $mode --num_inference_steps 50 \
#     --pck --matching_mode $matching_mode \
#     --txt_path ./dataset/txt_prompts/background.txt \
#     --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
#     --track_path ./dataset/cotracker/$model_name/$mode/tracks \
#     --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
#     --video_dir ./dataset/videos_scene \
#     --device cuda:$gpu --qk_device cuda:$gpu


# mode=bg
# matching_mode=feature
# python attention_statistics_i2v.py \
#     --output_dir ./i2v_pck/$matching_mode/$mode \
#     --model $model_name --video_mode $mode --num_inference_steps 50 \
#     --pck --matching_mode $matching_mode \
#     --txt_path ./dataset/txt_prompts/background.txt \
#     --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
#     --track_path ./dataset/cotracker/$model_name/$mode/tracks \
#     --visibility_path ./dataset/cotracker/$model_name/$mode/visibility \
#     --video_dir ./dataset/videos_scene \
#     --device cuda:$gpu --qk_device cuda:$gpu