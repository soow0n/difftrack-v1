model_name=cogvideox_t2v
gpu=7

mode=fg
python reproduce.py \
    --output_dir ./reproduce_t2v/$mode \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --txt_path ./dataset/txt_prompts/foreground.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --device cuda:$gpu --qk_device cuda:$gpu

mode=bg
python reproduce.py \
    --output_dir ./reproduce_t2v/$mode \
    --model $model_name --video_mode $mode --num_inference_steps 50 \
    --txt_path ./dataset/txt_prompts/background.txt \
    --idx_path ./dataset/cotracker/$model_name/${mode}_50.txt \
    --device cuda:$gpu --qk_device cuda:$gpu