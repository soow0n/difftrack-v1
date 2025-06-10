
CUDA_VISIBLE_DEVICES=0 python motion_guidance.py \
    --output_dir ./output \
    --txt_path ./dataset/txt_prompts/cag_prompts.txt \
    --pag_scale 1 --cfg_scale 6 \
    --model_version 2b \
    --pag_layers 13 17 21