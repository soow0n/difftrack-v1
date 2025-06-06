
CUDA_VISIBLE_DEVICES=7 python attention_statistics_cag.py \
    --start 100 --end 250 \
    --output_dir /mnt/ssd5/soowon/diff-track/cag_userstudy/cag_1 \
    --txt_path dataset/txt_prompts/fg.txt \
    --pag_scale 1 --cfg_scale 6 \
    --model_version 5b \
    --pag_layers 15 17 18 \
    --pag_mode cag
