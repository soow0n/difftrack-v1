
# CUDA_VISIBLE_DEVICES=6 python attention_statistics_cag.py \
#     --output_dir /mnt/ssd5/soowon/diff-track/cag/5b_wo_cfg_cag1 \
#     --txt_path dataset/txt_prompts/cag_cherrypick.txt \
#     --pag_scale 1 --cfg_scale 0 \
#     --model_version 5b \
#     --pag_layers 11 15 17 18 \
#     --pag_mode cag


CUDA_VISIBLE_DEVICES=3 python attention_statistics_cag.py \
    --start 0 --end 50 \
    --output_dir /mnt/ssd5/soowon/diff-track/cag_userstudy/cag_2 \
    --txt_path dataset/txt_prompts/fg.txt \
    --pag_scale 1 --cfg_scale 6 \
    --model_version 5b \
    --pag_layers 11 15 17 18 \
    --pag_mode cag
