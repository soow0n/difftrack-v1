gpu=6
model_name=cogvideox_t2v # cogvideox_i2v, hunyuan_t2v

mode=fg
python attention_statistics_real.py \
    --resize_h 480 --resize_w 720 \
    --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
    --output_dir ./output/real_statistics \
    --model $model_name --num_inference_steps 50 \
    --pck --affinity_score \
    --device cuda:$gpu

