
model=hunyuan_t2v
python evaluate_tapvid.py \
    --model $model \
    --output_dir ./output/tapvid/davis/${model} \
    --save_layer 16 --save_timestep 29 --inverse_step 29 \
    --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:6 --qk_device cuda:6 \
    --start 0 --end 200 --batch_size 4