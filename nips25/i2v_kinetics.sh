model=cogvideox_i2v
python evaluate_tapvid.py \
    --model $model \
    --save_layer 18 --save_timestep 39 --inverse_step 39 \
    --output_dir ./output/tapvid/kinetics/${model} \
    --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:5 --qk_device cuda:5 \
    --start 0 --end 200