model=cogvideox_t2v
python evaluate_tapvid.py \
    --model $model \
    --save_layer 17 --save_timestep 49 --inverse_step 49 \
    --output_dir ./output/tapvid/davis/${model} \
    --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:4 --qk_device cuda:4 \
    --start 0 --end 200 --vis_video True