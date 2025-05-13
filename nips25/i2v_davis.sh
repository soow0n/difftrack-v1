model=cogvideox_i2v
python evaluate_tapvid.py \
    --model $model \
    --save_layer 11 --save_timestep 24 --inverse_step 24 \
    --output_dir ./output/tapvid/davis/${model} \
    --eval_dataset davis_first --tapvid_root /root/data/tapvid_davis/tapvid_davis.pkl \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:0 --qk_device cuda:0 \
    --start 0 --end 200