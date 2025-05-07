start=0
end=200

python evaluate_tapvid.py \
    --output_dir ./output/cogvideox_t2v_no_overlap/kinetics_${start}_${end}/frame_1 \
    --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --chunk_frame_num 1 \
    --save_layer 17 --save_timestep 49 --inverse_step 49 \
    --pipe_device cuda:4 --qk_device cuda:4 \
    --start $start --end $end