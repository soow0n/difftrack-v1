start=1050
end=0
model=cogvideox_t2v_2b
python evaluate_tapvid.py \
    --start $start --end $end \
    --model $model \
    --matching_layer 17 --matching_timestep 48 --inverse_step 48 \
    --output_dir ./output/tapvid/kinetics/$model/${start}_${end} \
    --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
    --resize_h 480 --resize_w 720 \
    --chunk_frame_interval --average_overlapped_corr --add_noise \
    --pipe_device cuda:6