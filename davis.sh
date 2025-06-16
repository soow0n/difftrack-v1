
model=cogvideox_t2v_5b
for chunk_len in 5 8 11; do
    python evaluate_tapvid_vis.py \
        --model $model \
        --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
        --output_dir /mnt/ssd5/soowon/diff-track/last_last/$chunk_len \
        --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
        --resize_h 480 --resize_w 720 \
        --chunk_frame_interval --average_overlapped_corr --chunk_len $chunk_len \
        --pipe_device cuda:7 \
        --vis_video --tracks_leave_trace 15
done