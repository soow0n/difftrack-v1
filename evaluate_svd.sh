f_num=1
python evaluate_tapvid_svd.py \
        --output_dir ./output/svd_eval/frame_$f_num \
        --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
        --resize_h 480 --resize_w 720 \
        --chunk_interval True --chunk_frame_num $f_num \
        --save_layer 17 --save_timestep 49 --inverse_step 49 \
        --pipe_device cuda:7 --qk_device cuda:7