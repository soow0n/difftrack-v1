# f_num=1
# matching_type=qk
# python evaluate_tapvid_svd.py \
#         --output_dir ./output/svd_eval/block3_${matching_type}_frame_$f_num \
#         --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
#         --resize_h 480 --resize_w 720 \
#         --chunk_interval True --chunk_frame_num $f_num \
#         --save_layer 17 --save_timestep 49 --inverse_step 49 \
#         --pipe_device cuda:7 --qk_device cuda:7 \
#         --matching_type $matching_type

matching_type=qk
python evaluate_tapvid_svd.py \
        --output_dir ./output/eval_svd_kinetics \
        --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
        --resize_h 480 --resize_w 720 \
        --chunk_interval True --chunk_frame_num 12 \
        --save_layer 17 --save_timestep 49 --inverse_step 49 \
        --pipe_device cuda:7 --qk_device cuda:7 \
        --matching_type $matching_type \
        --start 0 --end 16 --batchfy True --batch_size 4