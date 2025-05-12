python evaluate_tapvid_svd.py \
    --output_dir ./output/tapvid/davis/svd \
    --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
    --resize_h 480 --resize_w 720 --average_chunk_overlap True \
    --chunk_interval True \
    --pipe_device cuda:7 --qk_device cuda:7 \
    --vis_video True