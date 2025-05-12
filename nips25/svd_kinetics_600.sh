start=300
end=600
python evaluate_tapvid_svd.py \
    --output_dir ./output/tapvid/kinetics/svd/${start}_${end} \
    --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True  --average_chunk_overlap True \
    --pipe_device cuda:5 --qk_device cuda:5 \
    --start $start --end $end --batch_size 1