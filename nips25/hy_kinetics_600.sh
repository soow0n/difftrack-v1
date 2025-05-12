start=300
end=600
model=hunyuan_t2v
python evaluate_tapvid.py \
    --model $model \
    --output_dir ./output/tapvid/kinetics/${model}/${start}_${end} \
    --save_layer 16 --save_timestep 29 --inverse_step 29 \
    --eval_dataset kinetics_first --tapvid_root /mnt/ssd1/PointTracking/tapvid/kinetics-dataset \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:5 --qk_device cuda:5 \
    --start $start --end $end --batch_size 1
