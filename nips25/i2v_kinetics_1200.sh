start=900
end=0
model=cogvideox_i2v
python evaluate_tapvid.py \
    --model $model \
    --save_layer 17 --save_timestep 39 --inverse_step 39 \
    --output_dir ./output/tapvid/kinetics/${model}/${start}_${end} \
    --eval_dataset kinetics_first --tapvid_root /root/data/tapvid_kinetics \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --pipe_device cuda:5 --qk_device cuda:5 \
    --start $start --end $end