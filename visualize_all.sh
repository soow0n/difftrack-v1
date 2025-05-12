
# model=hunyuan_t2v
# python evaluate_tapvid_all.py \
#     --model $model \
#     --output_dir ./output/eval_${model} \
#     --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
#     --resize_h 480 --resize_w 720 \
#     --chunk_interval True --average_chunk_overlap True \
#     --save_layer 33 --save_timestep 29 --inverse_step 29 \
#     --pipe_device cuda:4 --qk_device cuda:5 \
#     --start 0 --end 16 --batchfy True --batch_size 4

# model=cogvideox_i2v
# python evaluate_tapvid_all.py \
#     --model $model \
#     --output_dir ./output/eval_${model} \
#     --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
#     --resize_h 480 --resize_w 720 \
#     --chunk_interval True --average_chunk_overlap True \
#     --save_layer 17 --save_timestep 39 --inverse_step 39 \
#     --pipe_device cuda:5 --qk_device cuda:5 \
#     --start 0 --end 16 --batchfy True


model=cogvideox_t2v
python tracking_vis.py \
    --model $model \
    --output_dir ./visualize/${model} \
    --eval_dataset davis_first --tapvid_root /mnt/ssd4/PointTracking/tapvid_davis/tapvid_davis.pkl \
    --resize_h 480 --resize_w 720 \
    --chunk_interval True --average_chunk_overlap True \
    --save_layer 16 --save_timestep 49 --inverse_step 49 \
    --pipe_device cuda:6 --qk_device cuda:6 \
    --vis_video True