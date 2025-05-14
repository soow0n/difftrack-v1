start=0
end=300
python evaluate_tapvid_crw.py \
    --model-type scratch --resume ./videowalk/pretrained.pth \
    --output_dir ./output/tapvid/kinetics/crw/${start}_${end} \
    --eval_dataset kinetics_first --tapvid_root /root/data/tapvid_kinetics \
    --resize_h 256 --resize_w 384 \
    --device 'cuda:3' \
    --start $start --end $end --batch_size 1

