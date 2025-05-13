
python evaluate_tapvid_crw.py \
    --model-type scratch --resume ./videowalk/pretrained.pth \
    --output_dir ./output/tapvid/davis/crw \
    --eval_dataset davis_first --tapvid_root /root/data/tapvid_davis/tapvid_davis.pkl \
    --resize_h 256 --resize_w 256 \
    --device 'cuda:5' \
    --start 0 --end 200 --batch_size 4 \
    # --vis_video True

