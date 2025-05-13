start=600
end=900

CUDA_VISIBLE_DEVICES=2 python croco_eval.py \
    --output_dir ./output/tapvid/kinetics/croco/${start}_${end} \
    --resize_h 224 --resize_w 224 \
    --eval_dataset kinetics_first --tapvid_root /root/data/tapvid_kinetics \
    --model crocov2 \
    --croco_ckpt ./zeroco/pretrained_models/CroCo_V2_ViTLarge_BaseDecoder.pth \
    --output_correlation ca_map \
    --output_ca_map \
    --reciprocity \
    --heuristic_attn_map_refine \
    --softargmax_beta 1e-4 \
    --start $start --end $end
