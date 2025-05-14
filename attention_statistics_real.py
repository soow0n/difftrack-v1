import argparse
import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline2B, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers import CogVideoXTrackPipeline, CogVideoXImageToVideoTrackPipeline2B, CogVideoXInversePipeline, HunyuanVideoTransformer3DModel, HunyuanVideoTrackPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler
import os
import glob
import random
import numpy as np
from torchvision.transforms import ToPILImage
from utils.affinity_score import AffinityScore
from utils.evaluation import PCKEvaluator
from utils.query_key_vis import QueryKeyVisualizer, src_pos_img
from utils.track_vis import Visualizer
from utils.aggregate_results import pck_mean, affinity_mean
from dataset.tapvid import TAPVid

import cv2
from PIL import Image


def load_pipe(model, device):
    if model == "cogvideox_t2v":
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v":
        pipe = CogVideoXImageToVideoPipeline2B.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "hunyuan_t2v":
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoPipeline.from_pretrained(
            model_id, transformer=transformer, 
            torch_dtype=torch.float16
        ).to(device)
    else:
        raise ValueError("Select model in ['cogvideox_t2v', 'cogvideox_i2v', 'hunyuan_t2v']")


    return pipe

def main(args):

    device = args.device

    output_dir = args.output_dir

    params = {
        'trajectory': args.pck,
        'matching_mode': 'qk',
        'attn_weight': args.affinity_score,
        'query_key': False,
        'feature': False,
        'video_mode': 'fg',
        'qk_device': device,
        'debug': False,
        'save_layer': [],
        'save_timestep': [],
        'query_coords': None,
    }
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipe(model=args.model, device=device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Set up the dataset and evaluator.
    dataset = TAPVid(args, batchfy=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)


    pipe = load_pipe(model=args.model, device=device)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    for j, (input_video, gt_trajectory, gt_visibility, query_points_i, video_ori) in enumerate(dataloader):

        
        valid_mask = (query_points_i[:,:,0] == 0)
        if not torch.any(valid_mask):
            continue
        _, first_idx = torch.nonzero(valid_mask, as_tuple=True)
        query_points_i = query_points_i[:, first_idx, :]
        gt_trajectory = gt_trajectory[:, :, first_idx, :]
        gt_visibility = gt_visibility[:, :, first_idx]
        
        if args.video_max_len != -1 and args.video_max_len < input_video.size(1):
            input_video = input_video[:,:args.video_max_len, ...]
            gt_trajectory = gt_trajectory[:, :args.video_max_len, :, :]
            gt_visibility = gt_visibility[:, :args.video_max_len, :]
        
        params['query_coords'] = query_points_i[...,1:]

        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = torch.manual_seed(seed)
        
        save_dir = os.path.join(output_dir, f'{j:03d}')
        os.makedirs(save_dir, exist_ok=True)

        if args.affinity_score:
            affinity_score = AffinityScore(
                num_inference_steps=args.num_inference_steps,
                num_layers=pipe.transformer.config.num_layers, # need debug
                visibility=gt_visibility,
                model=args.model
            )
        else:
            affinity_score = None

        if args.pck:
            layer_num = 60 if args.model == 'hunyuan_t2v' else 30
            qk_pck_evaluator = PCKEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_trajectory,
                gt_visibility=gt_visibility
            )
            feat_pck_evaluator = PCKEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=layer_num,
                gt_tracks=gt_trajectory,
                gt_visibility=gt_visibility
            )
        else:
            qk_pck_evaluator = None
            feat_pck_evaluator = None

        if args.vis_attn_map:
            querykey_visualizer = QueryKeyVisualizer(
                save_timestep_idxs=args.vis_timesteps,
                save_layers=args.vis_layers,
                output_dir=save_dir,
                model=args.model
            )
        else:
            querykey_visualizer = None


        for inverse_timestep in range(50):
            with torch.no_grad():
                video, attn_query_keys, features, vis_trajectory = pipe(
                    prompt="",
                    height=480,
                    width=720,
                    num_frames=49,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    affinity_score=affinity_score,
                    qk_pck_evaluator=qk_pck_evaluator,
                    feat_pck_evaluator=feat_pck_evaluator,
                    querykey_visualizer=querykey_visualizer,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params,
                    video=input_video,
                    inverse_step=inverse_timestep
                )

        if affinity_score is not None:
            affinity_score.report(save_dir)
            print(f"Affinity score saved at {save_dir}")
        
        if qk_pck_evaluator is not None:
            qk_pck_evaluator.report(os.path.join(save_dir, 'qk_pck.txt'))
            print(f"PCK saved at {os.path.join(save_dir, 'qk_pck.txt')}")
        if feat_pck_evaluator is not None:
            feat_pck_evaluator.report(os.path.join(save_dir, 'feat_pck.txt'))
            print(f"PCK saved at {os.path.join(save_dir, 'feat_pck.txt')}")

    if args.pck:
        pck_mean(file_list=glob.glob(os.path.join(output_dir, '*/qk_pck.txt')), output_path=os.path.join(output_dir, 'total_qk_pck.csv'))
        pck_mean(file_list=glob.glob(os.path.join(output_dir, '*/feat_pck.txt')), output_path=os.path.join(output_dir, 'total_feat_pck.csv'))


    if args.affinity_score:
        affinity_mean(file_list=glob.glob(os.path.join(output_dir, f'*/affinity_max.xlsx')), output_path=os.path.join(output_dir, f'total_affinity_max.xlsx'))
        affinity_mean(file_list=glob.glob(os.path.join(output_dir, f'*/affinity_sum.xlsx')), output_path=os.path.join(output_dir, f'total_affinity_sum.xlsx'))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cogvideox-t2v')
    parser.add_argument("--num_inference_steps", type=int, default=50)

    parser.add_argument("--affinity_score", action='store_true')
    parser.add_argument("--pck", action='store_true')

    parser.add_argument("--vis_attn_map", action='store_true')
    parser.add_argument("--pos_y", type=int, default=16)
    parser.add_argument("--pos_x", type=int, default=16)

    parser.add_argument("--vis_track", action='store_true')
    parser.add_argument("--vis_timesteps", nargs='+', type=int, default=[49])
    parser.add_argument("--vis_layers", nargs='+', type=int, default=[17])

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "rgb_stacking_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)
    parser.add_argument("--video_max_len", type=int, default=49)

    
    args = parser.parse_args()


    main(args)
