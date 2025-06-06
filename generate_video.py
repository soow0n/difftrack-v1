import argparse
import torch
from diffusers import (
    CogVideoXImageToVideoPipeline, 
    CogVideoXImageToVideoPipeline2B, 
    CogVideoXPipeline, 
    HunyuanVideoTransformer3DModel, 
    HunyuanVideoPipeline,
    AutoencoderKLWan,
    WanPipeline
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

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


import cv2
from PIL import Image


def load_pipe(model, device):
    if model == "wan":
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        flow_shift = 3.0  # 5.0 for 720P, 3.0 for 480P
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        pipe.to(device)
    elif model == "cogvideox_t2v_5b":
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5b",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v_2b":
        pipe = CogVideoXImageToVideoPipeline2B.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device)
    elif model == "cogvideox_i2v_5b":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            "THUDM/CogVideoX-5b-I2V",
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
    prompt_path = args.txt_path 
    # index_path = args.idx_path

    params = {
        'query_coords': None,
        'video_mode': args.video_mode,
        'trajectory': False,
        'attn_weight': False,

        'matching_layer': [],
        'query_key': False,
        'feature': False,
        'head_matching_layer': -1,
        'trajectory_head': False,
    }
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipe(model=args.model, device=device)

    if args.model != "wan":
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()


    with open(prompt_path, "r") as file:
        prompts = file.readlines()
    
    # with open(index_path, "r") as file:
    #     indices = file.readlines()
    #     selected_indices = sorted([int(index.strip()) for index in indices])

    for i, prompt in enumerate(prompts):
        # if i < 47: 
        #     continue
        # if i in selected_indices:
        #     continue

        if i < args.start:
            continue
        
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        generator = torch.manual_seed(seed)

        if os.path.isfile(os.path.join(output_dir, f"{i:03d}.mp4")):
            continue

        if i < args.start or i > args.end:
            continue
        
        # save_dir = os.path.join(output_dir, f'{i:03d}')
        # os.makedirs(save_dir, exist_ok=True)

        prompt = prompt.strip()
        # gt_track = None # torch.from_numpy(np.load(os.path.join(args.track_path, f'{i:03d}.npy'))).to(args.qk_device)
        # gt_visibility = None #torch.from_numpy(np.load(os.path.join(args.visibility_path, f'{i:03d}.npy'))).to(args.qk_device)


        with torch.no_grad(): 
            if "i2v" in args.model:
                video_path = os.path.join(args.video_dir, f'{i:03d}.mp4')

                if not os.path.isfile(video_path):
                    continue
                cap = cv2.VideoCapture(video_path)

                # Read the first frame
                success, first_frame = cap.read()
                if success:
                    # Convert the frame from BGR (OpenCV default) to RGB
                    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
                    # Convert the NumPy array (frame) to a PIL Image
                    image = Image.fromarray(frame_rgb)
                else:
                    print("Failed to extract the first frame.")
                    continue

                video, attn_query_keys, feature, vis_trajectory = pipe(
                    image=image,
                    prompt=prompt,
                    height=480,
                    width=832,
                    num_frames=49,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params
                )

                image.save(os.path.join(save_dir,f'first_frame.png'))
                vis = Visualizer(save_dir=args.output_dir, pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15, fps=8)
                vis.save_video((video * 255).to(torch.uint8).byte(), filename=f"{i:03d}.mp4", writer=None, step=0)
            else:
                num_frames = 49
                if args.model == "wan":
                   num_frames = 81

                video, attn_query_keys, feature, vis_trajectory = pipe(
                    prompt=prompt,
                    num_frames=num_frames,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params,
                    guidance_scale=args.cfg_scale
                )

                vis = Visualizer(save_dir=args.output_dir, pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15, fps=8)
                vis.save_video((video * 255).to(torch.uint8).byte(), filename=f"{i:03d}.mp4", writer=None, step=0)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cogvideox-t2v')
    parser.add_argument("--video_mode", type=str, default='fg')
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=6)


    parser.add_argument("--affinity_score", action='store_true')
    parser.add_argument("--affinity_mode", type=str, default='max')
    parser.add_argument("--pck", action='store_true')

    parser.add_argument("--vis_attn_map", action='store_true')
    parser.add_argument("--pos_y", type=int, default=16)
    parser.add_argument("--pos_x", type=int, default=16)


    parser.add_argument("--vis_track", action='store_true')
    parser.add_argument("--vis_timesteps", nargs='+', type=int, default=[49])
    parser.add_argument("--vis_layers", nargs='+', type=int, default=[17])

    parser.add_argument("--txt_path", type=str, required=True)
    # parser.add_argument("--idx_path", type=str, required=True)
    # parser.add_argument("--track_path", type=str, required=True)
    # parser.add_argument("--visibility_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100)


    
    args = parser.parse_args()


    main(args)
