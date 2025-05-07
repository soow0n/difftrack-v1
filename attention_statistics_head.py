import argparse
import torch
from diffusers import CogVideoXImageToVideoPipeline2B, CogVideoXPipeline, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
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




def main(args):

    device = args.device

    output_dir = args.output_dir
    prompt_path = args.txt_path 
    index_path = args.idx_path

    params = {
        'trajectory': False,
        'matching_mode': 'qk',
        'attn_weight': args.affinity_score,
        'query_key': True,
        'video_mode': args.video_mode,
        'qk_device': args.qk_device,
        'debug': True,
        'head': True
    }
    os.makedirs(output_dir, exist_ok=True)

    # if args.model == "cogvideox_t2v":
    #     pipe = CogVideoXPipeline.from_pretrained(
    #         "THUDM/CogVideoX-2b",
    #         torch_dtype=torch.bfloat16
    #     ).to(device)
    # elif args.model == "cogvideox_i2v":
    #     pipe = CogVideoXImageToVideoPipeline2B.from_pretrained(
    #         "NimVideo/cogvideox-2b-img2vid",
    #         torch_dtype=torch.bfloat16
    #     ).to(device)
    # elif args.model == "hunyuan_t2v":
    #     model_id = "hunyuanvideo-community/HunyuanVideo"
    #     transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    #         model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    #     )
    #     pipe = HunyuanVideoPipeline.from_pretrained(
    #         model_id, transformer=transformer, 
    #         torch_dtype=torch.float16
    #     ).to(device)
    # else:
    #     raise ValueError("Select model in ['cogvideox_t2v', 'cogvideox_i2v', 'hunyuan_t2v']")
    
    # pipe.vae.enable_slicing()
    # pipe.vae.enable_tiling()

    with open(prompt_path, "r") as file:
        prompts = file.readlines()
    
    with open(index_path, "r") as file:
        indices = file.readlines()
        selected_indices = sorted([int(index.strip()) for index in indices])

    for i, prompt in enumerate(prompts):
        if i not in selected_indices:
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

        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.bfloat16
        ).to(device)

        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        save_dir = os.path.join(output_dir, f'{i:03d}')
        os.makedirs(save_dir, exist_ok=True)

        prompt = prompt.strip()
        gt_track = torch.from_numpy(np.load(os.path.join(args.track_path, f'{i:03d}.npy'))).to(args.qk_device)
        gt_visibility = torch.from_numpy(np.load(os.path.join(args.visibility_path, f'{i:03d}.npy'))).to(args.qk_device)

        if args.affinity_score:
            affinity_score = AffinityScore(
                num_inference_steps=args.num_inference_steps,
                num_layers=pipe.transformer.config.num_layers, # need debug
                mode=args.affinity_mode,
                visibility=gt_visibility,
                model=args.model
            )
        else:
            affinity_score = None

        if args.pck:
            pck_evaluator = PCKEvaluator(
                timestep_num=args.num_inference_steps,
                layer_num=pipe.transformer.config.num_layers,
                gt_tracks=gt_track,
                gt_visibility=gt_visibility
            )
        else:
            pck_evaluator = None

        if args.vis_attn_map:
            querykey_visualizer = QueryKeyVisualizer(
                save_timestep_idxs=args.vis_timesteps,
                save_layers=args.vis_layers,
                output_dir=save_dir,
                model=args.model
            )
        else:
            querykey_visualizer = None

        with torch.no_grad(): 
            if "i2v" in args.model:
                video_path = os.path.join(args.video_dir, f'{i:03d}.mp4')
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

                video, attn_query_keys, vis_trajectory = pipe(
                    image=image,
                    prompt=prompt,
                    height=480,
                    width=720,
                    num_frames=49,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    affinity_score=affinity_score,
                    pck_evaluator=pck_evaluator,
                    querykey_visualizer=querykey_visualizer,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params
                )

                image.save(os.path.join(save_dir,f'first_frame.png'))
            else:
                video, attn_query_keys, vis_trajectory = pipe(
                    prompt=prompt,
                    height=480,
                    width=720,
                    num_frames=49,
                    num_inference_steps=args.num_inference_steps,
                    return_dict=False,
                    generator=generator,
                    affinity_score=affinity_score,
                    pck_evaluator=pck_evaluator,
                    querykey_visualizer=querykey_visualizer,
                    vis_timesteps=args.vis_timesteps,
                    vis_layers=args.vis_layers,
                    output_type="pt",
                    params=params
                )

            if affinity_score is not None:
                affinity_score.report(os.path.join(save_dir, f'affinity_{args.affinity_mode}.xlsx'))
                print(f"Affinity score ({args.affinity_mode}) saved at {os.path.join(save_dir, f'affinity_{args.affinity_mode}.xlsx')}")
            
            if pck_evaluator is not None:
                pck_evaluator.report(os.path.join(save_dir, 'pck.txt'))
                print(f"PCK saved at {os.path.join(save_dir, 'pck.txt')}")
            

            if querykey_visualizer is not None:
                attention_dir = os.path.join(save_dir, 'attention_map')
                os.makedirs(attention_dir, exist_ok=True)

                pos_img = src_pos_img(video[0][0], args.pos_y, args.pos_x)
                pos_img.save(os.path.join(attention_dir, 'src_pos.png'))

                querykey_visualizer.save_i2i_attn_map(
                    attn_query_keys=attn_query_keys,
                    output_dir=attention_dir,
                    pos=(args.pos_y, args.pos_x)
                )

                os.makedirs(os.path.join(save_dir, 'pca'), exist_ok=True)
                querykey_visualizer.save_pcas(
                    attn_query_keys=attn_query_keys, 
                    output_dir=os.path.join(save_dir, 'pca')
                )
            

            if args.vis_track:
                track_dir = os.path.join(save_dir, 'track')
                os.makedirs(track_dir, exist_ok=True)

                valid_mask = gt_visibility[0,0,:]
                first_idx = torch.nonzero(valid_mask, as_tuple=True)[0]
                vis = Visualizer(save_dir=track_dir, pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15, fps=8)
                frames = vis.visualize(
                    video=video * 255, 
                    tracks=vis_trajectory[:, :, first_idx, :], 
                    visibility=gt_visibility[:, :, first_idx], 
                    filename="video.mp4", 
                    query_frame=0
                )
                
                # to_pil = ToPILImage()
                # for f in range(49):
                #     to_pil(frames[0,f]).save(os.path.join(track_dir, f'{f}.png'))


    if args.pck:
        pck_mean(file_list=glob.glob(os.path.join(output_dir, '*/pck.txt')), output_path=os.path.join(output_dir, 'total_pck.csv'))

    if args.affinity_score:
        affinity_mean(file_list=glob.glob(os.path.join(output_dir, f'*/affinity_{args.affinity_mode}.xlsx')), output_path=os.path.join(output_dir, f'total_affinity_{args.affinity_mode}.xlsx'))



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cogvideox-t2v')
    parser.add_argument("--video_mode", type=str, default='fg')
    parser.add_argument("--num_inference_steps", type=int, default=50)

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
    parser.add_argument("--idx_path", type=str, required=True)
    parser.add_argument("--track_path", type=str, required=True)
    parser.add_argument("--visibility_path", type=str, required=True)
    parser.add_argument("--video_dir", type=str, default='')
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--qk_device", type=str, default='cuda:0')
    
    args = parser.parse_args()


    main(args)
