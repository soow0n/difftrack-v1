


import torch
from diffusers import AutoencoderKLCogVideoX, CogVideoXTrackPipeline, CogVideoXImageToVideoTrackPipeline2B, CogVideoXInversePipeline, HunyuanVideoTransformer3DModel, HunyuanVideoTrackPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
import os
import random
import numpy as np
import argparse
from utils.matching import corr_to_matches
import torch.nn.functional as F
from einops import rearrange
from utils.track_vis import Visualizer
from utils.evaluation import Evaluator, compute_tapvid_metrics
from dataset.tapvid import TAPVid
from torchvision.transforms import ToPILImage
import math
import glob

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

from utils.query_key_vis import src_pos_img

from PIL import Image

PARAMS = {
    'trajectory': False,
    'attn_weight': False,
    'query_key': False,
    'head_matching_layer': -1
}

INFERENCE_STEP = {
    'cogvideox_t2v_2b': 50,
    'cogvideox_i2v': 50,
    'cogvideox_t2v_5b': 50,
    'cogvideox_i2v': 50,
    'hunyuan_t2v': 30
}

TOTAL_LAYER_NUM = {
    'cogvideox_t2v_2b': 30,
    'cogvideox_t2v_5b': 42,
    'cogvideox_i2v': 30,
    'hunyuan_t2v': 60
}


def process_chunk(
    chunk_video,
    generator,
    pipe,
    first_frame,
    matching_timestep,
    matching_layer,
    inverse_step,
    add_noise=False,
    inversion_pipe=None,
    model_name="cogvideox_t2v",
    frame_as_latent=True
):

    chunk_video = torch.cat([first_frame, chunk_video], dim=1) / 255.
    _, _, _, H, W = first_frame.shape 

    # Run the inversion and generation pipelines.
    with torch.no_grad():
        latents = None
        if inversion_pipe is not None:
            latents = inversion_pipe(
                height=H,
                width=W,
                prompt="",
                video=chunk_video,
                num_videos_per_prompt=1,
                num_inference_steps=INFERENCE_STEP[model_name],
                inverse_step=inverse_step,
                use_dynamic_cfg=True,
                guidance_scale=6,
                generator=generator,
                params=PARAMS
            )
        
        if "i2v" in model_name:
            to_pil = ToPILImage()
            frames, queries, keys, text_seq_length = pipe(
                image=to_pil(first_frame[0,0]),
                height=H,
                width=W,
                prompt="",
                guidance_scale=6,
                num_inference_steps=INFERENCE_STEP[model_name],
                generator=generator,
                latents=latents,
                video=chunk_video,
                frame_as_latent=True,
                inverse_step=inverse_step,
                matching_timestep=matching_timestep,
                matching_layer=matching_layer,
                add_noise=add_noise,
                return_dict=False,
                params=PARAMS
            )
        else:

            frames, frames_latent, queries, keys, text_seq_length = pipe(
                height=H,
                width=W,
                prompt="",
                guidance_scale=6,
                num_inference_steps=INFERENCE_STEP[model_name],
                generator=generator,
                latents=latents,
                video=chunk_video,
                frame_as_latent=frame_as_latent,
                inverse_step=inverse_step,
                matching_timestep=matching_timestep,
                matching_layer=matching_layer,
                add_noise=add_noise,
                return_dict=False,
                params=PARAMS
            )

    latent_scaling_size = 16
    h = H // latent_scaling_size
    w = W // latent_scaling_size

    f_num = queries[0][:, text_seq_length:].shape[1] // (h*w)  # 17550 // (30*45) => 13
    
    queries = torch.cat(queries, dim=-1)
    keys = torch.cat(keys, dim=-1)

    if "cogvideox" in model_name:
        query_frames = rearrange(queries[:, text_seq_length:][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
        key_frames = rearrange(keys[:, text_seq_length:][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
    elif "hunyuan" in model_name:
        query_frames = rearrange(queries[:, :-text_seq_length][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)
        key_frames = rearrange(keys[:, :-text_seq_length][None,...], 'b head (f h w) c -> b head f (h w) c', f=f_num, h=h, w=w)

    return query_frames, key_frames, frames, frames_latent


def combine_correlations(correlations, average_overlap=False):
    corr_dict = {}
    for frames_indices, corr in correlations:
        for i, frame_idx in enumerate(frames_indices):
            if frame_idx not in corr_dict:
                corr_dict[frame_idx] = [corr[i]]
            elif average_overlap:
                corr_dict[frame_idx].append(corr[i])
    combined_corr = [torch.mean(torch.stack(corr_dict[i], dim=0), dim=0) for i in sorted(corr_dict.keys())]
    return combined_corr



def interpolate_spatial(tensor, H, W):
    B, hw, h, w = tensor.shape

    x = tensor.permute(0, 2, 3, 1).view(-1, 1, h, w)
    x_up = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

    x_up = x_up.reshape(B, h, w, H, W).permute(0, 3, 4, 1, 2)

    return x_up


def load_pipe(model, device):
    if model == "cogvideox_t2v_2b":
        # Load the pipelines.
        pipe = CogVideoXTrackPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)


        # pipe.vae = AutoencoderKLCogVideoX.from_pretrained("THUDM/CogVideoX-2b", subfolder="vae", temporal_compression_ratio=1, torch_dtype=torch.bfloat16)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        # pipe = CogVideoXTrackPipeline(
        #     tokenizer=pipe.tokenizer,
        #     text_encoder=pipe.text_encoder,
        #     vae=vae,
        #     transformer=pipe.transformer,
        #     scheduler=scheduler
        # )
        
        pipe.to(device=device, dtype=torch.bfloat16)
    elif model == "cogvideox_t2v_5b":
        # Load the pipelines.
        pipe = CogVideoXTrackPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        pipe.to(device=device, dtype=torch.bfloat16)
    elif model == "cogvideox_i2v":
        pipe = CogVideoXImageToVideoTrackPipeline2B.from_pretrained(
            "NimVideo/cogvideox-2b-img2vid",
            torch_dtype=torch.bfloat16
        ).to(device=device)
        pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif model == "hunyuan_t2v":
        do_inversion = False
        model_id = "hunyuanvideo-community/HunyuanVideo"
        transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        pipe = HunyuanVideoTrackPipeline.from_pretrained(
            model_id, transformer=transformer, 
            torch_dtype=torch.bfloat16
        ).to(device=device)

    return pipe


def main(args):
    # Fix seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.manual_seed(seed)

    to_pil = ToPILImage()

    chunk_stride = args.chunk_stride if not args.chunk_frame_interval else 1
    inverse_step = args.inverse_step if args.inverse_step in range(INFERENCE_STEP[args.model]) else min(args.matching_timestep)
    
    assert inverse_step < INFERENCE_STEP[args.model], f"Inverse step should be smaller than inference step {INFERENCE_STEP[args.model]} for {args.model}"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

    # Set up the dataset and evaluator.
    dataset = TAPVid(args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset
    
    pipe = load_pipe(args.model, args.pipe_device)

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    assert not (args.do_inversion and args.add_noise), "Cannot inverse noise and add random noise at same time"
    inversion_pipe = None
    if args.do_inversion and "cogvideox" in args.model:
        inversion_pipe = CogVideoXInversePipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.bfloat16)
        inversion_pipe.scheduler = CogVideoXDDIMScheduler.from_config(inversion_pipe.scheduler.config, timestep_spacing="trailing")
        inversion_pipe.to(device=args.inv_pipe_device, dtype=torch.bfloat16)
        inversion_pipe.vae.enable_slicing()
        inversion_pipe.vae.enable_tiling()
    
    matching_timestep = args.matching_timestep
    matching_layer = args.matching_layer

    matching_head = list(range(pipe.transformer.config.num_attention_heads))
    matching_head = [x for x in matching_head if x not in args.bad_matching_head]
    

    assert min(matching_timestep) >= inverse_step, "Matching timestep should be bigger than inverse step"
    assert max(matching_layer) < TOTAL_LAYER_NUM[args.model], f"Matching layer index should be smaller than total number of layers({TOTAL_LAYER_NUM[args.model]} for {args.model})"
    assert max(matching_head) < pipe.transformer.config.num_attention_heads

    PARAMS['matching_layer'] = matching_layer

    for j, (video, gt_trajectory, visibility, query_points_i, video_ori) in enumerate(dataloader):
        if j not in [15, 6, 8, 17, 18, 21, 27]: continue
        valid_mask = (query_points_i[:,:,0] == 0)
        if not torch.any(valid_mask):
            continue
        _, first_idx = torch.nonzero(valid_mask, as_tuple=True)
        query_points_i = query_points_i[:, first_idx, :]
        gt_trajectory = gt_trajectory[:, :, first_idx, :]
        visibility = visibility[:, :, first_idx]
        
        if args.video_max_len != -1 and args.video_max_len < video.size(1):
            video = video[:,:args.video_max_len, ...]
            video_ori = video_ori[:,:args.video_max_len, ...]
            gt_trajectory = gt_trajectory[:, :args.video_max_len, :, :]
            visibility = visibility[:, :args.video_max_len, :]

        _, T, _, H, W = video.shape

        save_dir = os.path.join(output_dir, f'{j:03d}')
        os.makedirs(save_dir, exist_ok=True)

        # from PIL import Image
        # image_files = glob.glob(os.path.join(save_dir, '*.png'))
        # image_list = [Image.open(os.path.join(save_dir, f'{idx}.png')) for idx in range(len(image_files))]
        # export_to_video(image_list, os.path.join(save_dir, 'all.mp4'), fps=8)

        # if args.vis_video:
        #     _, T, _, H_ori, W_ori = video_ori.shape
        #     # gt_trajectory_vis = gt_trajectory.clone()
        #     # gt_trajectory_vis[..., 0] *= (W_ori / 256)
        #     # gt_trajectory_vis[..., 1] *= (H_ori / 256)
        #     # vis = Visualizer(save_dir=os.path.join(output_dir, 'gt'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
    
        #     # frames = vis.visualize(video=video_ori, tracks=gt_trajectory_vis, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
        #     vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15, fps=8)
        #     vis.save_video(video_ori.to(torch.uint8).byte(), filename=f"gt.mp4", writer=None, step=0)
        gt_dir = os.path.join(output_dir, 'gt', f'{j:03d}')
        os.makedirs(gt_dir, exist_ok=True)
        
        for f in range(T):
            from PIL import Image
            Image.fromarray(video_ori[0, f].to(torch.uint8).permute(1, 2, 0).numpy()).save(os.path.join(gt_dir, f'{f}.png'))

        
        latent_scaling_size = 16
        h = H // latent_scaling_size
        w = W // latent_scaling_size

        queried_coords_latent = query_points_i.clone().float()
        queried_coords_latent[:,:,1] = queried_coords_latent[:,:,1] / latent_scaling_size
        queried_coords_latent[:,:,2] = queried_coords_latent[:,:,2] / latent_scaling_size

        interval = T // 12 if args.chunk_frame_interval else 1
        chunk_start, chunk_end = 1, (args.chunk_len - 2) * interval + 1

        try:
            visited_idx = set([])
            correlation_per_chunk = []
            video_idx = 0
            while len(visited_idx) < T - 1:
                chunk_indices = list(range(chunk_start, chunk_end + 1, interval))

                # Extract the current chunk.
                chunk_video = video_ori[:, torch.tensor(chunk_indices),...]
                visited_idx.update(chunk_indices)

                frame_as_latent = True if args.chunk_len == 13 else False
                # Process the chunk.
                query_frames, key_frames, frames, frame_latent = process_chunk(
                    chunk_video=chunk_video,
                    generator=generator,
                    matching_timestep=args.matching_timestep, 
                    matching_layer=args.matching_layer,
                    inverse_step=inverse_step,
                    pipe=pipe,
                    first_frame=video_ori[:,0:1,...],
                    add_noise=args.add_noise,
                    inversion_pipe=inversion_pipe,
                    model_name=args.model,
                    frame_as_latent=frame_as_latent
                )

                B, head_dim, qk_len, _, C = query_frames.shape
                frames[0][0].save(os.path.join(save_dir, f'0.png'))

                for query_idx in range(queried_coords_latent.size(1)):
                    w_pos, h_pos = queried_coords_latent[0,query_idx,1:].int()

                    height_circle = int((h_pos+0.5) / h * H)
                    width_circle = int((w_pos+0.5) / w * W)
                    input_image = cv2.circle(np.array(frames[0][0]).copy(), (width_circle, height_circle), 15, (255, 255, 255), -1)        
                    input_image = cv2.circle(input_image, (width_circle, height_circle), 10, (0, 0, 255), -1)    
                            
                    Image.fromarray((input_image).astype(np.uint8)).save(os.path.join(save_dir, f'attn_src{query_idx}.png'))
                
                for chk, chunk_idx in enumerate(chunk_indices):
                    if frame_as_latent:
                        background = frames[0][4*(chk+1)]
                        frames[0][4*(chk+1)].save(os.path.join(save_dir, f'{chunk_idx}.png'))
                        frame_latent[chk+1].save(os.path.join(save_dir, f'latent_{chunk_idx}.png'))
                    else:
                        background = frames[0][chk+1]
                        frames[0][chk+1].save(os.path.join(save_dir, f'{chunk_idx}.png'))
                        frame_latent[chk // 4 + 1].save(os.path.join(save_dir, f'latent_{chunk_idx}.png'))
                    
                    k = chk + 1 if frame_as_latent else chk // 4 + 1
                    attn_tts = torch.einsum("b h i d, b h j d -> b h i j", query_frames[:, :, 0, :, :].to(device=args.qk_device), key_frames[:, :, k, :, :].to(device=args.qk_device)) / math.sqrt(head_dim)
                    
                    attn_tts = attn_tts.softmax(dim=-1)
                    attn_tts = attn_tts.mean(1)
                    # background = frames[4*k]
                    # background = background.permute(1, 2, 0).cpu().numpy()
                    background = np.array(background)
                    # breakpoint()
                    img_h, img_w, _ = background.shape
                    import torchvision
                    resize = torchvision.transforms.Resize((img_h, img_w))
                    
                    w_pos, h_pos = queried_coords_latent[0, 0, 1:].int()
                    attn_score = attn_tts.reshape(h, w, h, w)[h_pos, w_pos].cpu()
                    attn_score = resize(attn_score.unsqueeze(0).unsqueeze(0)).squeeze(0).permute(1,2,0)
                    
                    normalizer = mpl.colors.Normalize(vmin=attn_score.min(), vmax=attn_score.max())
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')
                    colormapped_im = (mapper.to_rgba(attn_score.to(dtype=torch.float32)[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
                    attn_map = cv2.addWeighted(background, 0.3, colormapped_im, 0.7, 0)
                    Image.fromarray(attn_map).save(os.path.join(save_dir, f'attn_{chunk_idx}.png'))

                chunk_start += chunk_stride
                chunk_end = min(chunk_start + (args.chunk_len - 2) * interval, T-1)

        except:
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='cogvideox_t2v')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)

    parser.add_argument('--chunk_stride', type=int, default=12)
    parser.add_argument('--chunk_len', type=int, default=13)
    parser.add_argument('--chunk_frame_interval', action='store_true')
    parser.add_argument('--average_overlapped_corr', action='store_true')
    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--do_inversion", action='store_true')
    parser.add_argument("--add_noise", action='store_true')

    parser.add_argument('--matching_layer', nargs='+', type=int)
    parser.add_argument('--bad_matching_head', nargs='+', type=int, default=[])
    parser.add_argument("--matching_timestep", nargs='+', type=int)
    parser.add_argument("--inverse_step", type=int, default=None)
    
    parser.add_argument("--vis_video", action='store_true')
    parser.add_argument("--tracks_leave_trace", type=int, default=15)

    parser.add_argument("--pipe_device", type=str, default='cuda:0')
    parser.add_argument("--inv_pipe_device", type=str, default='cuda:0')
    parser.add_argument("--qk_device", type=str, default='cuda:0')


    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    args = parser.parse_args()

    main(args)
    