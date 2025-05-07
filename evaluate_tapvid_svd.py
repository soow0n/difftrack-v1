


import torch
from diffusers import StableVideoDiffusionPipeline
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
import time
from PIL import Image

PARAMS = {
    'trajectory': False,
    'attn_weight': False,
    'query_key': True,
    'qk_device': '',
}


def process_chunk(
    chunk_video,
    generator,
    pipe,
    first_frame,
):
    # Convert the chunk (first video in the batch) to a list of PIL images.
    to_pil = ToPILImage()
    chunk_video = torch.cat([first_frame] + [chunk_video[:, i:i+1, ...] for i in range(chunk_video.size(1))], dim=1)
    video_pil = [to_pil(frame.to(dtype=torch.uint8)) for frame in chunk_video[0]]
    _, _, _, H, W = chunk_video.shape 

    # Run the inversion and generation pipelines.
    with torch.no_grad():
        latents = None
        frames, query, key = pipe(
            image=video_pil[0],
            video=video_pil, 
            num_frames=len(video_pil),
            decode_chunk_size=8, 
            generator=generator,
            height=480,
            width=720,
            return_dict=False
        )

        fhw, head_dim, B, C = query.shape
        query = query.permute(2, 1, 0, 3)
        key = key.permute(2, 1, 0, 3)

        f_len = fhw // 1350

        query = query.reshape(B, head_dim, f_len, fhw // f_len, C)[1].unsqueeze(0)
        key = key.reshape(B, head_dim, f_len, fhw // f_len, C)[1].unsqueeze(0)
        
    return query, key

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


def main(args):

    if args.chunk_interval:
        stride = 1
    else:
        stride = args.stride


    inverse_step = args.inverse_step if args.inverse_step in range(50) else min(args.save_timestep)
    output_dir = os.path.join(args.output_dir, f'{args.eval_dataset}_layer{args.save_layer}_inverse{inverse_step}_timestep{args.save_timestep}_stride{stride}_interval{args.chunk_interval}_overlap{args.average_chunk_overlap}_inversion{args.do_inversion}_noise{args.add_noise}')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)


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

    # Set up the dataset and evaluator.
    dataset = TAPVid(args, visualize=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset

    # Load the pipelines.
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.to(device=args.pipe_device)

    PARAMS['qk_device'] = args.qk_device
    
    save_timestep = args.save_timestep
    save_layer = args.save_layer
    
    video_ori = None
    for j, (video, gt_trajectory, visibility, query_points_i, video_ori) in enumerate(dataloader): 

        valid_mask = (query_points_i[:,:,0] == 0)
        if not torch.any(valid_mask):
            continue
        _, first_idx = torch.nonzero(valid_mask, as_tuple=True)
        query_points_i = query_points_i[:, first_idx, :]
        gt_trajectory = gt_trajectory[:, :, first_idx, :]
        visibility = visibility[:, :, first_idx]
        
        if args.video_max_len != -1 and args.video_max_len < video.size(1):
            video = video[:,:args.video_max_len, ...]
            gt_trajectory = gt_trajectory[:, :args.video_max_len, :, :]
            visibility = visibility[:, :args.video_max_len, :]

        _, T, _, H, W = video.shape

        if args.vis_video:
            gt_trajectory_vis = gt_trajectory.clone()
            gt_trajectory_vis[..., 0] *= (W / 256)
            gt_trajectory_vis[..., 1] *= (H / 256)
            vis = Visualizer(save_dir=os.path.join(output_dir, 'gt'), pad_value=120, linewidth=3, show_first_frame=1)
            vis.visualize(video=video, tracks=gt_trajectory_vis, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
            

        stride_val = 16
        h = H // stride_val
        w = W // stride_val

        chunk_len = args.chunk_frame_num - 1
        interval = T // 12 if args.chunk_interval else 1
        chunk_start, chunk_end = 1, chunk_len * interval + 1

        query_frames, key_frames = None, None
        visited_idx = set([])
        correlation_per_chunk = []
        while len(visited_idx) < T - 1:
            chunk_indices = list(range(chunk_start, chunk_end + 1, interval))

            # Extract the current chunk.
            chunk_video = video[:, torch.tensor(chunk_indices),...]
            visited_idx.update(chunk_indices)
            # Process the chunk.
            query_frames, key_frames = process_chunk(
                chunk_video=chunk_video,
                generator=generator,
                pipe=pipe,
                first_frame=video[:,0:1,...],
            )
            B, head_dim, qk_len, _, _ = query_frames.shape            

            corr = []
            for k in range(1, qk_len):
                attn_tts = torch.einsum("b h i d, b h j d -> b h i j", query_frames[:, :, 0, :, :], key_frames[:, :, k, :, :]) / math.sqrt(head_dim)
                attn_stt = torch.einsum("b h i d, b h j d -> b h i j", query_frames[:, :, k, :, :], key_frames[:, :, 0, :, :]) / math.sqrt(head_dim)
                attn_tts = attn_tts.softmax(dim=-1)
                attn_stt = attn_stt.softmax(dim=-1)
                attn_tts = attn_tts.mean(1)
                attn_stt = attn_stt.mean(1)
                correlation_from_t_to_s = rearrange(attn_tts, 'b (h w) c -> b c h w', h=h, w=w) # head mean
                correlation_from_t_to_s_T = rearrange(attn_stt, 'b c (h w) -> b c h w', h=h, w=w)
                correlation_from_t_to_s = (correlation_from_t_to_s + correlation_from_t_to_s_T) / 2 # torch.Size([1, 1350, 30, 45])
                corr.append(correlation_from_t_to_s)
            correlation_per_chunk.append((chunk_indices, corr))
            chunk_start += stride
            chunk_end = min(chunk_start + chunk_len * interval, T-1)
            
        correlation = combine_correlations(correlation_per_chunk, average_overlap=args.average_chunk_overlap)

        num_queries = query_points_i.size(1)

        margin = W/(64*stride_val)
        queried_coords_latent = query_points_i.clone().float()
        # Normalize the coordinate channels (assumes channel 1 and 2 are spatial coordinates)
        queried_coords_latent[:,:,1] = queried_coords_latent[:,:,1] / stride_val
        queried_coords_latent[:,:,2] = queried_coords_latent[:,:,2] / stride_val

        # The initial track is the (normalized) query coordinates.
        tracks = []
        tracks.append(queried_coords_latent[:,:,1:].unsqueeze(1).to(PARAMS['qk_device']))

        # Normalize coordinates for grid sampling.
        norm_coords = queried_coords_latent[:,:,1:].clone()
        norm_coords[:, :, 0] = (norm_coords[:, :, 0] / (w - margin)) * 2 - 1.0
        norm_coords[:, :, 1] = (norm_coords[:, :, 1] / (h - margin)) * 2 - 1.0
        grid = norm_coords.view(B, num_queries, 1, 2).to(PARAMS['qk_device'])

        # For each of 12 layers (or timesteps) compute the displacement.
        for k in range(len(correlation)):
            correlation_from_t_to_s = correlation[k].to(PARAMS['qk_device']) / (len(save_timestep) * len(save_layer))
            (x_source, y_source, x_target, y_target, score) = corr_to_matches(
                correlation_from_t_to_s.view(1, h, w, h, w).unsqueeze(1), get_maximum=True, do_softmax=True, device=PARAMS['qk_device']
            )
            mapping_set = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1)
            mapping_set = mapping_set.view(1, h, w, 2).permute(0, 3, 1, 2)
            track = F.grid_sample(mapping_set.to(correlation_from_t_to_s.device), grid=grid, align_corners=True)
            track = rearrange(track, "b c h w -> b () (h w) c")
            tracks.append(track)

        trajectory = torch.cat(tracks, dim=1)


        # Scale back the coordinates.
        scaling_factor_x = stride_val 
        scaling_factor_y = stride_val

        trajectory[..., 0] *= scaling_factor_x
        trajectory[..., 1] *= scaling_factor_y

        if args.vis_video:
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            
            vis = Visualizer(save_dir=os.path.join(output_dir, 'pred'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
            frames = vis.visualize(video=video_ori, tracks=trajectory, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
            
            # pred_dir = os.path.join(output_dir, 'pred', f'{j:03d}')
            # os.makedirs(pred_dir, exist_ok=True)
            # for f, frame in enumerate(frames[0]):
            #     to_pil(frame).save(os.path.join(pred_dir, f'{f}.png'))
        # breakpoint()
        query_points_np = query_points_i.clone().cpu().numpy()
        gt_tracks = gt_trajectory.clone().permute(0, 2, 1, 3)[:, :, :trajectory.shape[1], :].cpu().numpy()
        gt_occluded = torch.logical_not(visibility.clone().permute(0, 2, 1)).cpu().numpy()
        pred_tracks = trajectory.permute(0, 2, 1, 3).cpu().numpy()
        pred_tracks[...,0] *= (256 / W)
        pred_tracks[...,1] *= (256 / H)
        pred_occluded = torch.zeros_like(visibility).permute(0, 2, 1).cpu().numpy()
        
        out_metrics = compute_tapvid_metrics(query_points_np, gt_occluded, gt_tracks, pred_occluded, pred_tracks,"first" if queried_first else "strided")
        evaluator.update(out_metrics, T, log_file=os.path.join(output_dir, 'log.txt'))
        


    evaluator.report(log_file=os.path.join(output_dir, 'log.txt'))
    print(f"Evaluation result saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "rgb_stacking_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)

    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--chunk_interval', type=bool, default=True)
    parser.add_argument('--chunk_frame_num', type=int, default=12)
    parser.add_argument('--average_chunk_overlap', type=bool, default=False)
    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--do_inversion", type=bool, default=False)
    parser.add_argument("--add_noise", type=bool, default=False)

    parser.add_argument('--save_layer', nargs='+', type=int)
    parser.add_argument("--save_timestep", nargs='+', type=int)
    parser.add_argument("--inverse_step", type=int, default=None)
    
    parser.add_argument("--vis_video", type=bool, default=False)
    parser.add_argument("--tracks_leave_trace", type=int, default=0)

    parser.add_argument("--pipe_device", type=str, default='cuda:0')
    parser.add_argument("--inv_pipe_device", type=str, default='cuda:0')
    parser.add_argument("--qk_device", type=str, default='cuda:0')

    args = parser.parse_args()

    main(args)
    