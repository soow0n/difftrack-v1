


import torch
from diffusers import CogVideoXTrackPipeline, CogVideoXImageToVideoTrackPipeline2B, CogVideoXInversePipeline, HunyuanVideoTransformer3DModel, HunyuanVideoTrackPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler
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

from videowalk.code.model import CRW
import videowalk.code.utils as utils
from videowalk.code.utils import test_utils

PARAMS = {
    'trajectory': False,
    'attn_weight': False,
    'query_key': True,
    'qk_device': '',
}


def process_chunk(
    chunk_video,
    model,
    first_frame,
    device
):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # Shape: (1, C, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)   # Shape: (1, C, 1, 1)
    # Concatenate the first frame and chunk_video, then normalize
    chunk_video = torch.cat([first_frame, chunk_video], dim=1) / 255.0
    chunk_video = (chunk_video - mean) / std

    _, _, _, H, W = first_frame.shape

    feats = model.encoder(chunk_video.transpose(1,2).to(device))

    feats = F.normalize(feats, dim=1)

    correlation = []
    B, C, T, h, w = feats.shape
    query_feat = feats[:, :, 0].view(-1, C, h * w).to(args.device)

    
    for k in range(1, T):
        key_feat = feats[:, :, k].view(-1, C, h * w).to(args.device)
        attn_tts = query_feat.transpose(-2, -1) @ key_feat
        attn_tts = attn_tts.softmax(dim=-1)
        
        correlation_from_t_to_s = rearrange(attn_tts, 'b (h w) c -> b c h w', h=h, w=w)
        correlation.append(correlation_from_t_to_s)

    return correlation



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
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()

    if args.chunk_interval:
        stride = 1
    else:
        stride = args.stride

    output_dir = args.output_dir
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

    batchfy = args.batch_size > 1
    # Set up the dataset and evaluator.
    dataset = TAPVid(args, batchfy=False)
    batch_size = args.batch_size if args.eval_dataset == 'kinetics_first' else 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset

    model = CRW(args).to(args.device)
    checkpoint = torch.load(args.resume, map_location=args.device)
    utils.partial_load(checkpoint['model'], model, skip_keys=['head'])
    model.eval()
    

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

        B, T, _, H, W = video.shape

        stride_val = 8
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
            with torch.no_grad():
                corr = process_chunk(
                    chunk_video=chunk_video,
                    model=model,
                    first_frame=video[:,0:1,...],
                    device=args.device
                )
            
            correlation_per_chunk.append((chunk_indices, corr))
            # del corr
            # torch.cuda.empty_cache()
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
        tracks.append(queried_coords_latent[:,:,1:].unsqueeze(1).to(args.device))

        # Normalize coordinates for grid sampling.
        norm_coords = queried_coords_latent[:,:,1:].clone()
        norm_coords[:, :, 0] = (norm_coords[:, :, 0] / (w - margin)) * 2 - 1.0
        norm_coords[:, :, 1] = (norm_coords[:, :, 1] / (h - margin)) * 2 - 1.0
        grid = norm_coords.view(B, num_queries, 1, 2).to(args.device)

        # For each of 12 layers (or timesteps) compute the displacement.
        for k in range(len(correlation)):
            correlation_from_t_to_s = correlation[k].to(args.device)
            (x_source, y_source, x_target, y_target, score) = corr_to_matches(
                correlation_from_t_to_s.view(1, h, w, h, w).unsqueeze(1), get_maximum=True, do_softmax=True, device=args.device
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
            vis_traj = trajectory.clone()
            vis_traj[..., 0] *= (720 / W)
            vis_traj[..., 1] *= (480 / H)
            vis = Visualizer(save_dir=os.path.join(output_dir, 'pred'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
            frames = vis.visualize(video=video_ori, tracks=vis_traj, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
            
            pred_dir = os.path.join(output_dir, 'pred', f'{j:03d}')
            os.makedirs(pred_dir, exist_ok=True)
            for f, frame in enumerate(frames[0]):
                to_pil(frame).save(os.path.join(pred_dir, f'{f}.png'))

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
    parser.add_argument("--model", type=str, default='cogvideox_t2v')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "rgb_stacking_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)

    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--chunk_frame_num', type=int, default=12)
    parser.add_argument('--chunk_interval', type=bool, default=True)
    parser.add_argument('--average_chunk_overlap', type=bool, default=False)
    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--do_inversion", type=bool, default=False)
    parser.add_argument("--add_noise", type=bool, default=False)

    parser.add_argument('--save_layer', nargs='+', type=int)
    parser.add_argument("--save_timestep", nargs='+', type=int)
    parser.add_argument("--inverse_step", type=int, default=None)
    
    parser.add_argument("--vis_video", type=bool, default=False)
    parser.add_argument("--tracks_leave_trace", type=int, default=15)

    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--inv_pipe_device", type=str, default='cuda:0')
    parser.add_argument("--qk_device", type=str, default='cuda:0')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--head-depth', default=-1, type=int,
                        help='depth of mlp applied after encoder (0 = linear)')

    parser.add_argument('--remove-layers', default=[], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')
    parser.add_argument('--pca-vis', default=False, action='store_true')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    main(args)
    