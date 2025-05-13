


import torch
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

from videowalk.code.model import CRW
import videowalk.code.utils as utils

def main(args):
    to_pil = ToPILImage()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gt'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'pred'), exist_ok=True)

    batchfy = args.batch_size > 1
    # Set up the dataset and evaluator.
    dataset = TAPVid(args, batchfy=batchfy)
    batch_size = args.batch_size if args.eval_dataset == 'kinetics_first' else 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset

    model = CRW(args).to(args.device)
    checkpoint = torch.load(args.resume, map_location=args.device)
    if args.model_type == 'scratch':
        state = {}
        for k,v in checkpoint['model'].items():
            if 'conv1.1.weight' in k or 'conv2.1.weight' in k:
                state[k.replace('.1.weight', '.weight')] = v
            else:
                state[k] = v
        utils.partial_load(state, model, skip_keys=['head'])
    else:
        utils.partial_load(checkpoint['model'], model, skip_keys=['head'])

    del checkpoint
    
    model.eval()
    model = model.to(args.device)

    with torch.no_grad():
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
            if args.vis_video:
                gt_trajectory_vis = gt_trajectory.clone()
                gt_trajectory_vis[..., 0] *= (W / 256)
                gt_trajectory_vis[..., 1] *= (H / 256)
                vis = Visualizer(save_dir=os.path.join(output_dir, 'gt'), pad_value=60, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
                frames = vis.visualize(video=video, tracks=gt_trajectory_vis, filename =f"{j:03d}.mp4", query_frame=1, visibility=visibility)
                
                gt_dir = os.path.join(output_dir, 'gt', f'{j:03d}')
                os.makedirs(gt_dir, exist_ok=True)
                for f, frame in enumerate(frames[0]):
                    to_pil(frame).save(os.path.join(gt_dir, f'{f}.png'))

            bsize = args.chunk_len
            # correlation = []
            # query_feat = model.encoder(video[:, :1].transpose(1,2).to(args.device))
            # h, w = query_feat.shape[-2], query_feat.shape[-1]
            
            feats = []
            for b in range(0, video.shape[1], bsize):
                feat = model.encoder(video[:, b:b+bsize].transpose(1,2).to(args.device))
                feats.append(feat.cpu())
            feats = torch.cat(feats, dim=2).squeeze(1)
            feats = torch.nn.functional.normalize(feats, dim=1)
            
            torch.cuda.empty_cache()

            correlation = []
            B, C, T, h, w = feats.shape
            query_feat = feats[:, :, 0].view(-1, C, h * w).to(args.device)

            
            for k in range(1, T):
                key_feat = feats[:, :, k].view(-1, C, h * w).to(args.device)
                attn_tts = query_feat.transpose(-2, -1) @ key_feat
                attn_tts = attn_tts.softmax(dim=-1)
                
                correlation_from_t_to_s = rearrange(attn_tts, 'b (h w) c -> b c h w', h=h, w=w)
                correlation.append(correlation_from_t_to_s)

                
            num_queries = query_points_i.size(1)
            stride_val = H // h
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
                vis = Visualizer(save_dir=os.path.join(output_dir, 'pred'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=args.tracks_leave_trace)
                frames = vis.visualize(video=video_ori, tracks=trajectory, filename =f"{j:03d}.mp4", query_frame=0, visibility=visibility)
                
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
            # torch.cuda.empty_cache()
            

    evaluator.report(log_file=os.path.join(output_dir, 'log.txt'))
    print(f"Evaluation result saved at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "rgb_stacking_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=480)
    parser.add_argument('--resize_w', type=int, default=720)

    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--vis_video", type=bool, default=False)
    parser.add_argument("--tracks_leave_trace", type=int, default=15)

    parser.add_argument("--device", type=str, default='cuda:0')

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--chunk_len', type=int, default=8)

    parser.add_argument('--model-type', default='scratch', type=str)
    parser.add_argument('--resume', default='', type=str)

    parser.add_argument('--head-depth', default=-1, type=int,
                        help='depth of mlp applied after encoder (0 = linear)')

    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    parser.add_argument('--no-l2', default=False, action='store_true', help='')

    parser.add_argument('--long-mem', default=[0], type=int, nargs='*', help='')
    parser.add_argument('--texture', default=False, action='store_true', help='')
    parser.add_argument('--round', default=False, action='store_true', help='')

    parser.add_argument('--norm_mask', default=False, action='store_true', help='')
    parser.add_argument('--finetune', default=0, type=int, help='')
    parser.add_argument('--pca-vis', default=False, action='store_true')

    args = parser.parse_args()

    main(args)
    