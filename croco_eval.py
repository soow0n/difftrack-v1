


import torch
import os
import argparse
import torch.nn.functional as F
from einops import rearrange
from utils.evaluation import Evaluator, compute_tapvid_metrics
from utils.track_vis import Visualizer
from dataset.tapvid import TAPVid
import numpy as np

from zeroco.models.croco.croco import CroCoNet
from zeroco.models.croco.croco_downstream import croco_args_from_ckpt
from zeroco.models.croco.mod import FeatureL2Norm, unnormalise_and_convert_mapping_to_flow

def softmax_with_temperature(x, beta, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x/beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def soft_argmax(corr, beta=0.02, x_normal=None, y_normal=None):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    x_normal = torch.tensor(np.linspace(-1,1,14)).cuda()
    y_normal = torch.tensor(np.linspace(-1,1,14)).cuda()
    
    b,_,h,w = corr.size()
    corr = softmax_with_temperature(corr, beta=beta, d=1)
    corr = corr.view(-1,h,w,h,w) # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False) # marginalize to x-coord.
    x_normal = x_normal.expand(b,w)
    x_normal = x_normal.view(b,w,1,1)
    grid_x = (grid_x*x_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    
    grid_y = corr.sum(dim=2, keepdim=False) # marginalize to y-coord.
    y_normal = y_normal.expand(b,h)
    y_normal = y_normal.view(b,h,1,1)
    grid_y = (grid_y*y_normal).sum(dim=1, keepdim=True) # b x 1 x h x w
    return grid_x, grid_y

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

    H_224, W_224 = args.model_img_size

    os.makedirs(args.output_dir, exist_ok=True)

    # Set up the dataset and evaluator.
    dataset = TAPVid(args, batchfy=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)
    evaluator = Evaluator(zero_shot=True)
    queried_first = "first" in args.eval_dataset

    ckpt = torch.load(args.croco_ckpt,'cpu')
    croco_args = croco_args_from_ckpt(ckpt)
    croco_args['img_size'] = ((args.model_img_size[0]//32)*32,(args.model_img_size[1]//32)*32)
    croco_args['args'] = args
    network = CroCoNet(**croco_args)
    msg=network.load_state_dict(ckpt['model'], strict=False)
    print('missing keys: ', msg.missing_keys)
    print('unexpected keys: ', msg.unexpected_keys)
    print('CROCOV2 WEIGHT WELL LOADED: ', msg)
    network.eval()
    network = network.to('cuda')

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
        
        in1k_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        in1k_std =  torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

        source_img = video[:,0] / 255.0
        source_img = (source_img - in1k_mean) / in1k_std
        source_img = source_img.to('cuda')
        # source_img = F.interpolate(source_img, size=(H_224, W_224), mode='bilinear', align_corners=False) 

        with torch.no_grad():
            correlation = []
            for t in range(1, T):
                target_img = video[:, t] / 255.0
                target_img = (target_img - in1k_mean) / in1k_std
                target_img = target_img.to('cuda')
                # target_img = F.interpolate(target_img, size=(H_224, W_224), mode='bilinear', align_corners=False)
                corr = network(target_img, source_img, output_correlation='ca_map') # b 2 14 14

                # corr = corr.view(1, 14 * 14, 14, 14)  # Shape: (1, 196, 14, 14)

                # # Step 2: Interpolate over target spatial dims (h2, w2)
                # corr = F.interpolate(corr, size=(30, 45), mode='bilinear', align_corners=True)  # Shape: (1, 196, 30, 45)

                # # Step 3: Restore source dimensions
                # corr = corr.view(1, 14, 14, 30, 45)

                # # Step 4: Permute so target dims are now in batch for interpolation over source
                # corr = corr.permute(0, 3, 4, 1, 2)  # (1, 30, 45, 14, 14)
                # corr = corr.reshape(1, 30 * 45, 14, 14)  # Shape: (1, 1350, 14, 14)

                # # Step 5: Interpolate over source spatial dims (h1, w1)
                # corr = F.interpolate(corr, size=(30, 45), mode='bilinear', align_corners=True)  # (1, 1350, 30, 45)

                # # Step 6: Reshape and permute back to original format
                # corr = corr.view(1, 30, 45, 30, 45)  # Final shape
                correlation.append(corr)

        num_queries = query_points_i.size(1)
        stride_val = 16
        h = H // stride_val
        w = W // stride_val
        margin = W / (64*stride_val)
        
        queried_coords_latent = query_points_i.clone().float()
        # Normalize the coordinate channels (assumes channel 1 and 2 are spatial coordinates)
        # queried_coords_latent[:,:,1] = queried_coords_latent[:,:,1] / stride_val
        # queried_coords_latent[:,:,2] = queried_coords_latent[:,:,2] / stride_val
        queried_coords_latent[:,:,1] = queried_coords_latent[:,:,1]
        queried_coords_latent[:,:,2] = queried_coords_latent[:,:,2]


        # The initial track is the (normalized) query coordinates.
        tracks = []
        tracks.append(queried_coords_latent[:,:,1:].unsqueeze(1).to('cuda'))

        # Normalize coordinates for grid sampling.
        norm_coords = query_points_i[:,:,1:].clone().float()
        norm_coords[:, :, 0] = (norm_coords[:, :, 0] / (W - margin)) * 2 - 1.0
        norm_coords[:, :, 1] = (norm_coords[:, :, 1] / (H - margin) ) * 2 - 1.0
        grid = norm_coords.view(B, num_queries, 1, 2).to('cuda')
        
        # breakpoint()
        # For each of 12 layers (or timesteps) compute the displacement.
        for k in range(len(correlation)):
            correlation_from_t_to_s = correlation[k].to('cuda')
            # (x_source, y_source, x_target, y_target, score) = corr_to_matches(
            #     correlation_from_t_to_s.view(1, h, w, h, w).unsqueeze(1), get_maximum=True, do_softmax=True, device='cuda'
            # )
            # mapping_set = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1)
            # mapping_set = mapping_set.view(1, h, w, 2).permute(0, 3, 1, 2)
            grid_x, grid_y = soft_argmax(correlation_from_t_to_s.view(B, -1, 14, 14), beta=2e-2)
            coarse_flow = torch.cat((grid_x, grid_y), dim=1)    
            flow_est = unnormalise_and_convert_mapping_to_flow(coarse_flow) 
            

            flow_est = F.interpolate(flow_est, size=(224, 224), mode='bilinear', align_corners=False) 
            flow_est[:,0,:,:] *= 16
            flow_est[:,1,:,:] *= 16
            
            xx = torch.arange(0, 224).view(1, -1).repeat(224, 1)
            yy = torch.arange(0, 224).view(-1, 1).repeat(1, 224)
            xx = xx.view(1, 1, 224, 224).repeat(1, 1, 1, 1)
            yy = yy.view(1, 1, 224, 224).repeat(1, 1, 1, 1)
            grid_flow = torch.cat((xx, yy), 1).float().cuda()
            
                
            mapping_set = grid_flow + flow_est
            
            track = F.grid_sample(mapping_set.to(correlation_from_t_to_s.device).float(), grid=grid, align_corners=False)
            track = rearrange(track, "b c h w -> b () (h w) c")
            tracks.append(track)

        trajectory = torch.cat(tracks, dim=1)

        # Scale back the coordinates.
        # scaling_factor_x = stride_val
        # scaling_factor_y = stride_val

        # trajectory[..., 0] *= scaling_factor_x
        # trajectory[..., 1] *= scaling_factor_y

        if args.vis_video:
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            vis_traj = trajectory.clone()
            vis_traj[..., 0] *= (720 / 224)
            vis_traj[..., 1] *= (480 / 224)
            
            vis = Visualizer(save_dir=os.path.join(args.output_dir, 'pred'), pad_value=0, linewidth=3, show_first_frame=1, tracks_leave_trace=15)
            frames = vis.visualize(video=video_ori, tracks=vis_traj, filename =f"{j:03d}.mp4", query_frame=2, visibility=visibility)
            pred_dir = os.path.join(args.output_dir, 'pred', f'{j:03d}')
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
        evaluator.update(out_metrics, T, log_file=os.path.join(args.output_dir, 'log.txt'))
        

    evaluator.report(log_file=os.path.join(args.output_dir, 'log.txt'))
    print(f"Evaluation result saved at {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument('--tapvid_root', type=str)
    parser.add_argument('--eval_dataset', type=str, choices=["davis_first", "rgb_stacking_first", "kinetics_first"], default="davis_first")

    parser.add_argument('--resize_h', type=int, default=224)
    parser.add_argument('--resize_w', type=int, default=224)

    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=0)

    parser.add_argument('--model', type=str, required=True, help='Model to use')
    parser.add_argument("--video_max_len", type=int, default=-1)
    parser.add_argument("--croco_ckpt", type=str, default='./pretrained_weights/CroCo_V2_ViTLarge_BaseDecoder.pth')
    parser.add_argument('--output_flow_interp', action='store_true', help='output flow interpolation? default is False')
    parser.add_argument('--output_ca_map', action='store_true', help='output confidence map? default is False')
    parser.add_argument('--softmax_camap', action='store_true', help='softmax confidence map? default is False')
    parser.add_argument('--correlation', action='store_true', help='compute correlation? default is False')
    parser.add_argument('--reciprocity', action='store_true', help='compute reciprocity? default is False')
    parser.add_argument('--uncertainty', action='store_true', help='compute uncertainty? default is False')
    parser.add_argument('--model_img_size', nargs='+', type=int, default=[224, 224], help='model image size')
    parser.add_argument('--heuristic_attn_map_refine', action='store_true', help='heuristic attention map refine? default is False')
    parser.add_argument('--output_correlation', type=str, help='correlation type')
    parser.add_argument('--softargmax_beta', type=float, help='softargmax beta')

    parser.add_argument('--flipping_condition', action='store_true', help='flipping condition? default is False')
    parser.add_argument('--vis_video', action='store_true', help='output confidence map? default is False')


    args = parser.parse_args()

    main(args)
    