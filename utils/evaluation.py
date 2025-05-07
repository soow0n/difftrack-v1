import numpy as np
import os
import pickle

from typing import Iterable, Mapping, Tuple, Union

import torch

def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    evaluation_points = query_frame_to_eval_frames[query_frame] > 0

    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy"] = occ_acc

    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_jaccard = []
    for thresh in [1, 2, 4, 8, 16]:
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)

        true_positives = np.sum(
            is_correct & pred_visible & evaluation_points, axis=(1, 2)
        )

        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    return metrics




class Evaluator():
    def __init__(self, zero_shot=False):
        self.reset()
        self.zero_shot = zero_shot
        
    def reset(self):
        self.aj = []
        self.delta_avg = []
        self.oa = []
        self.delta_1 = []
        self.delta_2 = []
        self.delta_4 = []
        self.delta_8 = []
        self.delta_16 = []
        self.cnt = 0

    def update(self, out_metrics, video_len, verbose=True, log_file="log.txt"):
        aj = out_metrics['average_jaccard'][0] * 100
        delta = out_metrics['average_pts_within_thresh'][0] * 100
        delta_1 = out_metrics['pts_within_1'][0] * 100
        delta_2 = out_metrics['pts_within_2'][0] * 100
        delta_4 = out_metrics['pts_within_4'][0] * 100
        delta_8 = out_metrics['pts_within_8'][0] * 100
        delta_16 = out_metrics['pts_within_16'][0] * 100
        oa = out_metrics['occlusion_accuracy'][0] * 100

        message = ""
        if verbose:
            if self.zero_shot:
                message = f"Video {self.cnt} ({video_len} frames)| delta_avg: {delta:.2f} | delta_1: {delta_1:.2f} | delta_2: {delta_2:.2f} | delta_4: {delta_4:.2f} | delta_8: {delta_8:.2f} | delta_16: {delta_16:.2f}"
            else:
                message = f"Video {self.cnt} | AJ: {aj:.2f}, delta_avg: {delta:.2f}, OA: {oa:.2f}"
            # Print to console.
            print(message)
            # Append the message to the log file.
            with open(log_file, "a") as f:
                f.write(message + "\n")

        self.cnt += 1
        self.aj.append(aj)
        self.delta_avg.append(delta)
        self.oa.append(oa)
        self.delta_1.append(delta_1)
        self.delta_2.append(delta_2)
        self.delta_4.append(delta_4)
        self.delta_8.append(delta_8)
        self.delta_16.append(delta_16)


    def report(self, log_file="log.txt"):
        lines = []
        lines.append(f"Mean delta_avg: {sum(self.delta_avg) / len(self.delta_avg):.1f}")
        lines.append(f"Mean delta_1: {sum(self.delta_1) / len(self.delta_1):.1f}")
        lines.append(f"Mean delta_2: {sum(self.delta_2) / len(self.delta_2):.1f}")
        lines.append(f"Mean delta_4: {sum(self.delta_4) / len(self.delta_4):.1f}")
        lines.append(f"Mean delta_8: {sum(self.delta_8) / len(self.delta_8):.1f}")
        lines.append(f"Mean delta_16: {sum(self.delta_16) / len(self.delta_16):.1f}")
        if not self.zero_shot:
            lines.append(f"Mean AJ: {sum(self.aj) / len(self.aj):.1f}")
            lines.append(f"Mean delta_avg: {sum(self.delta_avg) / len(self.delta_avg):.1f}")

        # Print and log all lines.
        with open(log_file, "a") as f:
            for line in lines:
                print(line)
                f.write(line + "\n")
        print(f"Report saved to {log_file}")



class PCKEvaluator():
    def __init__(self, timestep_num, layer_num, gt_tracks, gt_visibility):
        self.pck = torch.zeros([layer_num, timestep_num])
        self.gt_tracks = gt_tracks
        self.gt_visibility = gt_visibility
        
    def update(self, pred_tracks, layer, timestep_idx):
        square_dist = torch.square(pred_tracks - self.gt_tracks) 
        masked_dist = self.gt_visibility.unsqueeze(-1) * square_dist
        euclidean_dist = torch.sqrt(torch.sum(masked_dist, dim=-1))

        # pck_thres = []
        # for thres in [1, 2, 4, 8, 16]:
        #     within_dist = (euclidean_dist < thres) * self.gt_visibility
        #     pck = (within_dist.sum() / self.gt_visibility.sum()) * 100
        #     pck_thres.append(pck)
        
        # pck_mean = sum(pck_thres) / 5

        thres = 8
        within_dist = (euclidean_dist < thres) * self.gt_visibility
        pck = (within_dist.sum() / self.gt_visibility.sum()) * 100

        self.pck[layer][timestep_idx] = pck

    
    def report(self, log_file='pck.txt'):
        with open(log_file, "w") as f:
            for i, row in enumerate(self.pck):
                row_str = ", ".join(f"{val:.4f}" for val in row)
                f.write(f"Layer {i}: {row_str}\n")