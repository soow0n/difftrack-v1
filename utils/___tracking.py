import torch
import torch.nn.functional as F
from typing import Optional, Tuple

def smart_cat(tensor1, tensor2, dim):
    if tensor1 is None:
        return tensor2
    return torch.cat([tensor1, tensor2], dim=dim)

def get_points_on_a_grid(
    size: int,
    extent: Tuple[float, ...],
    center: Optional[Tuple[float, ...]] = None,
    device: Optional[torch.device] = torch.device("cpu"),
):
    r"""Get a grid of points covering a rectangular region

    `get_points_on_a_grid(size, extent)` generates a :attr:`size` by
    :attr:`size` grid fo points distributed to cover a rectangular area
    specified by `extent`.

    The `extent` is a pair of integer :math:`(H,W)` specifying the height
    and width of the rectangle.

    Optionally, the :attr:`center` can be specified as a pair :math:`(c_y,c_x)`
    specifying the vertical and horizontal center coordinates. The center
    defaults to the middle of the extent.

    Points are distributed uniformly within the rectangle leaving a margin
    :math:`m=W/64` from the border.

    It returns a :math:`(1, \text{size} \times \text{size}, 2)` tensor of
    points :math:`P_{ij}=(x_i, y_i)` where

    .. math::
        P_{ij} = \left(
             c_x + m -\frac{W}{2} + \frac{W - 2m}{\text{size} - 1}\, j,~
             c_y + m -\frac{H}{2} + \frac{H - 2m}{\text{size} - 1}\, i
        \right)

    Points are returned in row-major order.

    Args:
        size (int): grid size.
        extent (tuple): height and with of the grid extent.
        center (tuple, optional): grid center.
        device (str, optional): Defaults to `"cpu"`.

    Returns:
        Tensor: grid.
    """
    if size == 1:
        return torch.tensor([extent[1] / 2, extent[0] / 2], device=device)[None, None]

    if center is None:
        center = [extent[0] / 2, extent[1] / 2]

    margin = extent[1] / 64
    range_y = (margin - extent[0] / 2 + center[0], extent[0] / 2 + center[0] - margin)
    range_x = (margin - extent[1] / 2 + center[1], extent[1] / 2 + center[1] - margin)
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(*range_y, size, device=device),
        torch.linspace(*range_x, size, device=device),
        indexing="ij",
    )
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)

def get_track_feat(fmaps, queried_frames, queried_coords):
        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        sample_track_feats = sample_features5d(fmaps, sample_coords)
        return sample_track_feats


def sample_features5d(input, coords):
    r"""Sample spatio-temporal features

    `sample_features5d(input, coords)` works in the same way as
    :func:`sample_features4d` but for spatio-temporal features and points:
    :attr:`input` is a 5D tensor :math:`(B, T, C, H, W)`, :attr:`coords` is
    a :math:`(B, R1, R2, 3)` tensor of spatio-temporal point :math:`(t_i,
    x_i, y_i)`. The output tensor has shape :math:`(B, R1, R2, C)`.

    Args:
        input (Tensor): spatio-temporal features.
        coords (Tensor): spatio-temporal points.

    Returns:
        Tensor: sampled features.
    """

    B, T, _, _, _ = input.shape

    # B T C H W -> B C T H W
    input = input.permute(0, 2, 1, 3, 4)

    # B R1 R2 3 -> B R1 R2 1 3
    coords = coords.unsqueeze(3)

    # B C R1 R2 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 3, 1, 4).view(
        B, feats.shape[2], feats.shape[3], feats.shape[1]
    )  # B C R1 R2 1 -> B R1 R2 C


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)], device=coords.device
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def _compute_dense_tracks(
     video, grid_query_frame, grid_size=80, backward_tracking=False
    ):
        *_, H, W = video.shape
        grid_step = W // grid_size
        grid_width = W // grid_step
        grid_height = H // grid_step
        tracks = visibilities = None
        grid_pts = torch.zeros((video.shape[0], grid_width * grid_height, 3)).to(device)
        grid_pts[:, :, 0] = grid_query_frame
        for offset in range(grid_step * grid_step):
            print(f"step {offset} / {grid_step * grid_step}")
            ox = offset % grid_step
            oy = offset // grid_step
            grid_pts[:, :, 1] = (
                torch.arange(grid_width).repeat(grid_height) * grid_step + ox
            )
            grid_pts[:, :, 2] = (
                torch.arange(grid_height).repeat_interleave(grid_width) * grid_step + oy
            )
            tracks_step, visibilities_step = _compute_sparse_tracks(
                video=video,
                queries=grid_pts,
                backward_tracking=backward_tracking,
            )
            tracks = smart_cat(tracks, tracks_step, dim=2)
            visibilities = smart_cat(visibilities, visibilities_step, dim=2)

        return tracks, visibilities

def _compute_sparse_tracks(
        frames, 
        queries,
        segm_mask=None,
        grid_size=0,
        add_support_grid=False,
        grid_query_frame=0,
        backward_tracking=False,
    ):  
        
        H = 480
        W = 720
        T = 49
        stride = 16
        latent_shape = (30, 45)

        device = queries.device
        
        # B, T, C, H, W = video.shape

        # video = video.reshape(B * T, C, H, W)
        # video = F.interpolate(
        #     video, tuple(latent_shape), mode="bilinear", align_corners=True
        # )
        # video = video.reshape(B, T, 3, latent_shape[0], latent_shape[1])

        if queries is not None:
            B, N, D = queries.shape
            assert D == 3
            queries = queries.clone()
            queries[:, :, 1:] *= queries.new_tensor(
                [
                    (latent_shape[1] - 1) / (W - 1),
                    (latent_shape[0] - 1) / (H - 1),
                ]
            )
        elif grid_size > 0:
            grid_pts = get_points_on_a_grid(
                grid_size, latent_shape, device=device
            )
            if segm_mask is not None:
                segm_mask = F.interpolate(
                    segm_mask, tuple(latent_shape), mode="nearest"
                )
                point_mask = segm_mask[0, 0][
                    (grid_pts[0, :, 1]).round().long().cpu(),
                    (grid_pts[0, :, 0]).round().long().cpu(),
                ].bool()
                grid_pts = grid_pts[:, point_mask]

            queries = torch.cat(
                [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                dim=2,
            ).repeat(B, 1, 1)

        if add_support_grid:
            grid_pts = get_points_on_a_grid(
                self.support_grid_size, latent_shape, device=device
            )
            grid_pts = torch.cat(
                [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
            )
            grid_pts = grid_pts.repeat(B, 1, 1)
            queries = torch.cat([queries, grid_pts], dim=1)

        # tracks, visibilities, *_ = self.model.forward(
        #     video=video, queries=queries, iters=6
        # )

        # queries (FloatTensor[B, N, 3]): point queries.
        # video = B T C H W
        # queries = B N 3
        # coords_init = B S N 2
        # vis_init = B S N 1
        
        # The first channel is the frame number
        # The rest are the coordinates of points we want to track
        queried_frames = queries[:, :, 0].long()

        queried_coords = queries[..., 1:]
        queried_coords = queried_coords / stride
        
        query_frames, key_frames = frames
        # We compute track features
        query_track = get_track_feat(
            query_frames,
            queried_frames,
            queried_coords,
        )
        
        import math
        head_dim = 30
        do_softmax = True
        attn_score = (query_track @ key_frames.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_score = attn_score.softmax(dim=-1)

        H, W = attn_score.shape[-2:]
        b = attn_score.shape[0]
        # get matches corresponding to maximum in correlation
        (x_source, y_source, x_target, y_target, score) = corr_to_matches(
            attn_score.view(b, 1, N, H, W).unsqueeze(1),
            get_maximum=True, do_softmax=do_softmax)

        # x_source dimension is B x H*W
        mapping_est = torch.cat((x_source.unsqueeze(-1), y_source.unsqueeze(-1)), dim=-1).view(b, H, W, 2).permute(0, 3, 1, 2)
        




        return tracks
     

        