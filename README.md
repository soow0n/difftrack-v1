<div align="center">
<h1>DiffTrack: Emergent Temporal Correspondences from Video Diffusion Transformers</h1>

[**Jisu Nam**]()<sup>1*</sup> · [**Soowon Son**]()<sup>1*</sup> · [**Dahyun Chung**]()<sup>2</sup> · [**Jiyoung Kim**]()<sup>1</sup> · [**Siyoon Jin**]()<sup>1</sup> · [**Junhwa Hur**]()<sup>3</sup> · [**Seungryong Kim**]()<sup>1</sup>

<sup>1</sup>KAIST AI&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>3</sup>Google DeepMind

<span style="font-size: 1.5em;"><b>CVPR 2025</b></span>

<a href=""><img src='https://img.shields.io/badge/arXiv-DiffTrack-red' alt='Paper PDF'></a>
<a href='https://cvlab-kaist.github.io/DiffTrack/'><img src='https://img.shields.io/badge/Project_Page-DiffTrack-green' alt='Project Page'></a>



</div>

***How do VDMs internally establish and represent temporal correspondences across frames?***

To answer this question, DiffTrack constructs:
- **Dataset of prompt-generated video with pseudo GT tracking annotations**
- **Novel evaluation metrics for analyzing temporal correspondences**

We also applicate DiffTrack to:
- **Zero-shot point tracking**, where it acheives SOTA performance compared to existing vision foundation and self-supervised video models
- **Motion-enhanced video generation** with a novel guidance method CAG


## Installation

```bash
git clone https://github.com/<your-org>/diff-track.git
cd diff-track

conda create -n diff-track python=3.10 -y
conda activate diff-track
pip install -r requirements.txt

cd diffusers
pip install -e .
```
</br>

# 1. Transformer Analysis

### Analyze Generated Videos
For more analysis with other model, please refer to `scripts/analysis` directory. 

```bash
model=cogvideox_t2v_2b
scene=fg
python analyze_generation.py \
    --output_dir ./output \
    --model $model --video_mode $scene --num_inference_steps 50 \
    --pck --affinity_score \
    --vis_timesteps 49 --vis_layers 17 \
    --vis_attn_map --pos_h 16 24 --pos_w 16 36 --vis_track \
    --txt_path ./dataset/txt_prompts/$scene.txt \
    --idx_path ./dataset/$model/${scene}_50.txt \
    --track_dir ./dataset/$model/$scene/tracks \
    --visibility_dir ./dataset/$model/$scene/visibility \
    --device cuda:0
```

#### Key Options

- `--model`: Supported models include `cogvideox_t2v_2b`, `cogvideox_t2v_5b`, `cogvideox_i2v_2b`, `cogvideox_i2v_5b`, `hunyuan_t2v`.
- `--video_mode`: Set to `fg` for object-centric or `bg` for scenic videos.
- `--pck`: Computes matching accuracy (PCK) using both query-key and intermediate features.
- `--affinity_score`: Computes max and sum of cross-frame attention.
- `--vis_attn_map`: Aggregates cost maps for attention visualization.
- `--vis_track`: Visualizes trajectory using query-key descriptors.



*This script should reproduce videos in `sample`.*

</br>

### Analyze Real Videos (TAP-Vid)
*(Implemented only for CogVideoX-2B/5B)*

```bash
python analyze_real.py \
    --output_dir ./output \
    --model cogvideox_t2v_2b --num_inference_steps 50 \
    --pck --affinity_score \
    --resize_h 480 --resize_w 720 \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --device cuda:0
```
</br>


# 2. Zero-Shot Point Tracking

### Download Dataset

```bash
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip
```

For TAP-Vid-Kinetics, please refer to the [TAP-Vid GitHub](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).


### Run Evaluation
For more evalutaion in other model and dataset, please refer to `scripts/point_tracking` directory

```bash
model=cogvideox_t2v_2b
python evaluate_tapvid.py \
    --model $model \
    --matching_layer 17 --matching_timestep 49 --inverse_step 49 \
    --output_dir ./output \
    --eval_dataset davis_first --tapvid_root /path/to/data \
    --resize_h 480 --resize_w 720 \
    --chunk_frame_interval --average_overlapped_corr \
    --vis_video --tracks_leave_trace 15 \
    --pipe_device cuda:0
```

#### Chunking Options

- `--chunk_len`: Number of frames per chunk. (default: `13`)
- `--chunk_frame_interval`: Interleave frames to reduce temporal gap.
- `--chunk_stride`: Stride for sliding window. (default: `1`)
- `--average_overlapped_corr`: Average overlapping correlation maps.

#### Cost Map Aggregation

- `--matching_layer`: Transformer layers for descriptor extraction. (e.g., `17` for cogvideox_t2v_2b).
- `--matching_timestep`: Denoising timesteps for descriptor extraction. (e.g., `49` for cogvideox_t2v_2b).

#### Dataset Options

- `--tapvid_root`: Path to TAP-Vid dataset.
- `--eval_dataset`: Choose from `davis_first`, etc.
- `--resize_h` / `--resize_w`: Resize video resolution.
- `--video_max_len`: Max length of input video.
- `--do_inversion` / `--add_noise`: Modify inversion strategy.

#### Visualization Options

- `--vis_video`: Visualize trajectories on video.
- `--tracks_leave_trace`: Number of frames for trajectory trail.


</br>


# 3. Cross Attention Guidance (CAG)

*(Implemented only for CogVideoX-2B/5B)*

```bash
CUDA_VISIBLE_DEVICES=0 python motion_guidance.py \
    --output_dir ./output \
    --model_version 2b \
    --txt_path ./dataset/txt_prompts/cag_prompts.txt \
    --pag_layers 13 17 21 \
    --pag_scale 1 \ 
    --cfg_scale 6
```
#### Key Options
- `--model_version`: Supported cogvideox models include `2b`, `5b`.
- `--pag_layers`: Layers where CAG is applied (e.g., `[13, 17, 21]` for 2B, `[15, 17, 18]` for 5B).
- `--pag_scale`: Cross attention guidance scale (default: `1.0`).
- `--cfg_scale`: Classifier-Free Guidance scale (default: `6.0`).


### Citing this Work
Please use the following bibtex to cite our work:
```

```