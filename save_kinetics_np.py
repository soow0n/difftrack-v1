import glob
import os
import pickle

import io
from PIL import Image
import numpy as np

def decode(frame):
    byteio = io.BytesIO(frame)
    img = Image.open(byteio)
    return np.array(img)


all_paths = glob.glob(os.path.join('/mnt/ssd1/PointTracking/tapvid/kinetics-dataset', "*_of_0010.pkl"))
points_dataset = []
for pickle_path in all_paths:
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
        points_dataset = points_dataset + data

save_dir = "/mnt/ssd3/kinetics_frames_npy"
os.makedirs(save_dir, exist_ok=True)
breakpoint()
for idx, data in enumerate(points_dataset):
    frames = data['video']
    if isinstance(frames[0], bytes):
        frames = np.array([decode(frame) for frame in frames])

    save_path = os.path.join(save_dir, f"video_{idx:05d}.npy")
    np.save(save_path, frames)
