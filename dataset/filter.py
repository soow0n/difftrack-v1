import os
from pathlib import Path


for model in ['cogvideox_i2v_2b', 'cogvideox_i2v_5b']:
    for scene in ['fg', 'bg']:
        for data in ['image']:
            # Set your directory path
            folder = Path(f"/home/cvlab16/projects/soowon/diff-track/iccv2025/code-refactor/dataset/{model}/{scene}/{data}")

            # Get all .npy files and sort them
            files = sorted(folder.glob("*.png"))

            # Sort naturally by numeric part
            files = sorted(files, key=lambda f: int(f.stem))

            # Rename each file
            for idx, file in enumerate(files):
                new_name = folder / f"{idx:03d}.png"
                print(f"Renaming {file.name} -> {new_name.name}")
                file.rename(new_name)