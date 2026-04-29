"""
data_loader.py -- Real-World Dataset Loader for Micro-AutoDrive
===============================================================
Loads the Sully Chen dashcam dataset (images + steering angles)
and formats them for PyTorch training, matching the 96x64 resolution
expected by our DrivingCNN.
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Normalise steering angles (which can be up to 540 degrees on a real wheel)
# down to the [-1.0, 1.0] range outputted by our tanh() activation.
STEERING_NORM_FACTOR = 540.0

# Crop boundaries: Sully Chen images are 256x455.
# Top 100px = sky/trees (useless), bottom 56px = car hood (useless).
# Cropping to [100:200] gives us only the road surface.
CROP_TOP    = 100
CROP_BOTTOM = 200

class RealDrivingDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        target_w: int = 96,
        target_h: int = 64,
        augment: bool = True,
        balance_seed: int = 42,
        cache_images: bool = True,
        balance_straight: bool = True,
    ):
        self.data_dir  = data_dir
        self.target_w  = target_w
        self.target_h  = target_h
        self.augment   = augment
        self.cache_images = cache_images
        self.balance_straight = balance_straight
        self.balance_rng = random.Random(balance_seed)

        self.image_paths = []
        self.steers      = []
        self.frame_ids   = []
        self.cached_images = []

        self._load_metadata()

    def _load_metadata(self):
        txt_path = os.path.join(self.data_dir, "data.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Dataset metadata not found at {txt_path}")

        print(f"[DATA] Loading metadata from {txt_path}...")
        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 2:
                continue

            img_name  = parts[0]
            angle_deg = float(parts[1])

            img_path = os.path.join(self.data_dir, img_name)
            if not os.path.exists(img_path):
                continue

            stem, _ = os.path.splitext(img_name)
            try:
                frame_id = int(stem)
            except ValueError:
                continue

            # --- DATASET BALANCING ---
            # 90% of real driving is perfectly straight.
            # Drop 80% of near-zero steering frames to prevent Mode Collapse
            # (the AI just predicting "straight" always).
            if self.balance_straight and abs(angle_deg) < 2.0 and self.balance_rng.random() < 0.80:
                continue

            self.image_paths.append(img_path)
            self.frame_ids.append(frame_id)
            # Normalise to [-1.0, 1.0] range to match tanh output
            norm_steer = float(np.clip(angle_deg / STEERING_NORM_FACTOR, -1.0, 1.0))
            self.steers.append(norm_steer)

        print(f"[DATA] Loaded {len(self.image_paths):,} balanced samples (from {len(lines):,} raw frames).")

        if self.cache_images:
            self._build_image_cache()

    def _prepare_base_image(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[CROP_TOP:CROP_BOTTOM, :]
        img = cv2.resize(img, (self.target_w, self.target_h))
        return img

    def _build_image_cache(self):
        print("[DATA] Caching preprocessed images in RAM...")
        self.cached_images = [self._prepare_base_image(path) for path in self.image_paths]
        print(f"[DATA] Cached {len(self.cached_images):,} images.")

    def __len__(self):
        return len(self.image_paths)

    def get_base_image(self, idx: int) -> np.ndarray:
        if self.cache_images:
            return self.cached_images[idx].copy()
        return self._prepare_base_image(self.image_paths[idx])

    def get_sample(self, idx: int, augment: bool | None = None):
        steer    = self.steers[idx]
        use_augment = self.augment if augment is None else augment

        img = self.get_base_image(idx)

        # --- DATA AUGMENTATION (only during training, not validation) ---
        if use_augment:
            # 1. Horizontal flip: mirrors image + inverts steering angle
            if random.random() < 0.5:
                img   = cv2.flip(img, 1)
                steer = -steer

            # 2. Brightness & Saturation jitter: simulates weather/lighting changes
            if random.random() < 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.6, 1.4)
                hsv[:, :, 2] = hsv[:, :, 2] * random.uniform(0.6, 1.4)
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            # 3. Gaussian blur: simulates rain / dirty windshield
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (3, 3), 0)

        # Convert to PyTorch tensor format (C, H, W), keeping as uint8
        # The model's forward() handles /255.0 normalization internally.
        img_tensor   = torch.from_numpy(img).permute(2, 0, 1).byte()
        steer_tensor = torch.tensor([steer], dtype=torch.float32)

        return img_tensor, steer_tensor

    def __getitem__(self, idx: int):
        return self.get_sample(idx)

if __name__ == "__main__":
    import sys
    base_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "Data", "driving_dataset")

    try:
        ds = RealDrivingDataset(data_path)
        img, steer = ds[0]
        print(f"Test Sample 0: Image shape {img.shape}, Steer {steer.item():.4f}")
        print(f"Image dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
    except Exception as e:
        print(f"Error testing dataset: {e}")
