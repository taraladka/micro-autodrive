"""
visualize_real.py -- Dashcam Inference Visualiser
=================================================
Loads the trained model and runs inference on the real-world dataset,
overlaying human steering versus AI steering with richer playback telemetry.
"""

import collections
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.model import load_model_bundle
try:
    from data_loader import STEERING_NORM_FACTOR, CROP_TOP, CROP_BOTTOM
except ModuleNotFoundError:
    from v2_real.data_loader import STEERING_NORM_FACTOR, CROP_TOP, CROP_BOTTOM

DATA_DIR = os.path.join(BASE_DIR, "Data", "driving_dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

SPEED_STEPS = [5, 15, 30, 60, 120]
DEFAULT_SPEED_IDX = 2


def get_latest_real_model():
    models = glob.glob(os.path.join(MODEL_DIR, "real_*_final.pth"))
    if not models:
        models = glob.glob(os.path.join(BASE_DIR, "checkpoints", "real_*.pth"))
    if not models:
        return None
    models.sort(key=os.path.getmtime)
    return models[-1]


def draw_steering_wheel(img, center, radius, angle_rad, color, thickness=2):
    cv2.circle(img, center, radius, color, thickness)
    cv2.circle(img, center, 4, color, -1)
    pt1 = (
        int(center[0] + radius * np.cos(angle_rad + np.pi)),
        int(center[1] + radius * np.sin(angle_rad + np.pi)),
    )
    pt2 = (
        int(center[0] + radius * np.cos(angle_rad)),
        int(center[1] + radius * np.sin(angle_rad)),
    )
    cv2.line(img, pt1, pt2, color, thickness)
    pt3 = (
        int(center[0] + radius * np.cos(angle_rad + np.pi / 2)),
        int(center[1] + radius * np.sin(angle_rad + np.pi / 2)),
    )
    cv2.line(img, center, pt3, color, thickness)


def prepare_model_input(frame_bgr: np.ndarray):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = rgb[CROP_TOP:CROP_BOTTOM, :]
    rgb = cv2.resize(rgb, (96, 64))
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor, rgb


def _turn_label(angle_deg: float) -> str:
    if angle_deg > 15:
        return "RIGHT TURN"
    if angle_deg < -15:
        return "LEFT TURN"
    return "STRAIGHT / SMALL CORRECTION"


def main():
    print("\n" + "=" * 60)
    print("  MICRO-AUTODRIVE -- REAL ROAD VISUALIZER")
    print("=" * 60)

    model_path = get_latest_real_model()
    if not model_path:
        print("[!] No trained model found in models/ or checkpoints/ directory.")
        print("    Please run `python v2_real/train_real.py` first.")
        sys.exit(1)

    print(f"[i] Loading model: {os.path.basename(model_path)}")
    model, metadata = load_model_bundle(model_path, map_location="cpu")
    model.eval()
    model_type = metadata.get("model_type", "cnn")
    model_cfg = metadata.get("config", {})

    print(f"[i] Loading dataset from: {DATA_DIR}")
    data_txt = os.path.join(DATA_DIR, "data.txt")
    if not os.path.exists(data_txt):
        print("[!] data.txt not found.")
        sys.exit(1)

    with open(data_txt, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    print(f"[i] Dataset has {len(lines):,} frames.")
    print(f"[i] Model type: {model_type}")
    print("\n[CONTROLS]")
    print("  [ESC] or [Q] : Quit")
    print("  [SPACE]      : Pause / Resume")
    print("  [S]          : Step forward one frame (when paused)")
    print("  [+] / [-]    : Speed up / slow down playback")
    print("\nStarting visualizer...\n")

    win_name = "Micro-AutoDrive | Real Road Inference"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1000, 620)

    paused = False
    speed_idx = DEFAULT_SPEED_IDX
    hidden = None
    prev_frame_id = None
    prev_crop_gray = None
    fps = 0.0
    t_prev = time.time()
    recent_errors = collections.deque(maxlen=30)

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            continue

        img_name, gt_angle_deg = parts[0], float(parts[1])
        stem, _ = os.path.splitext(img_name)
        try:
            frame_id = int(stem)
        except ValueError:
            frame_id = prev_frame_id + 1 if prev_frame_id is not None else 0

        img_path = os.path.join(DATA_DIR, img_name)
        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        tensor_in, cropped_rgb = prepare_model_input(frame)
        if model_type == "temporal_gru":
            if prev_frame_id is not None and frame_id != prev_frame_id + 1:
                hidden = None
            with torch.no_grad():
                ps, hidden = model.predict_step(tensor_in, hidden)
                ai_steer_norm = ps.item()
        else:
            with torch.no_grad():
                ps, _, _ = model(tensor_in)
                ai_steer_norm = ps.item()

        ai_angle_deg = ai_steer_norm * STEERING_NORM_FACTOR
        error_deg = abs(gt_angle_deg - ai_angle_deg)
        recent_errors.append(error_deg)

        crop_gray = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2GRAY)
        if prev_crop_gray is None:
            motion_proxy = 0.0
        else:
            motion_proxy = float(np.mean(cv2.absdiff(crop_gray, prev_crop_gray)))
        prev_crop_gray = crop_gray
        prev_frame_id = frame_id

        t_now = time.time()
        fps = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        frame_display = cv2.resize(frame, (1000, 500))
        panel = np.zeros((140, 1000, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)
        display = np.vstack([frame_display, panel])

        gt_rad = gt_angle_deg * np.pi / 180.0
        draw_steering_wheel(display, (120, 565), 45, gt_rad, (0, 220, 0), thickness=3)
        cv2.putText(display, "HUMAN", (88, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
        cv2.putText(display, f"{gt_angle_deg:>6.1f} deg", (70, 602), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 1)

        ai_rad = ai_angle_deg * np.pi / 180.0
        draw_steering_wheel(display, (880, 565), 45, ai_rad, (0, 80, 255), thickness=3)
        cv2.putText(display, "AI", (872, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)
        cv2.putText(display, f"{ai_angle_deg:>6.1f} deg", (845, 602), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 1)

        error_color = (0, 200, 0) if error_deg < 10 else (0, 165, 255) if error_deg < 30 else (0, 0, 255)
        rolling_mae = float(np.mean(recent_errors)) if recent_errors else error_deg

        cv2.rectangle(display, (0, 0), (520, 32), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"Frame {i:05d}/{len(lines)} | Delay {SPEED_STEPS[speed_idx]}ms | FPS {fps:4.1f}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
        )

        cv2.putText(display, f"Error now: {error_deg:5.1f} deg", (300, 548), cv2.FONT_HERSHEY_SIMPLEX, 0.65, error_color, 2)
        cv2.putText(display, f"Rolling MAE: {rolling_mae:5.1f} deg", (300, 578), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)
        cv2.putText(display, f"Turn state: {_turn_label(ai_angle_deg)}", (300, 608), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)

        info_x = 300
        cv2.putText(display, f"Model: {model_type}", (info_x, 638), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1)
        cv2.putText(display, f"Seq len: {model_cfg.get('sequence_length', 1)}", (info_x + 180, 638), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1)
        cv2.putText(display, f"Motion proxy: {motion_proxy:4.1f}", (info_x + 340, 638), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (200, 200, 200), 1)
        cv2.putText(display, "Vehicle speed: N/A in Sully dataset", (300, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (150, 180, 255), 1)

        if paused:
            cv2.putText(display, "PAUSED", (900, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow(win_name, display)

        while True:
            delay = 0 if paused else SPEED_STEPS[speed_idx]
            key = cv2.waitKey(delay) & 0xFF

            if key in (27, ord("q"), ord("Q")):
                cv2.destroyAllWindows()
                print("[i] Visualizer closed by user.")
                return
            if key == ord(" "):
                paused = not paused
                if not paused:
                    break
            elif key in (ord("s"), ord("S")) and paused:
                break
            elif key in (ord("+"), ord("=")):
                speed_idx = max(0, speed_idx - 1)
            elif key in (ord("-"), ord("_")):
                speed_idx = min(len(SPEED_STEPS) - 1, speed_idx + 1)
            elif not paused:
                break

    cv2.destroyAllWindows()
    print("[i] Finished dataset playback.")


if __name__ == "__main__":
    main()
