"""
train_real.py -- Training Pipeline for Real-World Dashcam Data
==============================================================
Run from project root: python v2_real/train_real.py

Pipeline:
  1. Loads Sully Chen dataset via RealDrivingDataset (images + steering)
  2. Builds contiguous short frame sequences for temporal learning
  3. Splits sequences into 80% Train / 20% Validation
  4. Trains a CNN + GRU model on steering output only
  4. Output: training loss graphs, JSON summary, saved model
"""

import os
import sys
import json
import time
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
from tqdm import tqdm

# ─── Path Setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.model import TemporalDrivingGRU, save_model_bundle
try:
    from data_loader import RealDrivingDataset, STEERING_NORM_FACTOR
except ModuleNotFoundError:
    from v2_real.data_loader import RealDrivingDataset, STEERING_NORM_FACTOR

# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sns.set_style("darkgrid")

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(BASE_DIR, "Data", "driving_dataset")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for _d in [MODEL_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(_d, exist_ok=True)

NUM_CORES = os.cpu_count() or 2

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    "batch_size":            64,
    "learning_rate":         0.0003,
    "num_epochs":            25,
    "val_split":             0.2,
    "seed":                  42,
    "save_frequency":        5,
    "cpu_threads":           max(1, min(8, NUM_CORES // 2)),
    "num_workers":           0,
    "live_viewer":           False,
    "validate_every":        2,
    "cache_images":          True,
    "sequence_length":       5,
    "target_smoothing":      3,
    "gru_hidden_size":       64,
    "gru_layers":            1,
    "straight_keep_prob":    0.2,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

DEVICE = torch.device("cpu")


class RealSequenceDataset(Dataset):
    """Sequence view over one shared base dataset for temporal steering."""

    def __init__(
        self,
        base_dataset: RealDrivingDataset,
        end_indices: list[int],
        sequence_length: int,
        augment: bool,
        target_smoothing: int,
    ):
        self.base_dataset = base_dataset
        self.end_indices = end_indices
        self.sequence_length = sequence_length
        self.augment = augment
        self.target_smoothing = max(1, target_smoothing)

    def __len__(self):
        return len(self.end_indices)

    def __getitem__(self, idx: int):
        end_idx = self.end_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        frame_indices = range(start_idx, end_idx + 1)
        frames = [self.base_dataset.get_base_image(i) for i in frame_indices]

        flip = self.augment and random.random() < 0.5
        apply_hsv = self.augment and random.random() < 0.5
        sat_scale = random.uniform(0.6, 1.4) if apply_hsv else 1.0
        val_scale = random.uniform(0.6, 1.4) if apply_hsv else 1.0
        apply_blur = self.augment and random.random() < 0.2

        if flip:
            frames = [cv2.flip(img, 1) for img in frames]

        if apply_hsv:
            updated = []
            for img in frames:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= sat_scale
                hsv[:, :, 2] *= val_scale
                hsv = np.clip(hsv, 0, 255).astype(np.uint8)
                updated.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
            frames = updated

        if apply_blur:
            frames = [cv2.GaussianBlur(img, (3, 3), 0) for img in frames]

        target_start = max(start_idx, end_idx - self.target_smoothing + 1)
        target_steer = float(np.mean(self.base_dataset.steers[target_start:end_idx + 1]))
        if flip:
            target_steer = -target_steer

        seq_tensor = torch.stack([
            torch.from_numpy(img).permute(2, 0, 1).byte() for img in frames
        ], dim=0)
        steer_tensor = torch.tensor([target_steer], dtype=torch.float32)
        return seq_tensor, steer_tensor


def _build_valid_sequence_end_indices(base_dataset: RealDrivingDataset, sequence_length: int) -> list[int]:
    valid = []
    for end_idx in range(sequence_length - 1, len(base_dataset)):
        start_idx = end_idx - sequence_length + 1
        contiguous = True
        for i in range(start_idx + 1, end_idx + 1):
            if base_dataset.frame_ids[i] != base_dataset.frame_ids[i - 1] + 1:
                contiguous = False
                break
        if contiguous:
            valid.append(end_idx)
    return valid


def _balance_sequence_indices(
    base_dataset: RealDrivingDataset,
    end_indices: list[int],
    keep_prob: float,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    balanced = []
    for end_idx in end_indices:
        angle_deg = abs(base_dataset.steers[end_idx] * STEERING_NORM_FACTOR)
        if angle_deg < 2.0 and rng.random() > keep_prob:
            continue
        balanced.append(end_idx)
    return balanced


def _configure_runtime() -> int:
    torch_threads = min(CONFIG["cpu_threads"], NUM_CORES)
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if CONFIG["cache_images"]:
        return 0
    return max(0, min(CONFIG["num_workers"], NUM_CORES - 1))


def _show_live_view(window_name: str, frame: np.ndarray) -> bool:
    """Best-effort live preview that disables itself if GUI display is unavailable."""
    try:
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)
        return True
    except cv2.error as e:
        print(f"[WARN] Live viewer disabled: {e}")
        return False

# ─── Training & Evaluation ──────────────────────────────────────────────────

def train_and_validate(
    model: TemporalDrivingGRU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    effective_num_workers: int,
):
    print("\n" + "="*60)
    print("  [TRAIN]  TRAINING ON REAL ROAD DATA")
    print("="*60)
    print(f"  Device        : {DEVICE}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Batch Size    : {CONFIG['batch_size']}")
    print(f"  PyTorch threads: {CONFIG['cpu_threads']}")
    print(f"  Workers       : {effective_num_workers}")

    model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-5)
    regression_loss = nn.SmoothL1Loss(beta=0.05)

    history = {"train_loss": [], "val_loss": [], "val_epoch": []}
    best_val_loss = float("inf")
    best_val_epoch = None
    best_state = None
    live_view_enabled = CONFIG["live_viewer"]
    live_view_window = "Micro-AutoDrive | Live Training Viewer"
    last_val_avg = None

    try:
        for epoch in range(1, CONFIG["num_epochs"] + 1):
            # -- Train --
            model.train()
            train_running = 0.0

            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{CONFIG['num_epochs']} [Train]")
            for i, (imgs, steers) in enumerate(pbar):
                imgs   = imgs.to(DEVICE)
                steers = steers.to(DEVICE)

                ps, _ = model(imgs)
                loss = regression_loss(ps, steers)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_running += loss.item()

                pbar.set_postfix({"Loss": f"{train_running/(i+1):.4f}"})

                # --- Live Visualization (every 20 batches) ---
                if live_view_enabled and i % 20 == 0:
                    img_np = imgs[0, -1].cpu().permute(1, 2, 0).numpy()
                    # img_np is uint8 [0,255] since model handles normalization
                    img_disp = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                    img_disp = cv2.resize(img_disp, (480, 320))

                    gt_steer = steers[0].item()
                    ai_steer = ps[0].detach().item()  # FIX: detach before .item()

                    gt_deg = gt_steer * STEERING_NORM_FACTOR
                    ai_deg = ai_steer * STEERING_NORM_FACTOR

                    cv2.putText(img_disp, f"Human : {gt_deg:>6.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img_disp, f"AI Pred: {ai_deg:>6.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(img_disp, f"Epoch {epoch}/{CONFIG['num_epochs']}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    live_view_enabled = _show_live_view(live_view_window, img_disp)

            train_avg = train_running / len(train_loader)
            history["train_loss"].append(train_avg)

            should_validate = (
                epoch == 1 or
                epoch == CONFIG["num_epochs"] or
                epoch % CONFIG["validate_every"] == 0
            )

            if should_validate:
                model.eval()
                val_running = 0.0

                pbar_val = tqdm(val_loader, desc=f"  Epoch {epoch}/{CONFIG['num_epochs']} [Val]  ")
                with torch.no_grad():
                    for i, (imgs, steers) in enumerate(pbar_val):
                        imgs   = imgs.to(DEVICE)
                        steers = steers.to(DEVICE)
                        ps, _ = model(imgs)
                        loss = regression_loss(ps, steers)
                        val_running += loss.item()

                        pbar_val.set_postfix({"Loss": f"{val_running/(i+1):.4f}"})

                last_val_avg = val_running / len(val_loader)
                history["val_loss"].append(last_val_avg)
                history["val_epoch"].append(epoch)
                if last_val_avg < best_val_loss:
                    best_val_loss = last_val_avg
                    best_val_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            else:
                print(f"  [VAL] Skipped full validation at epoch {epoch} to save time.")

            val_avg = last_val_avg if last_val_avg is not None else train_avg

            scheduler.step()
            bar = "█" * int(30 * max(0, 1 - train_avg / (history["train_loss"][0] + 1e-9)))
            print(f"  Ep {epoch:>3}/{CONFIG['num_epochs']} | "
                  f"T-Loss={train_avg:.5f} | V-Loss={val_avg:.5f} | LR={scheduler.get_last_lr()[0]:.6f} | {bar}")

            if epoch % CONFIG["save_frequency"] == 0:
                ck = os.path.join(CHECKPOINT_DIR, f"real_epoch_{epoch:03d}.pth")
                torch.save(model.state_dict(), ck)
                print(f"  [CKPT] Checkpoint saved -> {ck}")

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user.")
        print("[i] Saving current state before exiting...")

    if live_view_enabled:
        cv2.destroyAllWindows()
    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_epoch"] = best_val_epoch
    history["best_val_loss"] = best_val_loss if best_state is not None else None
    return history


def evaluate_real_metrics(model: TemporalDrivingGRU, loader: DataLoader) -> dict:
    model.eval()
    preds_deg = []
    targets_deg = []

    with torch.no_grad():
        for imgs, steers in loader:
            imgs = imgs.to(DEVICE)
            steers = steers.to(DEVICE)
            ps, _ = model(imgs)
            preds_deg.extend((ps.squeeze(1).cpu().numpy() * STEERING_NORM_FACTOR).tolist())
            targets_deg.extend((steers.squeeze(1).cpu().numpy() * STEERING_NORM_FACTOR).tolist())

    preds = np.array(preds_deg, dtype=np.float32)
    targets = np.array(targets_deg, dtype=np.float32)
    signed_err = preds - targets
    abs_err = np.abs(signed_err)

    def _mask_mae(mask):
        if not np.any(mask):
            return None
        return float(np.mean(abs_err[mask]))

    straight_mask = np.abs(targets) < 5.0
    left_mask = targets < -5.0
    right_mask = targets > 5.0

    return {
        "mae_deg": float(np.mean(abs_err)),
        "rmse_deg": float(np.sqrt(np.mean(np.square(signed_err)))),
        "median_abs_error_deg": float(np.median(abs_err)),
        "bias_deg": float(np.mean(signed_err)),
        "within_1deg": float(np.mean(abs_err <= 1.0)),
        "within_2deg": float(np.mean(abs_err <= 2.0)),
        "within_5deg": float(np.mean(abs_err <= 5.0)),
        "within_10deg": float(np.mean(abs_err <= 10.0)),
        "straight_mae_deg": _mask_mae(straight_mask),
        "left_turn_mae_deg": _mask_mae(left_mask),
        "right_turn_mae_deg": _mask_mae(right_mask),
        "sample_count": int(len(abs_err)),
    }

# ─── Outputs ────────────────────────────────────────────────────────────────

def save_graphs(history: dict, run_name: str, metrics: dict | None = None):
    if not history["train_loss"]:
        print("  [!] No epochs completed, skipping graph.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    epochs = range(1, len(history["train_loss"]) + 1)
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color="#58a6ff", lw=2, label="Train Loss")
    ax.plot(history["val_epoch"], history["val_loss"], color="#f78166", lw=2,
            marker="o", label="Val Loss (measured epochs)")
    ax.set_title(f"Real Data Training - {run_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SmoothL1 Loss")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")

    ax = axes[0, 1]
    if metrics:
        labels = ["±1°", "±2°", "±5°", "±10°"]
        vals = [
            metrics["within_1deg"] * 100,
            metrics["within_2deg"] * 100,
            metrics["within_5deg"] * 100,
            metrics["within_10deg"] * 100,
        ]
        ax.bar(labels, vals, color=["#f78166", "#ffa657", "#3fb950", "#58a6ff"])
        ax.set_ylim(0, 100)
        ax.set_title("Validation Accuracy Bands")
        ax.set_ylabel("Samples Within Error Band (%)")
    else:
        ax.set_title("Validation Accuracy Bands")

    ax = axes[1, 0]
    if metrics:
        labels = ["MAE", "RMSE", "Median"]
        vals = [
            metrics["mae_deg"],
            metrics["rmse_deg"],
            metrics["median_abs_error_deg"],
        ]
        ax.bar(labels, vals, color=["#58a6ff", "#d2a8ff", "#79c0ff"])
        ax.axhline(2.0, color="#f78166", linestyle="--", linewidth=1.5, label="2 deg target")
        ax.set_title("Validation Error in Degrees")
        ax.set_ylabel("Degrees")
        ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")
    else:
        ax.set_title("Validation Error in Degrees")

    ax = axes[1, 1]
    ax.axis("off")
    if metrics:
        lines = [
            f"MAE: {metrics['mae_deg']:.2f} deg",
            f"RMSE: {metrics['rmse_deg']:.2f} deg",
            f"Bias: {metrics['bias_deg']:+.2f} deg",
            f"Straight MAE: {metrics['straight_mae_deg'] if metrics['straight_mae_deg'] is not None else 'n/a'}",
            f"Left-turn MAE: {metrics['left_turn_mae_deg'] if metrics['left_turn_mae_deg'] is not None else 'n/a'}",
            f"Right-turn MAE: {metrics['right_turn_mae_deg'] if metrics['right_turn_mae_deg'] is not None else 'n/a'}",
            "How to read:",
            "- Lower MAE/RMSE is better.",
            "- Bias near 0 means no consistent left/right drift.",
            "- ±1° and ±2° rates show how often the model is truly tight.",
        ]
        for i, line in enumerate(lines):
            ax.text(0.02, 0.95 - i * 0.09, line, color="#e6edf3", fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{run_name}_graphs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [GRAPH] Saved -> {path}")

def save_summary(history: dict, run_name: str, train_samples: int, val_samples: int, metrics: dict | None = None) -> dict:
    # FIX: Guard against empty history after early exit
    if not history["train_loss"]:
        print("  [!] No training history, skipping JSON summary.")
        return {}

    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "dataset": "Sully Chen Dataset 1 (2017)",
        "config": CONFIG,
        "validation_protocol": {
            "split": "chronological holdout",
            "purged_overlap_frames": CONFIG["sequence_length"] - 1,
            "scope": "one recording; not an independent-drive evaluation",
        },
        "model": {
            "type": "temporal_gru",
            "sequence_length": CONFIG["sequence_length"],
            "hidden_size": CONFIG["gru_hidden_size"],
            "num_layers": CONFIG["gru_layers"],
        },
        "data": {
            "train_sequences": train_samples,
            "val_sequences": val_samples,
        },
        "training": {
            "epochs_completed": len(history["train_loss"]),
            "initial_train_loss": float(history["train_loss"][0]),
            "final_train_loss":   float(history["train_loss"][-1]),
            "initial_val_loss": float(history["val_loss"][0]) if history["val_loss"] else None,
            "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
            "best_val_epoch": history.get("best_val_epoch"),
            "best_val_loss": history.get("best_val_loss"),
        },
        "evaluation": metrics or {},
    }
    path = os.path.join(RESULTS_DIR, f"{run_name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [JSON] Saved -> {path}")
    return summary

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  MICRO-AUTODRIVE -- REAL DATA TRAINING")
    print("="*60)

    effective_num_workers = _configure_runtime()
    if torch.cuda.is_available():
        global DEVICE
        DEVICE = torch.device("cuda")
        print("[OK] GPU detected:", torch.cuda.get_device_name(0))
    else:
        print(
            f"[i]  Running on CPU with {CONFIG['cpu_threads']} PyTorch threads "
            f"/ {effective_num_workers} data-loader workers."
        )

    run_name = f"real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # 1. Load Dataset
    print(f"  Loading dataset from {DATA_DIR}...")
    try:
        base_dataset = RealDrivingDataset(
            DATA_DIR,
            augment=False,
            balance_seed=CONFIG["seed"],
            cache_images=CONFIG["cache_images"],
            balance_straight=False,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Please ensure you extracted the dataset to the correct folder.")
        sys.exit(1)

    valid_end_indices = _build_valid_sequence_end_indices(base_dataset, CONFIG["sequence_length"])
    val_size = int(len(valid_end_indices) * CONFIG["val_split"])
    train_size = len(valid_end_indices) - val_size
    split_index = train_size
    purge_count = CONFIG["sequence_length"] - 1
    train_end_indices = valid_end_indices[:max(0, split_index - purge_count)]
    val_end_indices = valid_end_indices[split_index:]
    if not train_end_indices or not val_end_indices:
        raise ValueError("Not enough contiguous sequences for the requested train/validation split.")
    train_end_indices = _balance_sequence_indices(
        base_dataset,
        train_end_indices,
        keep_prob=CONFIG["straight_keep_prob"],
        seed=CONFIG["seed"],
    )

    train_dataset = RealSequenceDataset(
        base_dataset,
        train_end_indices,
        sequence_length=CONFIG["sequence_length"],
        augment=True,
        target_smoothing=CONFIG["target_smoothing"],
    )
    val_dataset = RealSequenceDataset(
        base_dataset,
        val_end_indices,
        sequence_length=CONFIG["sequence_length"],
        augment=False,
        target_smoothing=CONFIG["target_smoothing"],
    )

    print(f"  Raw frames     : {len(base_dataset):,}")
    print(f"  Seq length     : {CONFIG['sequence_length']}")
    print(f"  Valid sequences: {len(valid_end_indices):,}")
    print(f"  Train split    : {len(train_dataset):,}")
    print(f"  Val split      : {len(val_dataset):,}")
    if CONFIG["cache_images"]:
        print("  Cache mode    : enabled (single shared in-memory dataset)")

    use_persistent = effective_num_workers > 0
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG["batch_size"],
        shuffle=True, drop_last=True,
        num_workers=effective_num_workers,
        persistent_workers=use_persistent,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG["batch_size"],
        shuffle=False, drop_last=False,
        num_workers=effective_num_workers,
        persistent_workers=use_persistent,
        pin_memory=False
    )

    # 2. Initialize Model
    model = TemporalDrivingGRU(
        hidden_size=CONFIG["gru_hidden_size"],
        num_layers=CONFIG["gru_layers"],
    )

    # 3. Train
    t0 = time.time()
    history = train_and_validate(model, train_loader, val_loader, effective_num_workers)
    train_time = time.time() - t0
    print(f"  Training took {train_time/60:.1f} minutes.")

    # 4. Save Final Model
    model_path = os.path.join(MODEL_DIR, f"{run_name}_final.pth")
    save_model_bundle(
        model_path,
        model,
        model_type="temporal_gru",
        config={
            "hidden_size": CONFIG["gru_hidden_size"],
            "num_layers": CONFIG["gru_layers"],
            "sequence_length": CONFIG["sequence_length"],
            "target_smoothing": CONFIG["target_smoothing"],
            "best_val_epoch": history.get("best_val_epoch"),
            "best_val_loss": history.get("best_val_loss"),
        },
    )
    print(f"  [SAVED] Model -> {model_path}")

    # 5. Output Artifacts
    print("\n" + "="*60)
    print("  SAVING OUTPUTS")
    print("="*60)
    metrics = evaluate_real_metrics(model, val_loader)
    print(
        f"  [EVAL] Val MAE={metrics['mae_deg']:.2f} deg | "
        f"within ±1°={metrics['within_1deg']:.1%} | "
        f"within ±2°={metrics['within_2deg']:.1%}"
    )
    save_graphs(history, run_name, metrics)
    save_summary(history, run_name, len(train_dataset), len(val_dataset), metrics)
    print(f"  All results in: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
