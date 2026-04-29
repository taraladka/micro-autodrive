"""
train.py -- Main Training Pipeline for Micro-AutoDrive
======================================================
Run with: python v1_sim/train.py

Pipeline:
  1. Generate track & collect expert demonstrations (seconds, CPU only)
  2. Train DrivingCNN via imitation learning (minutes, CPU only)
  3. Evaluate AI autonomously & capture frames
  4. Output: graphs, GIF, trajectory map, JSON, saved model
"""

import sys, os, json, math, random, time
# Force UTF-8 on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import imageio

from micro_env import MicroEnv, ExpertDriver
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.model import DrivingCNN, CAM_W, CAM_H

sns.set_style("darkgrid")

# ─── Paths (always relative to THIS script, not the cwd) ───────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR   = os.path.join(BASE_DIR, "training_data")
MODEL_DIR      = os.path.join(BASE_DIR, "models")
RESULTS_DIR    = os.path.join(BASE_DIR, "results")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

for _d in [TRAINING_DIR, MODEL_DIR, RESULTS_DIR, CHECKPOINT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    # Data collection
    "expert_episodes":       12,
    "max_steps_per_episode": 1500,
    # Training
    "batch_size":            64,
    "learning_rate":         0.001,
    "num_epochs":            25,
    # Evaluation
    "eval_episodes":         5,
    # Misc
    "seed":                  42,
    "gif_fps":               20,
    "save_frequency":        5,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

DEVICE = torch.device("cpu")   # GPU used automatically if available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("[OK] GPU detected:", torch.cuda.get_device_name(0))
else:
    print("[i]  Running on CPU -- this is fine for Micro-AutoDrive!")



# ─── Dataset ────────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self):
        self.images    = []
        self.steers    = []
        self.throttles = []
        self.brakes    = []
        self.rewards   = []

    def add(self, img, steer, throttle, brake, reward=0.0):
        self.images.append(img)
        self.steers.append(steer)
        self.throttles.append(throttle)
        self.brakes.append(brake)
        self.rewards.append(reward)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # CHW
        return (
            img,
            torch.tensor([self.steers[idx]],    dtype=torch.float32),
            torch.tensor([self.throttles[idx]],  dtype=torch.float32),
            torch.tensor([self.brakes[idx]],     dtype=torch.float32),
        )

    def save(self, path):
        np.savez_compressed(
            path,
            images=np.array(self.images, dtype=np.uint8),
            steers=np.array(self.steers),
            throttles=np.array(self.throttles),
            brakes=np.array(self.brakes),
            rewards=np.array(self.rewards),
        )
        print(f"  [SAVED] Dataset saved: {len(self)} samples -> {path}")

    @classmethod
    def load(cls, path):
        d = cls()
        data = np.load(path)
        d.images    = list(data["images"])
        d.steers    = list(data["steers"])
        d.throttles = list(data["throttles"])
        d.brakes    = list(data["brakes"])
        d.rewards   = list(data["rewards"])
        print(f"  [LOAD] Dataset loaded: {len(d)} samples <- {path}")
        return d


# ─── Expert Data Collection ─────────────────────────────────────────────────

def collect_expert_data(env: MicroEnv, n_episodes: int) -> DrivingDataset:
    print("\n" + "="*60)
    print("  [EXPERT]  COLLECTING EXPERT DEMONSTRATIONS")
    print("="*60)
    expert = ExpertDriver(env)
    dataset = DrivingDataset()

    for ep in range(n_episodes):
        obs = env.reset(new_track=(ep == 0))
        done = False
        step = 0
        ep_reward = 0.0

        while not done and step < CONFIG["max_steps_per_episode"]:
            steer, throttle, brake = expert.get_action()
            next_obs, reward, done, info = env.step((steer, throttle, brake))
            dataset.add(obs, steer, throttle, brake, reward)
            obs = next_obs
            ep_reward += reward
            step += 1

        print(f"  Ep {ep+1:>2}/{n_episodes} | steps={step:>4} | "
              f"reward={ep_reward:>7.1f} | collisions={info['collisions']}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset.save(os.path.join(TRAINING_DIR, f"expert_{ts}.npz"))
    return dataset


# ─── Training ───────────────────────────────────────────────────────────────

def train(model: DrivingCNN, dataset: DrivingDataset) -> list[float]:
    print("\n" + "="*60)
    print("  [TRAIN]  TRAINING IMITATION LEARNING MODEL")
    print("="*60)
    print(f"  Device        : {DEVICE}")
    print(f"  Dataset size  : {len(dataset):,}")
    print(f"  Parameters    : {sum(p.numel() for p in model.parameters()):,}")

    loader = DataLoader(dataset, batch_size=CONFIG["batch_size"],
                        shuffle=True, drop_last=True)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG["num_epochs"], eta_min=1e-5)
    mse = nn.MSELoss()
    epoch_losses = []

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        model.train()
        running = 0.0
        for imgs, steers, throttles, brakes in loader:
            imgs      = imgs.to(DEVICE)
            steers    = steers.to(DEVICE)
            throttles = throttles.to(DEVICE)
            brakes    = brakes.to(DEVICE)

            ps, pt, pb = model(imgs)
            loss = mse(ps, steers) + mse(pt, throttles) + mse(pb, brakes)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()

        scheduler.step()
        avg = running / len(loader)
        epoch_losses.append(avg)
        bar = "█" * int(30 * (1 - avg / (epoch_losses[0] + 1e-9)))
        print(f"  Epoch {epoch:>3}/{CONFIG['num_epochs']} | "
              f"loss={avg:.5f} | lr={scheduler.get_last_lr()[0]:.6f} | {bar}")

        if epoch % CONFIG["save_frequency"] == 0:
            ck = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ck)

    return epoch_losses


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model: DrivingCNN, env: MicroEnv) -> dict:
    print("\n" + "="*60)
    print("  [EVAL]  EVALUATING AI DRIVER")
    print("="*60)
    model.to(DEVICE)
    model.eval()

    all_rewards, all_lengths, all_collisions, all_speeds = [], [], [], []
    gif_frames = []
    ai_trajectory = []

    for ep in range(CONFIG["eval_episodes"]):
        obs = env.reset(new_track=(ep == 0))
        done = False
        step = 0
        ep_reward = 0.0
        speeds = []
        capture_gif = (ep == 0)

        while not done and step < CONFIG["max_steps_per_episode"]:
            t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                steer, throttle, brake = model(t)
            action = (steer.item(), throttle.item(), brake.item())
            obs, reward, done, info = env.step(action)

            ep_reward += reward
            speeds.append(info["speed_kmh"])
            step += 1

            if capture_gif and step % 3 == 0:
                frame = env.render_birdseye(show_trajectory=True)
                # Overlay HUD
                cv2.putText(frame, f"Speed: {info['speed_kmh']:.1f} km/h",
                            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Step: {step}",
                            (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1, cv2.LINE_AA)
                gif_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if ep == 0:
            ai_trajectory = env.get_trajectory()

        all_rewards.append(ep_reward)
        all_lengths.append(step)
        all_collisions.append(info["collisions"])
        all_speeds.append(float(np.mean(speeds)) if speeds else 0.0)
        print(f"  Ep {ep+1}/{CONFIG['eval_episodes']} | "
              f"steps={step:>4} | reward={ep_reward:>7.1f} | "
              f"speed={all_speeds[-1]:.1f} km/h | collisions={info['collisions']}")

    return {
        "rewards":     all_rewards,
        "lengths":     all_lengths,
        "collisions":  all_collisions,
        "speeds":      all_speeds,
        "gif_frames":  gif_frames,
        "ai_trajectory": ai_trajectory,
    }


# ─── Outputs ────────────────────────────────────────────────────────────────

def save_gif(frames: list, path: str) -> None:
    if not frames:
        return
    imageio.mimsave(path, frames, fps=CONFIG["gif_fps"], loop=0)
    print(f"  [GIF]  GIF saved  -> {path}")


def save_training_graphs(losses: list, eval_data: dict, run_name: str) -> None:  # noqa
    rewards    = eval_data["rewards"]
    lengths    = eval_data["lengths"]
    collisions = eval_data["collisions"]
    speeds     = eval_data["speeds"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes.flat:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    fig.suptitle(f"Micro-AutoDrive — {run_name}", color="#e6edf3", fontsize=15, y=1.01)

    # 1 Training loss
    ax = axes[0, 0]
    ax.plot(losses, color="#58a6ff", lw=1.5, alpha=0.7, label="batch loss")
    w = min(5, len(losses))
    ma = np.convolve(losses, np.ones(w)/w, mode="valid")
    ax.plot(range(w-1, len(losses)), ma, color="#f78166", lw=2.5, label="MA")
    ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")

    # 2 Eval rewards
    ax = axes[0, 1]
    ax.bar(range(1, len(rewards)+1), rewards, color="#3fb950", alpha=0.8, edgecolor="#238636")
    ax.axhline(np.mean(rewards), color="#f78166", lw=2, linestyle="--", label=f"mean={np.mean(rewards):.1f}")
    ax.set_title("Eval Episode Rewards"); ax.set_xlabel("Episode"); ax.set_ylabel("Total Reward")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")

    # 3 Episode lengths
    ax = axes[0, 2]
    ax.bar(range(1, len(lengths)+1), lengths, color="#58a6ff", alpha=0.8, edgecolor="#1f6feb")
    ax.axhline(CONFIG["max_steps_per_episode"], color="#f78166", lw=2,
               linestyle="--", label=f"max={CONFIG['max_steps_per_episode']}")
    ax.set_title("Survival (Episode Length)"); ax.set_xlabel("Episode"); ax.set_ylabel("Steps")
    ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")

    # 4 Collisions
    ax = axes[1, 0]
    colors = ["#3fb950" if c == 0 else "#f85149" for c in collisions]
    ax.bar(range(1, len(collisions)+1), collisions, color=colors, edgecolor="#30363d")
    ax.set_title("Collisions per Episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Collisions")

    # 5 Average speed
    ax = axes[1, 1]
    ax.plot(range(1, len(speeds)+1), speeds, marker="o", color="#d2a8ff",
            linewidth=2.5, markersize=7)
    ax.fill_between(range(1, len(speeds)+1), speeds, alpha=0.2, color="#d2a8ff")
    ax.set_title("Average Speed per Episode"); ax.set_xlabel("Episode"); ax.set_ylabel("Speed (km/h)")

    # 6 Success rate (collision-free)
    ax = axes[1, 2]
    success = [1 if c == 0 else 0 for c in collisions]
    rate = np.cumsum(success) / np.arange(1, len(success)+1)
    ax.plot(range(1, len(rate)+1), rate * 100, color="#ffa657", linewidth=2.5, marker="s", markersize=7)
    ax.set_ylim(0, 105)
    ax.set_title("Cumulative Success Rate"); ax.set_xlabel("Episode"); ax.set_ylabel("Success %")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"{run_name}_graphs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [GRAPH]  Graphs saved -> {path}")


def save_trajectory_map(env: MicroEnv, expert_traj: list,
                        ai_traj: list, run_name: str) -> None:
    frame = env.render_birdseye(
        show_trajectory=False,
        expert_traj=expert_traj,
        ai_traj=ai_traj,
    )
    # Legend
    cv2.rectangle(frame, (8, 8), (200, 68), (20, 20, 30), -1)
    cv2.putText(frame, "EXPERT", (24, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (100, 255, 150), 1, cv2.LINE_AA)
    cv2.line(frame, (8, 27), (20, 27), (100, 255, 150), 2, cv2.LINE_AA)
    cv2.putText(frame, "AI MODEL", (24, 55), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 100, 100), 1, cv2.LINE_AA)
    cv2.line(frame, (8, 52), (20, 52), (255, 100, 100), 2, cv2.LINE_AA)

    path = os.path.join(RESULTS_DIR, f"{run_name}_trajectory.png")
    cv2.imwrite(path, frame)
    print(f"  [MAP]  Trajectory map saved -> {path}")


def save_json_summary(losses: list, eval_data: dict, run_name: str) -> dict:
    summary = {
        "run_name":            run_name,
        "timestamp":           datetime.now().isoformat(),
        "config":              CONFIG,
        "training": {
            "initial_loss":    float(losses[0]) if losses else 0,
            "final_loss":      float(losses[-1]) if losses else 0,
            "loss_improvement_pct": float(
                100 * (losses[0] - losses[-1]) / (losses[0] + 1e-9)) if losses else 0,
        },
        "evaluation": {
            "episodes":          CONFIG["eval_episodes"],
            "mean_reward":       float(np.mean(eval_data["rewards"])),
            "mean_length":       float(np.mean(eval_data["lengths"])),
            "total_collisions":  int(sum(eval_data["collisions"])),
            "collision_rate":    float(np.mean([1 if c > 0 else 0 for c in eval_data["collisions"]])),
            "mean_speed_kmh":    float(np.mean(eval_data["speeds"])),
            "success_rate":      float(np.mean([1 if c == 0 else 0 for c in eval_data["collisions"]])),
        }
    }
    path = os.path.join(RESULTS_DIR, f"{run_name}_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [JSON]  JSON summary -> {path}")
    return summary


def print_final_report(summary: dict) -> None:
    ev = summary["evaluation"]
    tr = summary["training"]
    print("\n" + "="*60)
    print("  [DONE]  TRAINING COMPLETE - FINAL REPORT")
    print("="*60)
    print(f"  Loss reduction    : {tr['initial_loss']:.5f} -> {tr['final_loss']:.5f}"
          f"  ({tr['loss_improvement_pct']:.1f}% down)")
    print(f"  Mean reward       : {ev['mean_reward']:.2f}")
    print(f"  Survival (steps)  : {ev['mean_length']:.1f} / {CONFIG['max_steps_per_episode']}")
    print(f"  Mean speed        : {ev['mean_speed_kmh']:.1f} km/h")
    print(f"  Collision rate    : {ev['collision_rate']:.0%}")
    print(f"  Success rate      : {ev['success_rate']:.0%}")
    print("="*60 + "\n")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  MICRO-AUTODRIVE -- DL Self-Driving (No CARLA, No GPU req.)")
    print("="*60)

    run_name = f"micro_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    env = MicroEnv(seed=CONFIG["seed"])

    # ── Step 1: Expert demonstrations ────────────────────────────────────
    dataset = collect_expert_data(env, CONFIG["expert_episodes"])

    # Collect expert trajectory for comparison map
    env.reset(new_track=False)
    expert = ExpertDriver(env)
    for _ in range(CONFIG["max_steps_per_episode"]):
        action = expert.get_action()
        _, _, done, _ = env.step(action)
        if done:
            break
    expert_traj = env.get_trajectory()

    # ── Step 2: Train CNN ─────────────────────────────────────────────────
    model = DrivingCNN()
    losses = train(model, dataset)

    # Save final model
    model_path = os.path.join(MODEL_DIR, f"{run_name}_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"  [SAVED]  Model saved -> {model_path}")

    # ── Step 3: Evaluate & capture GIF ───────────────────────────────────
    eval_data = evaluate(model, env)

    # ── Step 4: Save all outputs ──────────────────────────────────────────
    print("\n" + "="*60)
    print("  SAVING OUTPUTS")
    print("="*60)

    save_gif(eval_data["gif_frames"], os.path.join(RESULTS_DIR, f"{run_name}_drive.gif"))
    save_training_graphs(losses, eval_data, run_name)
    save_trajectory_map(env, expert_traj, eval_data["ai_trajectory"], run_name)
    summary = save_json_summary(losses, eval_data, run_name)
    print_final_report(summary)

    print(f"  All results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
