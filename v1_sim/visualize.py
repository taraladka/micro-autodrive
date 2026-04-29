"""
visualize.py -- Live Real-Time Visualization for Micro-AutoDrive
================================================================
Run with: python v1_sim/visualize.py

Opens a live OpenCV window showing:
  - LEFT  : Full bird's-eye view of the track with the car moving on it
  - RIGHT : What the AI "sees" (dashcam / forward-facing view)
  - HUD   : Speed, step count, episode number, reward, mode label

Controls:
  Q / ESC  : Quit
  E        : Switch to Expert driver mode
  A        : Switch to AI model mode
  R        : Reset (new episode on same track)
  N        : New random track + reset
  +/-      : Speed up / slow down playback

Usage:
  python v1_sim/visualize.py                    # auto-loads latest model
  python v1_sim/visualize.py --model models/my_model.pth
  python v1_sim/visualize.py --mode expert      # watch the expert only
  python v1_sim/visualize.py --episodes 5       # run 5 episodes then exit
"""

import sys, os, argparse, glob, time
import numpy as np
import cv2
import torch

# ── resolve project root so we can import micro_env from anywhere ──────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from v1_sim.micro_env import MicroEnv, ExpertDriver   # type: ignore
from core.model import DrivingCNN, CAM_W, CAM_H

# ── Layout constants ────────────────────────────────────────────────────────
BEV_SIZE   = 600          # bird's-eye canvas size (must match MicroEnv.CANVAS)
CAM_SCALE  = 4            # upscale dashcam for display
CAM_DISP_W = CAM_W * CAM_SCALE   # 384
CAM_DISP_H = CAM_H * CAM_SCALE   # 256

PANEL_W    = 300          # right info-panel width
WIN_W      = BEV_SIZE + CAM_DISP_W + PANEL_W
WIN_H      = BEV_SIZE

# Colour palette (BGR)
C = {
    "bg":       (18,  18,  28),
    "accent":   (255, 170,  50),
    "green":    ( 60, 220,  80),
    "red":      ( 80,  80, 245),
    "blue":     (240, 150,  50),
    "white":    (230, 230, 230),
    "dim":      (120, 120, 120),
    "panel":    ( 28,  28,  44),
    "bar_bg":   ( 50,  50,  70),
    "bar_fg":   ( 80, 200, 100),
    "bar_bad":  ( 80,  80, 230),
}


# ── Helpers ─────────────────────────────────────────────────────────────────

def _text(img, txt, pos, scale=0.5, color=C["white"], bold=False):
    thick = 2 if bold else 1
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, max_val, color):
    """Draw a horizontal progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), C["bar_bg"], -1)
    fill = int(w * min(value, max_val) / max(max_val, 1e-9))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), C["dim"], 1)


def find_latest_model() -> str | None:
    model_dir = os.path.join(BASE_DIR, "models")
    pattern   = os.path.join(model_dir, "*_final.pth")
    files     = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_model(path: str) -> "DrivingCNN | None":
    try:
        model = DrivingCNN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        print(f"[OK]  Model loaded: {path}")
        return model
    except Exception as e:
        print(f"[WARN] Could not load model: {e}")
        return None


# ── Info Panel ───────────────────────────────────────────────────────────────

def draw_panel(panel: np.ndarray, state: dict) -> None:
    """Draw the right-side info panel."""
    panel[:] = C["panel"]
    W = panel.shape[1]
    y = 20

    # Title
    _text(panel, "MICRO-AUTODRIVE", (10, y), 0.55, C["accent"], bold=True)
    y += 28

    # Mode badge
    mode_label = "  AI MODEL  " if state["mode"] == "ai" else "  EXPERT   "
    mode_color = C["blue"] if state["mode"] == "ai" else C["green"]
    cv2.rectangle(panel, (10, y), (W - 10, y + 22), mode_color, -1)
    _text(panel, mode_label, (18, y + 16), 0.5, (255, 255, 255), bold=True)
    y += 36

    # Separator
    cv2.line(panel, (10, y), (W - 10, y), C["dim"], 1)
    y += 14

    # Episode / step
    _text(panel, f"Episode  {state['episode']:>3}", (10, y), 0.48, C["white"])
    y += 22
    _text(panel, f"Step     {state['step']:>4} / {state['max_steps']}", (10, y), 0.48, C["white"])
    y += 22
    _text(panel, f"Reward   {state['reward']:>8.1f}", (10, y), 0.48, C["white"])
    y += 30

    # Speed bar
    speed = state["speed"]
    _text(panel, f"Speed  {speed:.1f} km/h", (10, y), 0.47, C["white"])
    y += 18
    _bar(panel, 10, y, W - 20, 12, speed, 120.0,
         C["bar_fg"] if speed < 90 else C["bar_bad"])
    y += 26

    # Separator
    cv2.line(panel, (10, y), (W - 10, y), C["dim"], 1)
    y += 14

    # Collisions
    col_color = C["green"] if state["collisions"] == 0 else C["red"]
    _text(panel, f"Collisions: {state['collisions']}", (10, y), 0.48, col_color)
    y += 22

    # On-track indicator
    ot_color = C["green"] if state["on_track"] else C["red"]
    ot_label = "ON TRACK" if state["on_track"] else "OFF TRACK!"
    _text(panel, ot_label, (10, y), 0.5, ot_color, bold=True)
    y += 30

    # Separator
    cv2.line(panel, (10, y), (W - 10, y), C["dim"], 1)
    y += 14

    # Controls legend
    _text(panel, "CONTROLS", (10, y), 0.45, C["accent"])
    y += 20
    controls = [
        ("A", "AI driver mode"),
        ("E", "Expert mode"),
        ("R", "Reset episode"),
        ("N", "New track"),
        ("+", "Speed up"),
        ("-", "Slow down"),
        ("Q / ESC", "Quit"),
    ]
    for key, desc in controls:
        _text(panel, f"  {key:<8} {desc}", (10, y), 0.38, C["dim"])
        y += 16

    # FPS counter
    y = panel.shape[0] - 20
    _text(panel, f"FPS: {state['fps']:.1f}", (10, y), 0.4, C["dim"])


# ── Dashcam Panel ────────────────────────────────────────────────────────────

def draw_dashcam_panel(obs: np.ndarray) -> np.ndarray:
    """Return a panel showing the upscaled dashcam + label."""
    upscaled = cv2.resize(obs, (CAM_DISP_W, CAM_DISP_H),
                          interpolation=cv2.INTER_NEAREST)
    # Add a bottom label bar
    label_h = 30
    canvas = np.zeros((CAM_DISP_H + label_h, CAM_DISP_W, 3), dtype=np.uint8)
    canvas[:CAM_DISP_H] = upscaled
    canvas[CAM_DISP_H:] = C["panel"]
    _text(canvas, "AI DASHCAM VIEW  (what the car sees)",
          (8, CAM_DISP_H + 20), 0.42, C["accent"])
    # Border
    cv2.rectangle(canvas, (0, 0), (CAM_DISP_W - 1, CAM_DISP_H - 1),
                  C["accent"], 2)
    return canvas


# ── Main Visualizer ──────────────────────────────────────────────────────────

def run(args):
    # ── Load model ──────────────────────────────────────────────────────────
    model_path = args.model or find_latest_model()
    model = None

    if model_path:
        model = load_model(model_path)
    if model is None:
        print("[INFO] No model found — forcing Expert mode.")
        args.mode = "expert"

    mode = args.mode   # "ai" or "expert"

    # ── Environment ─────────────────────────────────────────────────────────
    env    = MicroEnv(seed=42)
    expert = ExpertDriver(env)

    obs    = env.reset(new_track=True)
    done   = False

    # ── State ────────────────────────────────────────────────────────────────
    episode    = 1
    step       = 0
    ep_reward  = 0.0
    delay_ms   = 16          # ~60 FPS target
    fps        = 0.0
    t_prev     = time.time()
    max_steps  = 2000

    # ── Window ───────────────────────────────────────────────────────────────
    WIN_NAME = "Micro-AutoDrive  |  Q to quit"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    print("\n[LIVE] Window opened. Press Q or ESC to quit.\n")

    ep_limit = args.episodes if args.episodes > 0 else 999999

    while True:
        # ── Get action ───────────────────────────────────────────────────────
        if mode == "ai" and model is not None:
            t = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                steer, throttle, brake = model(t)
            action = (steer.item(), throttle.item(), brake.item())
        else:
            action = expert.get_action()

        obs, reward, done, info = env.step(action)
        step       += 1
        ep_reward  += reward

        # ── Render ────────────────────────────────────────────────────────────
        bev = env.render_birdseye(show_trajectory=True)   # (600, 600, 3)

        # FPS
        t_now = time.time()
        fps   = 0.9 * fps + 0.1 / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        # Build right panel
        panel_img = np.zeros((WIN_H, PANEL_W, 3), dtype=np.uint8)
        state = dict(
            mode=mode, episode=episode, step=step, max_steps=max_steps,
            reward=ep_reward, speed=info["speed_kmh"],
            collisions=info["collisions"], on_track=info["on_track"],
            fps=fps,
        )
        draw_panel(panel_img, state)

        # Build dashcam panel (below BEV height — pad vertically)
        dashcam_img = draw_dashcam_panel(obs)
        # Pad to full WIN_H
        dashcam_col = np.zeros((WIN_H, CAM_DISP_W, 3), dtype=np.uint8)
        dashcam_col[:] = C["bg"]
        # Place dashcam centered vertically
        y_off = (WIN_H - dashcam_img.shape[0]) // 2
        dashcam_col[y_off: y_off + dashcam_img.shape[0]] = dashcam_img

        # Compose final frame: [BEV | DASHCAM | PANEL]
        frame = np.concatenate([bev, dashcam_col, panel_img], axis=1)

        cv2.imshow(WIN_NAME, frame)

        # ── Episode reset ─────────────────────────────────────────────────────
        if done or step >= max_steps:
            print(f"  Episode {episode:>3} done | steps={step} | "
                  f"reward={ep_reward:.1f} | collisions={info['collisions']}")
            episode   += 1
            step       = 0
            ep_reward  = 0.0
            if episode > ep_limit:
                break
            obs  = env.reset(new_track=False)
            done = False

        # ── Keyboard ──────────────────────────────────────────────────────────
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (ord('q'), ord('Q'), 27):   # Q or ESC
            break
        elif key in (ord('a'), ord('A')):
            mode = "ai"
            print("[MODE] Switched to AI driver")
        elif key in (ord('e'), ord('E')):
            mode = "expert"
            print("[MODE] Switched to Expert driver")
        elif key in (ord('r'), ord('R')):
            obs  = env.reset(new_track=False)
            step = 0; ep_reward = 0.0; done = False
            print("[RESET] Same track, new episode")
        elif key in (ord('n'), ord('N')):
            obs  = env.reset(new_track=True)
            step = 0; ep_reward = 0.0; done = False
            print("[RESET] New random track")
        elif key in (ord('+'), ord('=')):
            delay_ms = max(1, delay_ms - 5)
            print(f"[SPEED] delay={delay_ms}ms")
        elif key in (ord('-'), ord('_')):
            delay_ms = min(200, delay_ms + 5)
            print(f"[SPEED] delay={delay_ms}ms")

    cv2.destroyAllWindows()
    print("\n[DONE] Visualizer closed.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live visualizer for Micro-AutoDrive")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to .pth model file (default: latest in models/)")
    parser.add_argument("--mode",     type=str, default="ai",
                        choices=["ai", "expert"],
                        help="Start in AI or Expert mode (default: ai)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Stop after N episodes (0 = run forever)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
