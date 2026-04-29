"""
micro_env.py — Lightweight 2D Self-Driving Simulator
======================================================
A fully self-contained environment that simulates a car driving on a
procedurally generated winding track. No CARLA, no GPU, no simulator install
required. Runs purely on numpy + opencv.

Key components:
  - TrackGenerator  : Creates smooth looping tracks via cubic splines
  - CarPhysics      : Simple 2D kinematic bicycle model
  - MicroEnv        : Gym-like API (reset / step / render)
"""

import math
import random
import numpy as np
import cv2
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth_spline(points: np.ndarray, n_samples: int = 800) -> np.ndarray:
    """Resample a closed polyline into `n_samples` evenly-spaced points using
    a simple Catmull-Rom spline approach."""
    pts = np.vstack([points, points[0]])            # close the loop
    out = []
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]
        for t in np.linspace(0, 1, n_samples // len(points), endpoint=False):
            t2, t3 = t * t, t * t * t
            q = (
                0.5 * (
                    (2 * p1) +
                    (-p0 + p2) * t +
                    (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                    (-p0 + 3*p1 - 3*p2 + p3) * t3
                )
            )
            out.append(q)
    return np.array(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# Track Generator
# ---------------------------------------------------------------------------

class TrackGenerator:
    """Generates a random smooth looping race track."""

    def __init__(self, canvas_size: int = 600, track_width: int = 40, seed: int | None = None):
        self.canvas_size = canvas_size
        self.track_width = track_width
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.centerline: np.ndarray | None = None   # (N, 2) float32
        self.track_mask: np.ndarray | None = None   # (H, W) uint8

    def generate(self) -> None:
        """Generate a new random track."""
        cx, cy = self.canvas_size // 2, self.canvas_size // 2
        r_base = self.canvas_size * 0.32

        # Random control points on a roughly circular layout
        n_ctrl = self.rng.randint(6, 12)
        angles = np.linspace(0, 2 * math.pi, n_ctrl, endpoint=False)
        radii = r_base + self.np_rng.uniform(-r_base * 0.35, r_base * 0.35, n_ctrl)
        ctrl = np.stack([
            cx + radii * np.cos(angles),
            cy + radii * np.sin(angles)
        ], axis=1).astype(np.float32)

        self.centerline = _smooth_spline(ctrl, n_samples=800)

        # Build track mask
        mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        pts = self.centerline.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(mask, [pts], isClosed=True, color=255, thickness=self.track_width)
        self.track_mask = mask

    def is_on_track(self, x: float, y: float) -> bool:
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= self.canvas_size or yi < 0 or yi >= self.canvas_size:
            return False
        return self.track_mask[yi, xi] > 0

    def nearest_centerline_point(self, x: float, y: float) -> tuple[int, float]:
        """Return (index_of_nearest_point, distance)."""
        pos = np.array([x, y], dtype=np.float32)
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def centerline_tangent(self, idx: int) -> float:
        """Return heading angle (radians) of the centerline at index `idx`."""
        n = len(self.centerline)
        p1 = self.centerline[(idx + 1) % n]
        p0 = self.centerline[(idx - 1) % n]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        return math.atan2(dy, dx)


# ---------------------------------------------------------------------------
# Car Physics (Kinematic Bicycle Model)
# ---------------------------------------------------------------------------

class CarPhysics:
    """Simple 2D kinematic bicycle model."""

    MAX_SPEED = 5.0          # pixels per step
    ACCELERATION = 0.4
    BRAKE_DECEL = 0.8
    FRICTION = 0.05
    MAX_STEER = math.radians(30)   # max steering angle per step
    WHEELBASE = 12.0               # pixels

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0          # radians
        self.speed = 0.0

    def reset(self, x: float, y: float, heading: float) -> None:
        self.x, self.y, self.heading = x, y, heading
        self.speed = 0.0

    def step(self, steer: float, throttle: float, brake: float) -> None:
        """Apply one simulation step.
        steer    ∈ [-1, 1]   (negative = left)
        throttle ∈ [ 0, 1]
        brake    ∈ [ 0, 1]
        """
        steer = float(np.clip(steer, -1, 1))
        throttle = float(np.clip(throttle, 0, 1))
        brake = float(np.clip(brake, 0, 1))

        # Speed update
        self.speed += throttle * self.ACCELERATION
        self.speed -= brake * self.BRAKE_DECEL
        self.speed -= self.FRICTION
        self.speed = float(np.clip(self.speed, 0.0, self.MAX_SPEED))

        # Heading update (bicycle steering geometry)
        steer_angle = steer * self.MAX_STEER
        if self.speed > 0.01:
            self.heading += (self.speed / self.WHEELBASE) * math.tan(steer_angle)

        # Position update
        self.x += self.speed * math.cos(self.heading)
        self.y += self.speed * math.sin(self.heading)

    @property
    def speed_kmh(self) -> float:
        return self.speed * 20.0   # scale for display


# ---------------------------------------------------------------------------
# Rendering Constants
# ---------------------------------------------------------------------------

# Bird's-eye colour palette
PALETTE = {
    "sky":        (18, 18, 28),
    "grass":      (30, 60, 30),
    "track":      (60, 60, 65),
    "centerline": (220, 200, 50),
    "car":        (50, 200, 255),
    "car_outline":(20, 120, 200),
    "car_dir":    (255, 80, 80),
    "expert":     (100, 255, 150),
    "ai":         (255, 100, 100),
}

CAM_W, CAM_H = 96, 64   # dashcam resolution fed to CNN


# ---------------------------------------------------------------------------
# MicroEnv — the Gym-like Environment
# ---------------------------------------------------------------------------

class MicroEnv:
    """
    Lightweight 2D self-driving environment.

    Observation : (CAM_H, CAM_W, 3) uint8  — forward-facing dashcam view
    Action      : (steer, throttle, brake)  — floats
    """

    CANVAS = 600

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.track = TrackGenerator(canvas_size=self.CANVAS, track_width=44, seed=seed)
        self.car = CarPhysics()
        self.step_count = 0
        self.max_steps = 2000
        self.collision_history: list[int] = []
        self._trajectory: list[tuple[float, float]] = []
        self._last_idx = 0
        self._laps = 0
        self._prev_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, new_track: bool = False) -> np.ndarray:
        """Reset environment, optionally generating a new track."""
        if new_track or self.track.centerline is None:
            self.track.generate()
        # Spawn at a random point on the centerline
        start_idx = random.randint(0, len(self.track.centerline) - 1)
        sx, sy = self.track.centerline[start_idx]
        heading = self.track.centerline_tangent(start_idx)
        self.car.reset(sx, sy, heading)
        self.step_count = 0
        self.collision_history = []
        self._trajectory = [(sx, sy)]
        self._last_idx = start_idx
        self._prev_idx = start_idx
        self._laps = 0
        return self._get_obs()

    def step(self, action: tuple) -> tuple:
        """
        Returns: obs, reward, done, info
        """
        steer, throttle, brake = action
        self.car.step(steer, throttle, brake)
        self.step_count += 1
        self._trajectory.append((self.car.x, self.car.y))

        # Check on-track
        on_track = self.track.is_on_track(self.car.x, self.car.y)

        # Progress along centerline
        idx, dist_to_center = self.track.nearest_centerline_point(self.car.x, self.car.y)
        n = len(self.track.centerline)

        # Detect lap completion
        progress = (idx - self._prev_idx) % n
        if progress > n // 2:
            progress -= n
        self._prev_idx = idx

        # Reward shaping
        reward = 0.0
        if on_track:
            reward += self.car.speed * 0.5           # speed reward
            reward += max(0, 1.0 - dist_to_center / 22.0) * 0.3  # center bonus
        else:
            reward -= 50.0
            self.collision_history.append(self.step_count)

        # Done conditions
        done = (
            not on_track or
            self.step_count >= self.max_steps
        )

        info = {
            "speed_kmh": self.car.speed_kmh,
            "on_track": on_track,
            "dist_to_center": dist_to_center,
            "step": self.step_count,
            "collisions": len(self.collision_history),
        }

        return self._get_obs(), reward, done, info

    def render_birdseye(
        self,
        show_trajectory: bool = True,
        ai_traj: list | None = None,
        expert_traj: list | None = None,
    ) -> np.ndarray:
        """Render a full bird's-eye view canvas (BGR, uint8)."""
        canvas = np.full((self.CANVAS, self.CANVAS, 3), PALETTE["grass"], dtype=np.uint8)

        # Track surface
        track_layer = np.zeros_like(canvas)
        track_layer[self.track.track_mask > 0] = PALETTE["track"]
        canvas = cv2.addWeighted(canvas, 1.0, track_layer, 1.0, 0)

        # Dashed centerline
        cl = self.track.centerline.astype(np.int32)
        for i in range(0, len(cl), 12):
            p1 = tuple(cl[i])
            p2 = tuple(cl[min(i + 6, len(cl) - 1)])
            cv2.line(canvas, p1, p2, PALETTE["centerline"], 1, cv2.LINE_AA)

        # Expert trajectory
        if expert_traj and len(expert_traj) > 1:
            pts = np.array(expert_traj, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, PALETTE["expert"], 2, cv2.LINE_AA)

        # AI trajectory
        if ai_traj and len(ai_traj) > 1:
            pts = np.array(ai_traj, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, PALETTE["ai"], 2, cv2.LINE_AA)

        # Current car trajectory
        if show_trajectory and len(self._trajectory) > 1:
            pts = np.array(self._trajectory, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], False, (200, 180, 255), 1, cv2.LINE_AA)

        # Car body
        self._draw_car(canvas)

        return canvas

    def get_trajectory(self) -> list[tuple[float, float]]:
        return list(self._trajectory)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Simulate a forward-facing 'dashcam' view by cropping and warping a
        wedge in front of the car from the bird's-eye canvas.
        """
        bev = self.render_birdseye(show_trajectory=False)

        # Project a trapezoidal region in front of the car → rectangle
        car_x, car_y = self.car.x, self.car.y
        h = self.car.heading

        def ahead(forward, sideways):
            return (
                car_x + math.cos(h) * forward - math.sin(h) * sideways,
                car_y + math.sin(h) * forward + math.cos(h) * sideways,
            )

        src = np.array([
            ahead(8,  -14),
            ahead(8,   14),
            ahead(80, -40),
            ahead(80,  40),
        ], dtype=np.float32)

        dst = np.array([
            [0,          CAM_H],
            [CAM_W,      CAM_H],
            [0,          0],
            [CAM_W,      0],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        obs = cv2.warpPerspective(bev, M, (CAM_W, CAM_H))
        return obs   # (H, W, 3) uint8

    def _draw_car(self, canvas: np.ndarray) -> None:
        cx, cy = int(self.car.x), int(self.car.y)
        h = self.car.heading
        length, width = 14, 8

        def rot(dx, dy):
            return (
                int(cx + math.cos(h) * dx - math.sin(h) * dy),
                int(cy + math.sin(h) * dx + math.cos(h) * dy),
            )

        corners = [rot(-length//2, -width//2), rot(length//2, -width//2),
                   rot(length//2,  width//2),  rot(-length//2,  width//2)]
        pts = np.array(corners, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(canvas, [pts], PALETTE["car"])
        cv2.polylines(canvas, [pts], True, PALETTE["car_outline"], 1, cv2.LINE_AA)

        # Direction arrow
        front = rot(length // 2 + 4, 0)
        cv2.arrowedLine(canvas, (cx, cy), front, PALETTE["car_dir"], 2,
                        tipLength=0.5, line_type=cv2.LINE_AA)



# ---------------------------------------------------------------------------
# Expert Driver (mathematical path follower -- no ML needed)
# ---------------------------------------------------------------------------

class ExpertDriver:
    """
    Pure-pursuit path follower that acts as the ground-truth 'expert'.
    No machine learning — just geometry.
    """

    LOOKAHEAD = 30   # pixels ahead to aim for

    def __init__(self, env: MicroEnv):
        self.env = env

    def get_action(self) -> tuple[float, float, float]:
        car = self.env.car
        track = self.env.track
        n = len(track.centerline)

        # Current closest point
        idx, _ = track.nearest_centerline_point(car.x, car.y)

        # Find lookahead point (walk forward along centerline)
        dist = 0.0
        look_idx = idx
        while dist < self.LOOKAHEAD:
            next_idx = (look_idx + 1) % n
            dist += float(np.linalg.norm(
                track.centerline[next_idx] - track.centerline[look_idx]
            ))
            look_idx = next_idx

        target = track.centerline[look_idx]
        tx, ty = float(target[0]), float(target[1])

        # Compute heading error
        angle_to_target = math.atan2(ty - car.y, tx - car.x)
        heading_err = angle_to_target - car.heading

        # Normalise to [-pi, pi]
        while heading_err >  math.pi: heading_err -= 2 * math.pi
        while heading_err < -math.pi: heading_err += 2 * math.pi

        steer = float(np.clip(heading_err / math.radians(30), -1.0, 1.0))
        throttle = 0.7 - 0.4 * abs(steer)   # slow on sharp turns
        brake = 0.0

        return steer, throttle, brake
