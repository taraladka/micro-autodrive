"""
core/model.py
=============
Shared model definitions for both synthetic and real-world pipelines.
"""

import numpy as np
import torch
import torch.nn as nn


CAM_W, CAM_H = 96, 64


class DrivingEncoder(nn.Module):
    """Shared visual encoder for both single-frame and temporal models."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, CAM_H, CAM_W)
            flat = int(np.prod(self.conv(dummy).shape[1:]))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.feature_dim = 32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        return self.fc(self.conv(x))


class DrivingCNN(nn.Module):
    """
    Lightweight CNN: raw dashcam pixels (uint8 CHW) -> (steer, throttle, brake).
    Input is normalised to [0, 1] internally.
    """

    def __init__(self):
        super().__init__()
        self.encoder = DrivingEncoder()
        self.steer_head = nn.Linear(self.encoder.feature_dim, 1)
        self.throttle_head = nn.Linear(self.encoder.feature_dim, 1)
        self.brake_head = nn.Linear(self.encoder.feature_dim, 1)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        return (
            torch.tanh(self.steer_head(feat)),
            torch.sigmoid(self.throttle_head(feat)),
            torch.sigmoid(self.brake_head(feat)),
        )


class TemporalDrivingGRU(nn.Module):
    """CNN encoder plus a lightweight GRU for temporally-smoothed steering."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 1):
        super().__init__()
        self.encoder = DrivingEncoder()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        batch, steps, channels, height, width = x.shape
        feat = self.encoder(x.reshape(batch * steps, channels, height, width))
        feat = feat.view(batch, steps, -1)
        out, hidden = self.temporal(feat, hidden)
        steer = torch.tanh(self.head(out[:, -1]))
        return steer, hidden

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def predict_step(self, frame: torch.Tensor, hidden: torch.Tensor | None = None):
        steer, hidden = self.forward(frame, hidden)
        return steer, hidden


def save_model_bundle(path: str, model: nn.Module, model_type: str, config: dict | None = None):
    payload = {
        "model_type": model_type,
        "state_dict": model.state_dict(),
        "config": config or {},
    }
    torch.save(payload, path)


def load_model_bundle(path: str, map_location: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if isinstance(payload, dict) and "state_dict" in payload and "model_type" in payload:
        model_type = payload["model_type"]
        config = payload.get("config", {})
        if model_type == "temporal_gru":
            model = TemporalDrivingGRU(
                hidden_size=config.get("hidden_size", 64),
                num_layers=config.get("num_layers", 1),
            )
        else:
            model = DrivingCNN()
        model.load_state_dict(payload["state_dict"])
        return model, payload

    model = DrivingCNN()
    model.load_state_dict(payload)
    return model, {"model_type": "cnn", "config": {}}
