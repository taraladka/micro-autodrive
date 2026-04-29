# Micro-AutoDrive

Micro-AutoDrive is a lightweight autonomous driving project designed to run on standard laptop hardware. It combines a procedural 2D simulation pipeline with a real-world steering prediction pipeline, making it useful for experimentation with behavioral cloning, temporal modeling, and model evaluation without depending on large simulator stacks or GPU-only infrastructure.

The repository is intentionally kept focused on source code and project metadata. Datasets, trained weights, checkpoints, cached outputs, and other generated artifacts are not versioned.

## Overview

The project is split into two main workflows:

- A synthetic driving pipeline built around a custom 2D environment, expert demonstration generation, and CNN-based imitation learning.
- A real-world steering pipeline built around dashcam frame preprocessing, class balancing, short temporal sequences, and a CNN+GRU architecture.

The overall aim is to provide a compact, understandable codebase for autonomous driving experiments that can be run and extended in a typical student or research environment.

## Repository Structure

```text
Micro-AutoDrive/
├── core/
│   └── model.py
├── v1_sim/
│   ├── analytics.py
│   ├── micro_env.py
│   ├── train.py
│   └── visualize.py
├── v2_real/
│   ├── data_loader.py
│   ├── train_real.py
│   └── visualize_real.py
├── CHANGELOG.md
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux
- 8 GB RAM minimum
- 16 GB RAM recommended for larger real-data preprocessing workflows

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running The Project

### Synthetic pipeline

```bash
python v1_sim/train.py
python v1_sim/analytics.py
python v1_sim/visualize.py
```

### Real-world pipeline

```bash
python v2_real/train_real.py
python v2_real/visualize_real.py
```

## Notes On Data

The repository does not include third-party datasets or locally generated training outputs. To run the real-world pipeline, place the required driving data locally in the path expected by `v2_real/data_loader.py`.

This keeps the repository compact and avoids mixing source control with large machine-specific artifacts.

## Design Notes

- `core/` contains the shared model definitions.
- `v1_sim/` contains the simulation environment and synthetic training workflow.
- `v2_real/` contains the real-road data pipeline and temporal steering model workflow.
- Runtime outputs such as models, checkpoints, plots, and cached data are intentionally excluded from version control.
