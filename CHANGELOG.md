# Micro-AutoDrive Changelog

## [v2.2.0] - Temporal Real-Driving Model
**Date:** 2026-04-28

### Added
- **Temporal Steering Model:** Replaced the real-data path's single-frame steering model with a CNN + GRU sequence model so inference can use short-term visual memory.
- **Sequence-Based Training:** `train_real.py` now builds contiguous frame windows, uses a small amount of target smoothing, and trains on sequence endpoints rather than isolated shuffled frames.
- **Richer Real-Data Visualizer:** `visualize_real.py` now shows temporal-model playback telemetry such as rolling error, playback FPS, motion proxy, model type, and sequence length.
- **Degree-Based Validation Metrics:** Real-data training now reports steering MAE, RMSE, signed bias, and hit rates within ±1°, ±2°, ±5°, and ±10°.
- **Detailed Analytics Guide:** `v1_sim/analytics.py` now generates a markdown guide explaining how to interpret both simulation charts and real-data steering-error charts.

### Changed
- **Honest Validation Split:** Real-data validation is now sequence-aware rather than random single-frame based, reducing leakage from heavily overlapping neighboring windows.
- **Model Saving Format:** Real-data checkpoints now save model metadata alongside weights so recurrent models can be restored correctly for inference.
- **Real-Run Analytics:** Mixed-run analytics now produce a dedicated real-data leaderboard and comparison chart focused on steering error quality rather than simulator reward metrics.

### Notes
- **Current Real-Data Scope:** The real-road path is currently a steering-only temporal model. The viewer can show richer telemetry, but true throttle, brake, and vehicle-speed targets are not available from the current Sully Chen training setup.
- **Error Expectations:** Degree-based metrics now make the steering target explicit, but a sustained `±1°` to `±2°` error band should be treated as an aggressive goal rather than an already-achieved guarantee.

## [v2.1.1] - Training Throughput & Stability Improvements
**Date:** 2026-04-28

### Added
- **Preprocessed Image Caching:** `RealDrivingDataset` can now cache cropped and resized road images in RAM, eliminating repeated OpenCV decode/crop/resize work on every epoch.

### Changed
- **Faster Real-Data Training Defaults:** `train_real.py` now disables the live viewer by default, reuses cached preprocessed images, and validates every 2 epochs instead of every epoch to reduce total wall-clock training time without increasing CPU load.
- **Laptop-Friendly Runtime Defaults:** Real-data training now uses a single shared cached dataset, defaults to `num_workers=0` when caching is enabled, and caps PyTorch CPU threads to a more conservative default to avoid excessive load on Windows laptops.
- **Lower-Overhead Optimizer Step:** Switched the training loop to `optimizer.zero_grad(set_to_none=True)` to trim per-step bookkeeping overhead.

### Fixed
- **Validation Split Integrity:** Training and validation now use separate dataset views so augmentation remains enabled only for training and disabled for validation.
- **Analytics Compatibility:** `v1_sim/analytics.py` now handles mixed simulation and real-data summaries in `results/` without crashing, and no longer depends on emoji-capable Windows console output.

## [v2.1.0] - Architecture Refactoring & ML Optimizations
**Date:** 2026-04-28

### Added
- **Live Training UX:** Implemented a live `cv2` visualizer inside the `train_real.py` loop to observe AI predictions vs Ground Truth in real-time, plus a `tqdm` progress bar for accurate ETA monitoring.
- **Graceful Interrupts:** Added a `KeyboardInterrupt` exception handler in `train_real.py` allowing users to stop training early (`Ctrl+C`) and safely save their current model weights without losing progress.
- **Dataset Balancing:** Modified `data_loader.py` to randomly drop 80% of frames where the steering angle is near zero. This prevents the "Mode Collapse" (predicting the average) issue common in the Sully Chen dataset and forces the network to learn cornering.
- **Region of Interest Cropping:** Added image cropping (`img[100:200, :]`) to both the training loader and visualizer to strip out useless sky and hood pixels, massively improving model convergence.

### Changed
- **Domain Separation Architecture:** Cleanly restructured the repository into a professional layout:
  - Extracted the shared neural network into `core/model.py`.
  - Moved all synthetic simulation scripts into `v1_sim/`.
  - Moved all real-world dashcam scripts into `v2_real/`.
## [v2.0.0] - Real-World Dataset Integration
**Date:** 2026-04-28

### Added
- **Real-World Training Pipeline:** Integrated support for training on the Sully Chen dashcam dataset (45,500 frames of real-world driving).
- **`data_loader.py`:** PyTorch Dataset class (`RealDrivingDataset`) that handles loading images via OpenCV, resizing to 96x64, and mathematically normalising steering angles ([-540, 540] to [-1, 1]).
- **`train_real.py`:** A dedicated training script for the real dataset. It splits data into 80% train / 20% validation and trains the `DrivingCNN` on the steering output using MSE loss, ignoring throttle and brake (as real dataset lacks these).
- **`visualize_real.py`:** A new visual inference tool that plays back the Sully Chen dashcam footage and overlays both the Ground Truth human steering angle and the AI's predicted steering angle in real-time.

### Changed
- **Sim-to-Real Architecture Strategy:** Maintained the exact same `DrivingCNN` model architecture (`micro_env.py`) but added the parallel real-world pipeline to prove that the architecture can transfer to actual road data without modifications.

## [v1.0.0] - Initial Synthetic 2D Simulator
**Date:** 2026-04-28

### Added
- **Synthetic MicroEnv Simulator:** A complete Gym-like 2D kinematic bicycle model environment (`micro_env.py`).
- **Imitation Learning Pipeline:** `train.py` script to generate expert trajectories and train the `DrivingCNN`.
- **Live Visualization Tool:** `visualize.py` utilizing OpenCV for live monitoring of AI inference and HUD telemetry.
- **Model:** `DrivingCNN`, a lightweight 350K parameter model capable of fast CPU training.
