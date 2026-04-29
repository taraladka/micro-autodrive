"""
analytics.py - Micro-AutoDrive analysis tool
============================================
Run with: python v1_sim/analytics.py

Loads saved run summaries from results/ and generates:
  - Simulation leaderboard + charts for synthetic runs
  - Real-data error analysis for temporal steering runs
  - A plain-language guide explaining how to read the outputs
"""

import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sns.set_style("darkgrid")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DARK_BG = "#0d1117"
PANEL_BG = "#161b22"
TEXT = "#e6edf3"
GRID = "#30363d"
COLORS = ["#58a6ff", "#3fb950", "#ffa657", "#f78166", "#d2a8ff", "#79c0ff"]


def _apply_dark(ax):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)


def _is_real_run(run: dict) -> bool:
    return run.get("dataset") == "Sully Chen"


def _is_sim_run(run: dict) -> bool:
    return not _is_real_run(run) and "evaluation" in run and "training" in run


def load_runs() -> list[dict]:
    runs = []
    if not os.path.isdir(RESULTS_DIR):
        print("[ERROR] No results/ directory found. Run a training pipeline first.")
        return runs
    for fn in sorted(os.listdir(RESULTS_DIR)):
        if fn.endswith("_summary.json"):
            path = os.path.join(RESULTS_DIR, fn)
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
    print(f"[LOAD] Loaded {len(runs)} run(s) from {RESULTS_DIR}")
    return runs


def generate_sim_leaderboard(sim_runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in sim_runs:
        ev = r["evaluation"]
        tr = r["training"]
        rows.append({
            "Run": r["run_name"],
            "Mean Reward": round(ev["mean_reward"], 2),
            "Survival (steps)": round(ev["mean_length"], 1),
            "Success Rate": f"{ev['success_rate']:.0%}",
            "Collision Rate": f"{ev['collision_rate']:.0%}",
            "Mean Speed": f"{ev['mean_speed_kmh']:.1f} km/h",
            "Final Loss": f"{tr['final_loss']:.5f}",
            "Loss Down %": f"{tr['loss_improvement_pct']:.1f}%",
        })

    df = pd.DataFrame(rows).sort_values("Mean Reward", ascending=False)
    df.insert(0, "Rank", range(1, len(df) + 1))

    print("\n" + "=" * 90)
    print("  MICRO-AUTODRIVE SIMULATION LEADERBOARD")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")

    csv_path = os.path.join(RESULTS_DIR, f"leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Simulation leaderboard CSV -> {csv_path}")
    return df


def compare_sim_runs(sim_runs: list[dict]) -> None:
    if not sim_runs:
        print("[WARN] No simulation runs available for comparison charts.")
        return

    names = [r["run_name"][-18:] for r in sim_runs]
    mean_rewards = [r["evaluation"]["mean_reward"] for r in sim_runs]
    success_rates = [r["evaluation"]["success_rate"] for r in sim_runs]
    mean_speeds = [r["evaluation"]["mean_speed_kmh"] for r in sim_runs]
    final_losses = [r["training"]["final_loss"] for r in sim_runs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Micro-AutoDrive - Simulation Run Comparison", color=TEXT, fontsize=15)

    datasets = [
        (axes[0, 0], mean_rewards, "Mean Reward", COLORS[0]),
        (axes[0, 1], success_rates, "Success Rate", COLORS[1]),
        (axes[1, 0], mean_speeds, "Mean Speed (km/h)", COLORS[2]),
        (axes[1, 1], final_losses, "Final Training Loss", COLORS[3]),
    ]

    for ax, values, title, color in datasets:
        _apply_dark(ax)
        bars = ax.barh(names, values, color=color, alpha=0.85, edgecolor=GRID)
        ax.set_title(title)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", color=TEXT, fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[SAVE] Simulation comparison chart -> {path}")


def plot_sim_progression(sim_runs: list[dict]) -> None:
    if len(sim_runs) < 2:
        print("[WARN] Need at least 2 simulation runs for progression chart.")
        return

    sorted_runs = sorted(sim_runs, key=lambda r: r["timestamp"])
    x = range(len(sorted_runs))
    labels = [r["run_name"][-12:] for r in sorted_runs]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Simulation Progression Over Time", color=TEXT, fontsize=14)

    def _line(ax, values, ylabel, color):
        _apply_dark(ax)
        ax.plot(x, values, marker="o", lw=2.5, color=color, markersize=8)
        ax.fill_between(x, values, alpha=0.15, color=color)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)

    _line(axes[0], [r["evaluation"]["mean_reward"] for r in sorted_runs], "Mean Reward", COLORS[0])
    axes[0].set_title("Reward Progression")

    _line(axes[1], [r["evaluation"]["success_rate"] for r in sorted_runs], "Success Rate", COLORS[1])
    axes[1].set_title("Safety Progression")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"progression_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[SAVE] Simulation progression chart -> {path}")


def generate_real_leaderboard(real_runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in real_runs:
        ev = r.get("evaluation", {})
        tr = r["training"]
        model = r.get("model", {})
        rows.append({
            "Run": r["run_name"],
            "Model": model.get("type", "unknown"),
            "Seq Len": model.get("sequence_length", 1),
            "MAE (deg)": round(ev.get("mae_deg", np.nan), 2),
            "RMSE (deg)": round(ev.get("rmse_deg", np.nan), 2),
            "Bias (deg)": round(ev.get("bias_deg", np.nan), 2),
            "±1 deg": f"{ev.get('within_1deg', 0.0):.0%}",
            "±2 deg": f"{ev.get('within_2deg', 0.0):.0%}",
            "±5 deg": f"{ev.get('within_5deg', 0.0):.0%}",
            "Final Val Loss": f"{tr['final_val_loss']:.5f}",
        })

    df = pd.DataFrame(rows).sort_values(["MAE (deg)", "RMSE (deg)"], ascending=[True, True])
    df.insert(0, "Rank", range(1, len(df) + 1))

    print("\n" + "=" * 110)
    print("  MICRO-AUTODRIVE REAL-DATA LEADERBOARD")
    print("=" * 110)
    print(df.to_string(index=False))
    print("=" * 110 + "\n")

    csv_path = os.path.join(RESULTS_DIR, f"real_leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[SAVE] Real-data leaderboard CSV -> {csv_path}")
    return df


def compare_real_runs(real_runs: list[dict]) -> None:
    if not real_runs:
        print("[WARN] No real-data runs available for comparison charts.")
        return

    names = [r["run_name"][-18:] for r in real_runs]
    mae = [r.get("evaluation", {}).get("mae_deg", np.nan) for r in real_runs]
    rmse = [r.get("evaluation", {}).get("rmse_deg", np.nan) for r in real_runs]
    within2 = [100 * r.get("evaluation", {}).get("within_2deg", 0.0) for r in real_runs]
    bias = [r.get("evaluation", {}).get("bias_deg", 0.0) for r in real_runs]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle("Micro-AutoDrive - Real Data Error Analysis", color=TEXT, fontsize=15)

    plots = [
        (axes[0, 0], mae, "Validation MAE (deg)", COLORS[0]),
        (axes[0, 1], rmse, "Validation RMSE (deg)", COLORS[3]),
        (axes[1, 0], within2, "Within ±2 deg (%)", COLORS[1]),
        (axes[1, 1], bias, "Signed Bias (deg)", COLORS[2]),
    ]

    for ax, values, title, color in plots:
        _apply_dark(ax)
        bars = ax.barh(names, values, color=color, alpha=0.85, edgecolor=GRID)
        ax.set_title(title)
        if "Bias" in title:
            ax.axvline(0.0, color="#f78166", linestyle="--", linewidth=1.5)
        if "MAE" in title or "RMSE" in title:
            ax.axvline(2.0, color="#ffa657", linestyle="--", linewidth=1.5, label="2 deg target")
            ax.legend(facecolor="#21262d", labelcolor="#c9d1d9")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + (0.5 if val >= 0 else -0.5), bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", color=TEXT, fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"real_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"[SAVE] Real-data comparison chart -> {path}")


def write_reading_guide(sim_runs: list[dict], real_runs: list[dict]) -> None:
    lines = [
        "# Micro-AutoDrive Analytics Guide",
        "",
        "## How To Read Simulation Metrics",
        "- Mean Reward: higher is better. It blends forward progress, staying on track, and speed.",
        "- Survival (steps): closer to the max episode length means the car stays alive longer.",
        "- Success Rate: percentage of collision-free episodes. Higher is better.",
        "- Collision Rate: lower is better.",
        "- Mean Speed: useful only together with collision metrics. High speed with high collision rate is not good.",
        "- Final Loss: lower means the policy matches the expert better, but it is not the final driving metric.",
        "",
        "## How To Read Real-Data Metrics",
        "- MAE (deg): average absolute steering error in degrees. Lower is better.",
        "- RMSE (deg): like MAE, but punishes large misses more strongly. Lower is better.",
        "- Bias (deg): signed average error. Positive means the model tends to steer too far right. Negative means too far left.",
        "- Within ±1 deg / ±2 deg / ±5 deg: share of samples that land inside each tolerance band.",
        "- Straight / Left-turn / Right-turn MAE: shows whether the model is mostly failing on curves or also drifting on straight roads.",
        "",
        "## Practical Targets",
        "- A stable road model should push bias close to 0 deg.",
        "- Within ±2 deg should trend upward as the model improves.",
        "- Reaching average error consistently inside ±1 to ±2 deg on this dataset is ambitious; treat it as a stretch goal, not a guaranteed baseline.",
        "",
        "## What To Trust Most",
        "- For simulation runs: prioritize success rate, collision rate, and survival before raw speed.",
        "- For real-data runs: prioritize MAE, RMSE, bias, and the ±2 deg hit rate over raw training loss.",
        "",
        f"Simulation runs analyzed: {len(sim_runs)}",
        f"Real-data runs analyzed: {len(real_runs)}",
    ]
    path = os.path.join(RESULTS_DIR, f"analytics_guide_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[SAVE] Analytics reading guide -> {path}")


def detailed_report(sim_runs: list[dict], real_runs: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  DETAILED REPORT")
    print("=" * 80)

    if sim_runs:
        print("\n  Simulation runs")
        for i, r in enumerate(sim_runs, 1):
            ev, tr = r["evaluation"], r["training"]
            print(f"\n  Run {i}: {r['run_name']}")
            print(f"  {'-' * 60}")
            print(f"    Timestamp       : {r['timestamp']}")
            print(f"    Initial loss    : {tr['initial_loss']:.5f}")
            print(f"    Final loss      : {tr['final_loss']:.5f}")
            print(f"    Improvement     : {tr['loss_improvement_pct']:.1f}%")
            print(f"    Mean reward     : {ev['mean_reward']:.2f}")
            print(f"    Mean survival   : {ev['mean_length']:.1f} steps")
            print(f"    Mean speed      : {ev['mean_speed_kmh']:.1f} km/h")
            print(f"    Collision rate  : {ev['collision_rate']:.0%}")
            print(f"    Success rate    : {ev['success_rate']:.0%}")

    if real_runs:
        print("\n  Real-data runs")
        for i, r in enumerate(real_runs, 1):
            tr = r["training"]
            ev = r.get("evaluation", {})
            model = r.get("model", {})
            print(f"\n  Run {i}: {r['run_name']}")
            print(f"  {'-' * 60}")
            print(f"    Timestamp       : {r['timestamp']}")
            print(f"    Dataset         : {r.get('dataset', 'unknown')}")
            print(f"    Model           : {model.get('type', 'unknown')}")
            print(f"    Sequence length : {model.get('sequence_length', 1)}")
            print(f"    Epochs          : {tr.get('epochs_completed', 'n/a')}")
            print(f"    Train loss      : {tr['initial_train_loss']:.5f} -> {tr['final_train_loss']:.5f}")
            print(f"    Val loss        : {tr['initial_val_loss']:.5f} -> {tr['final_val_loss']:.5f}")
            if ev:
                print(f"    MAE             : {ev.get('mae_deg', np.nan):.2f} deg")
                print(f"    RMSE            : {ev.get('rmse_deg', np.nan):.2f} deg")
                print(f"    Bias            : {ev.get('bias_deg', np.nan):+.2f} deg")
                print(f"    Within ±1 deg   : {ev.get('within_1deg', 0.0):.0%}")
                print(f"    Within ±2 deg   : {ev.get('within_2deg', 0.0):.0%}")
                print(f"    Within ±5 deg   : {ev.get('within_5deg', 0.0):.0%}")
                print(f"    Straight MAE    : {ev.get('straight_mae_deg', np.nan)}")
                print(f"    Left-turn MAE   : {ev.get('left_turn_mae_deg', np.nan)}")
                print(f"    Right-turn MAE  : {ev.get('right_turn_mae_deg', np.nan)}")

    print("=" * 80 + "\n")


def main():
    print("\n" + "=" * 70)
    print("  MICRO-AUTODRIVE ANALYSIS TOOL")
    print("=" * 70 + "\n")

    runs = load_runs()
    if not runs:
        return

    sim_runs = [run for run in runs if _is_sim_run(run)]
    real_runs = [run for run in runs if _is_real_run(run)]
    skipped_runs = len(runs) - len(sim_runs) - len(real_runs)

    print(f"[INFO] Simulation runs: {len(sim_runs)} | Real-data runs: {len(real_runs)} | Skipped: {skipped_runs}")
    detailed_report(sim_runs, real_runs)

    if sim_runs:
        generate_sim_leaderboard(sim_runs)
        compare_sim_runs(sim_runs)
        plot_sim_progression(sim_runs)
    else:
        print("[WARN] No simulation summaries found, skipping simulation charts.")

    if real_runs:
        generate_real_leaderboard(real_runs)
        compare_real_runs(real_runs)
    else:
        print("[WARN] No real-data summaries found, skipping real-data charts.")

    write_reading_guide(sim_runs, real_runs)
    print("\n[OK] Analysis complete. Check results/ for charts and the reading guide.\n")


if __name__ == "__main__":
    main()
