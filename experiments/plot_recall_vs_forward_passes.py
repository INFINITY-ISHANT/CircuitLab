"""Create a source-backed IOI recall-versus-experimental-cost benchmark plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib

# Rendering is a file-producing batch task; avoid any platform GUI/Tk backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
PLOTTED_COLUMNS = ["strategy", "seed", "forward_passes", "recall", "source", "source_path"]
DISPLAY_NAMES = {"circuitlab": "CircuitLab", "exhaustive": "Exhaustive patching", "acdc": "ACDC"}
COLORS = {"circuitlab": "#5B5BD6", "exhaustive": "#E76F51", "acdc": "#2A9D8F"}


def _normalise_strategy(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.lower().replace("_", "").replace("-", "").replace(" ", "")
    return {"circuitlab": "circuitlab", "exhaustive": "exhaustive", "acdc": "acdc"}.get(key)


def _trajectory_points(results_dir: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for json_path in results_dir.glob("trajectories/**/*_recovery_trajectory.json"):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        strategy = _normalise_strategy(payload.get("strategy"))
        seed = payload.get("seed")
        if strategy is None or isinstance(seed, bool) or not isinstance(seed, int):
            # A trajectory without an explicit seed cannot be used to represent
            # a multi-seed benchmark point.
            continue
        for row in payload.get("trajectory", []):
            if not isinstance(row, dict):
                continue
            passes, recall = row.get("forward_passes"), row.get("recall")
            if isinstance(passes, int) and isinstance(recall, (int, float)):
                points.append({"strategy": strategy, "seed": seed, "forward_passes": passes, "recall": float(recall), "source": "trajectory", "source_path": str(json_path)})
    return points


def _benchmark_points(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    frame = pd.read_csv(raw_path)
    required = {"strategy", "seed", "forward_passes", "recall", "status"}
    if not required <= set(frame.columns):
        raise ValueError(f"{raw_path} is missing columns {sorted(required - set(frame.columns))}.")
    points: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        strategy = _normalise_strategy(row["strategy"])
        if strategy is None or row["status"] != "complete" or pd.isna(row["forward_passes"]) or pd.isna(row["recall"]):
            continue
        points.append({"strategy": strategy, "seed": int(row["seed"]), "forward_passes": float(row["forward_passes"]), "recall": float(row["recall"]), "source": "benchmark_raw", "source_path": str(raw_path)})
    return points


def collect_plot_data(results_dir: Path | str = RESULTS_DIRECTORY) -> pd.DataFrame:
    """Load only completed trajectory/benchmark records; never fill missing curves."""
    directory = Path(results_dir)
    points = _trajectory_points(directory) + _benchmark_points(directory / "benchmark_raw.csv")
    frame = pd.DataFrame(points, columns=PLOTTED_COLUMNS)
    if frame.empty:
        return frame
    # Prefer trajectory points when a strategy/seed/cost occurs in both sources.
    frame["source_priority"] = frame["source"].eq("trajectory").astype(int)
    frame = frame.sort_values(["strategy", "seed", "forward_passes", "source_priority"]).drop_duplicates(
        ["strategy", "seed", "forward_passes"], keep="last"
    )
    return frame[PLOTTED_COLUMNS].sort_values(["strategy", "seed", "forward_passes"]).reset_index(drop=True)


def _mean_curve(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    strategy_frame = frame[frame["strategy"] == strategy]
    # Cost values are integer for trajectories and pass counts for benchmarks;
    # group exact observed values only, without interpolation or fabricated data.
    grouped = strategy_frame.groupby("forward_passes")["recall"].agg(["mean", "std", "count"]).reset_index()
    grouped["std"] = grouped["std"].fillna(0.0)
    return grouped.sort_values("forward_passes")


def create_plot(results_dir: Path | str = RESULTS_DIRECTORY) -> dict[str, str | int]:
    """Write PNG, PDF, and underlying plotted CSV, using available evidence only."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    plotted = collect_plot_data(directory)
    plotted_path = directory / "recall_vs_forward_passes_data.csv"
    plotted.to_csv(plotted_path, index=False)
    png_path, pdf_path = directory / "recall_vs_forward_passes.png", directory / "recall_vs_forward_passes.pdf"

    figure, axis = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    axis.axhline(0.80, color="#6C757D", linestyle="--", linewidth=1.4, label="80% recall target")
    plotted_strategies: list[str] = []
    for strategy in ("circuitlab", "exhaustive", "acdc"):
        if strategy not in set(plotted["strategy"]):
            continue
        curve = _mean_curve(plotted, strategy)
        color = COLORS[strategy]
        label = DISPLAY_NAMES[strategy]
        axis.plot(curve["forward_passes"], curve["mean"], marker="o", linewidth=2.4, color=color, label=label)
        if (curve["count"] > 1).any():
            axis.fill_between(curve["forward_passes"], curve["mean"] - curve["std"], curve["mean"] + curve["std"], color=color, alpha=0.18, linewidth=0)
        plotted_strategies.append(strategy)
    axis.set_xlabel("Forward passes")
    axis.set_ylabel("Recall of known IOI circuit")
    axis.set_ylim(-0.02, 1.05)
    axis.set_title("IOI circuit recovery versus experimental cost")
    axis.grid(axis="y", alpha=0.24)
    if plotted_strategies:
        axis.legend(frameon=False, loc="lower right")
    else:
        axis.text(0.5, 0.48, "No completed benchmark trajectories available", ha="center", va="center", transform=axis.transAxes, color="#6C757D")
        axis.text(0.5, 0.40, "Unavailable strategies are intentionally omitted.", ha="center", va="center", transform=axis.transAxes, color="#6C757D", fontsize=9)
    figure.savefig(png_path, dpi=200, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"png_path": str(png_path), "pdf_path": str(pdf_path), "data_path": str(plotted_path), "points": len(plotted), "strategies": len(plotted_strategies)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIRECTORY)
    args = parser.parse_args()
    result = create_plot(args.results_dir)
    print(f"Plotted {result['points']} source-backed points for {result['strategies']} strategies.")
    print(f"PNG:  {result['png_path']}")
    print(f"PDF:  {result['pdf_path']}")
    print(f"Data: {result['data_path']}")


if __name__ == "__main__":
    main()
