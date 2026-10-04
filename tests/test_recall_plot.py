"""Source-backed recall-cost plot tests."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

from experiments.plot_recall_vs_forward_passes import collect_plot_data, create_plot


class RecallPlotTests(unittest.TestCase):
    def test_uses_completed_rows_only_and_writes_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (root / "benchmark_raw.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["strategy", "seed", "forward_passes", "recall", "status"])
                writer.writeheader()
                writer.writerows([
                    {"strategy": "exhaustive", "seed": 0, "forward_passes": 100, "recall": 0.8, "status": "complete"},
                    {"strategy": "circuitlab", "seed": 0, "forward_passes": "", "recall": "", "status": "unavailable"},
                ])
            trajectories = root / "trajectories"
            trajectories.mkdir()
            (trajectories / "circuitlab_recovery_trajectory.json").write_text(json.dumps({
                "strategy": "circuitlab", "seed": 0,
                "trajectory": [{"forward_passes": 10, "recall": 0.4}, {"forward_passes": 50, "recall": 0.9}],
            }), encoding="utf-8")
            frame = collect_plot_data(root)
            self.assertEqual(frame["strategy"].tolist(), ["circuitlab", "circuitlab", "exhaustive"])
            result = create_plot(root)
            self.assertEqual(result["strategies"], 2)
            self.assertEqual(result["points"], 3)
            self.assertTrue(Path(result["png_path"]).exists())
            self.assertTrue(Path(result["pdf_path"]).exists())
            self.assertTrue(Path(result["data_path"]).exists())

    def test_no_data_creates_honest_empty_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = create_plot(directory)
            self.assertEqual(result["points"], 0)
            self.assertEqual(result["strategies"], 0)
            self.assertTrue(Path(result["png_path"]).exists())
            self.assertTrue(Path(result["pdf_path"]).exists())


if __name__ == "__main__":
    unittest.main()
