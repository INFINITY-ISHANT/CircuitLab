"""Synthetic truth tests for cost-indexed circuit recovery trajectories."""

import json
from pathlib import Path
import tempfile
import unittest

from circuitlab.evaluation.recovery import evaluate_recovery_trajectory


def _truth() -> dict:
    return {
        "task": "ioi",
        "status": "verified",
        "source": "synthetic fixture",
        "heads": [
            {"layer": 1, "head": index, "component": f"L1H{index}", "role": "fixture", "source": "test"}
            for index in range(5)
        ],
    }


class RecoveryTrajectoryTests(unittest.TestCase):
    def test_first_success_cost_and_plot_ready_artifacts(self) -> None:
        steps = [
            {"forward_passes": 2, "circuit": ["L1H0", "L1H1"]},
            {"forward_passes": 4, "component": "L1H2"},
            {"forward_passes": 7, "component": "L1H3"},
            {"forward_passes": 9, "component": "L0H0"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_recovery_trajectory("fixture", steps, seed=0, truth=_truth(), output_dir=directory)
            self.assertEqual(result["passes_to_target"], 7)
            self.assertEqual(result["seed"], 0)
            self.assertEqual([row["forward_passes"] for row in result["trajectory"]], [2, 4, 7, 9])
            self.assertEqual(result["trajectory"][2]["circuit_size"], 4)
            self.assertTrue(result["trajectory"][2]["success"])
            self.assertTrue(Path(result["json_path"]).exists())
            self.assertTrue(Path(result["csv_path"]).exists())
            persisted = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
            self.assertEqual(len(persisted["trajectory"]), 4)

    def test_never_reaching_target_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_recovery_trajectory(
                "never", [{"forward_passes": 3, "circuit": ["L1H0", "L0H0"]}], truth=_truth(), output_dir=directory
            )
        self.assertIsNone(result["passes_to_target"])
        self.assertFalse(result["trajectory"][0]["success"])

    def test_invalid_trace_order_or_ambiguous_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            evaluate_recovery_trajectory(
                "bad", [{"forward_passes": 2, "circuit": ["L1H0"]}, {"forward_passes": 1, "component": "L1H1"}],
                truth=_truth(), output_dir=tempfile.gettempdir(),
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            evaluate_recovery_trajectory(
                "bad2", [{"forward_passes": 1, "circuit": ["L1H0"], "component": "L1H1"}],
                truth=_truth(), output_dir=tempfile.gettempdir(),
            )


if __name__ == "__main__":
    unittest.main()
