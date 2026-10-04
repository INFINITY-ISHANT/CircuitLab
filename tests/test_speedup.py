"""Synthetic target-cost speedup tests."""

import unittest

import lab_tools
from circuitlab.evaluation.speedup import calculate_speedup


class SpeedupTests(unittest.TestCase):
    def test_finite_per_seed_speedups_and_summary(self) -> None:
        result = calculate_speedup(
            [{"seed": 0, "passes_to_target": 100}, {"seed": 1, "passes_to_target": 80}],
            [{"seed": 0, "passes_to_target": 25}, {"seed": 1, "passes_to_target": 40}],
        )
        self.assertEqual([row["speedup"] for row in result["per_seed"]], [4.0, 2.0])
        self.assertEqual(result["mean_speedup"], 3.0)
        self.assertAlmostEqual(result["std_speedup"], 2**0.5)
        self.assertEqual(result["successful_seeds"], 2)

    def test_unreached_or_missing_targets_are_explicit_not_infinite(self) -> None:
        result = lab_tools.speedup(
            [
                {"seed": 0, "passes_to_target": None},
                {"seed": 1, "passes_to_target": 50},
                {"seed": 2, "passes_to_target": 20},
            ],
            [
                {"seed": 0, "passes_to_target": 10},
                {"seed": 1, "passes_to_target": None},
                {"seed": 3, "passes_to_target": 5},
            ],
        )
        self.assertEqual([row["status"] for row in result["per_seed"]], [
            "baseline_did_not_reach_target", "circuitlab_did_not_reach_target", "missing_strategy_record", "missing_strategy_record"
        ])
        self.assertTrue(all(row["speedup"] is None for row in result["per_seed"]))
        self.assertIsNone(result["mean_speedup"])
        self.assertIsNone(result["std_speedup"])
        self.assertEqual(result["successful_seeds"], 0)

    def test_zero_circuitlab_cost_is_not_reported_as_infinite(self) -> None:
        result = calculate_speedup(
            [{"seed": 0, "passes_to_target": 10}], [{"seed": 0, "passes_to_target": 0}]
        )
        self.assertEqual(result["per_seed"][0]["status"], "invalid_zero_circuitlab_target_cost")
        self.assertIsNone(result["per_seed"][0]["speedup"])


if __name__ == "__main__":
    unittest.main()
