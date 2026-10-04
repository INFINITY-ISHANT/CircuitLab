"""Unit tests for consistent forward-pass accounting."""

import unittest

from circuitlab.budget import BudgetTracker


class BudgetTrackerTests(unittest.TestCase):
    def test_unbounded_tracker_counts_batch_calls_not_examples(self):
        tracker = BudgetTracker()
        tracker.consume(operation="baseline.clean", batch_size=100)
        tracker.consume(operation="baseline.clean", batch_size=1)

        self.assertEqual(tracker.passes_used, 2)
        self.assertIsNone(tracker.passes_left)
        self.assertEqual(
            tracker.snapshot(),
            {
                "passes_used": 2,
                "passes_left": None,
                "max_passes": None,
                "by_operation": {"baseline.clean": 2},
            },
        )

    def test_configured_maximum_and_atomic_over_budget_rejection(self):
        tracker = BudgetTracker(max_passes=3)
        tracker.consume(2, operation="patch", batch_size=50)

        with self.assertRaisesRegex(RuntimeError, "budget exceeded"):
            tracker.consume(2, operation="patch", batch_size=50)

        self.assertEqual(tracker.passes_used, 2)
        self.assertEqual(tracker.passes_left, 1)
        self.assertEqual(tracker.snapshot()["by_operation"], {"patch": 2})

    def test_reset_clears_usage_and_can_reconfigure_maximum(self):
        tracker = BudgetTracker(max_passes=4)
        tracker.consume(operation="baseline.clean")
        self.assertEqual(tracker.reset()["max_passes"], 4)
        self.assertEqual(tracker.passes_used, 0)

        snapshot = tracker.reset(max_passes=2)
        self.assertEqual(snapshot["max_passes"], 2)
        self.assertEqual(snapshot["passes_left"], 2)

    def test_rejects_invalid_inputs(self):
        tracker = BudgetTracker()
        with self.assertRaisesRegex(ValueError, "positive"):
            tracker.consume(0)
        with self.assertRaisesRegex(ValueError, "batch_size"):
            tracker.consume(batch_size=0)
        with self.assertRaisesRegex(ValueError, "max_passes"):
            BudgetTracker(max_passes=-1)


if __name__ == "__main__":
    unittest.main()
