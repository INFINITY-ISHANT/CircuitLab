"""Small real-IOI checks for the public final-token patching tool."""

import unittest

import pandas as pd

import lab_tools
from circuitlab.budget import get_tracker
from circuitlab.datasets import make_ioi_dataset


class PublicPatchingTests(unittest.TestCase):
    def test_multiple_heads_reuse_clean_and_corrupt_caches(self):
        tracker = get_tracker()
        tracker.reset(max_passes=None)
        dataset = make_ioi_dataset(n=1, seed=7)

        results = lab_tools.patch(["L0H0", "L0H1"], positions="final", ds=dataset)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(
            list(results.columns),
            [
                "component",
                "layer",
                "head",
                "position",
                "clean_logit_diff",
                "corrupted_logit_diff",
                "patched_logit_diff",
                "recovery",
                "effect",
                "passes_used",
            ],
        )
        self.assertEqual(results["component"].tolist(), ["L0H0", "L0H1"])
        self.assertEqual(results["position"].tolist(), ["final", "final"])
        # 1 clean cache + 1 corrupt cache + 1 patched run per head.
        self.assertEqual(results["passes_used"].tolist(), [3, 4])
        self.assertEqual(tracker.snapshot()["passes_used"], 4)


if __name__ == "__main__":
    unittest.main()
