"""Tensor-level and small-IOI checks for corrupted-reference mean ablation."""

from types import SimpleNamespace
import unittest

import pandas as pd
import torch

import lab_tools
from circuitlab.budget import get_tracker
from circuitlab.datasets import make_ioi_dataset
from circuitlab.interpretability.ablation import make_mean_head_ablation_hook
from circuitlab.interpretability.patching import attention_head_output_hook_name


class MeanAblationHookTests(unittest.TestCase):
    def test_replaces_only_selected_head_with_batch_mean(self) -> None:
        layer, head = 0, 2
        hook_name = attention_head_output_hook_name(layer)
        # Mean over reference examples is 2.0 at every retained position/width.
        reference_result = torch.stack(
            [torch.ones((3, 12, 5)), torch.full((3, 12, 5), 3.0)], dim=0
        )
        clean_result = torch.zeros((4, 3, 12, 5))
        _, replacement = make_mean_head_ablation_hook({hook_name: reference_result}, layer, head)

        ablated = replacement(clean_result, SimpleNamespace(name=hook_name))

        torch.testing.assert_close(ablated[:, :, head, :], torch.full((4, 3, 5), 2.0))
        torch.testing.assert_close(ablated[:, :, 0, :], torch.zeros((4, 3, 5)))
        torch.testing.assert_close(clean_result, torch.zeros((4, 3, 12, 5)))

    def test_rejects_position_or_width_mismatch(self) -> None:
        layer, head = 0, 0
        hook_name = attention_head_output_hook_name(layer)
        _, replacement = make_mean_head_ablation_hook(
            {hook_name: torch.ones((2, 3, 12, 5))}, layer, head
        )
        with self.assertRaisesRegex(ValueError, "position or d_model"):
            replacement(torch.zeros((4, 2, 12, 5)), SimpleNamespace(name=hook_name))


class PublicMeanAblationTests(unittest.TestCase):
    def test_multiple_heads_are_individually_tracked(self) -> None:
        tracker = get_tracker()
        tracker.reset(max_passes=None)
        dataset = make_ioi_dataset(n=1, seed=9)

        results = lab_tools.ablate(["L0H0", "L0H1"], ds=dataset)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results["component"].tolist(), ["L0H0", "L0H1"])
        self.assertEqual(results["mode"].tolist(), ["mean", "mean"])
        self.assertEqual(results["passes_used"].tolist(), [3, 4])
        self.assertEqual(tracker.snapshot()["passes_used"], 4)

    def test_zero_ablation_is_not_silently_substituted(self) -> None:
        dataset = make_ioi_dataset(n=1, seed=10)
        with self.assertRaisesRegex(NotImplementedError, "zero ablation"):
            lab_tools.ablate("L0H0", ds=dataset, mode="zero")


if __name__ == "__main__":
    unittest.main()
