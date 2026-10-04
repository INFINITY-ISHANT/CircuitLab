"""Tensor and small-IOI validation for two-stage attention-head path patching."""

from types import SimpleNamespace
import unittest

import pandas as pd
import torch

import lab_tools
from circuitlab.budget import get_tracker
from circuitlab.datasets import make_ioi_dataset
from circuitlab.interpretability.path_patching import _make_full_head_replacement_hook
from circuitlab.interpretability.patching import AttentionHeadComponent, attention_head_output_hook_name


class PathPatchHookTests(unittest.TestCase):
    def test_replaces_the_selected_head_at_all_positions(self) -> None:
        component = AttentionHeadComponent(layer=0, head=2)
        hook_name = attention_head_output_hook_name(component.layer)
        source = torch.ones((2, 3, 12, 5))
        _, hook = _make_full_head_replacement_hook({hook_name: source}, component)

        patched = hook(torch.zeros((2, 3, 12, 5)), SimpleNamespace(name=hook_name))

        torch.testing.assert_close(patched[:, :, 2, :], torch.ones((2, 3, 5)))
        torch.testing.assert_close(patched[:, :, 0, :], torch.zeros((2, 3, 5)))


class PublicPathPatchTests(unittest.TestCase):
    def test_one_receiver_uses_four_budgeted_forwards(self) -> None:
        tracker = get_tracker()
        tracker.reset(max_passes=None)
        dataset = make_ioi_dataset(n=1, seed=11)

        results = lab_tools.path_patch("L0H0", ["L1H0"], ds=dataset)

        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(results["sender"].tolist(), ["L0H0"])
        self.assertEqual(results["receiver"].tolist(), ["L1H0"])
        self.assertEqual(results["positions"].tolist(), ["all_true_prompt_positions"])
        self.assertEqual(results["passes_used"].tolist(), [4])
        self.assertEqual(tracker.snapshot()["passes_used"], 4)

    def test_joint_only_returns_just_the_all_row(self) -> None:
        dataset = make_ioi_dataset(n=1, seed=13)

        joint = lab_tools.path_patch("L0H0", ["L1H0", "L1H1"], ds=dataset, joint_only=True)
        default = lab_tools.path_patch("L0H0", ["L1H0", "L1H1"], ds=dataset)

        self.assertEqual(len(joint), 1)
        self.assertEqual(joint["receiver"].tolist(), ["ALL"])
        self.assertEqual(len(default), 3)

    def test_same_layer_receiver_is_rejected(self) -> None:
        dataset = make_ioi_dataset(n=1, seed=12)
        with self.assertRaisesRegex(ValueError, "strictly later layer"):
            lab_tools.path_patch("L1H0", ["L1H1"], ds=dataset)


if __name__ == "__main__":
    unittest.main()
