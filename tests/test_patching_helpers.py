"""Tensor-level checks for selected-head activation replacement."""

from types import SimpleNamespace
import unittest

import torch

from circuitlab.interpretability.patching import (
    attention_head_output_hook_name,
    make_attention_head_replacement_hook,
)


class HeadReplacementHookTests(unittest.TestCase):
    def test_replaces_only_the_selected_head_slice(self):
        layer, head = 0, 2
        hook_name = attention_head_output_hook_name(layer)
        clean_result = torch.ones((1, 3, 12, 5))
        corrupt_result = torch.zeros((1, 3, 12, 5))
        _, replacement = make_attention_head_replacement_hook({hook_name: clean_result}, layer, head)

        patched = replacement(corrupt_result, SimpleNamespace(name=hook_name))

        torch.testing.assert_close(patched[:, :, head, :], torch.ones((1, 3, 5)))
        torch.testing.assert_close(patched[:, :, 0, :], torch.zeros((1, 3, 5)))
        torch.testing.assert_close(corrupt_result, torch.zeros((1, 3, 12, 5)))

    def test_rejects_unaligned_clean_and_corrupt_positions(self):
        layer, head = 0, 0
        hook_name = attention_head_output_hook_name(layer)
        _, replacement = make_attention_head_replacement_hook(
            {hook_name: torch.ones((1, 3, 12, 5))}, layer, head
        )

        with self.assertRaisesRegex(ValueError, "same batch and position"):
            replacement(torch.zeros((1, 2, 12, 5)), SimpleNamespace(name=hook_name))


if __name__ == "__main__":
    unittest.main()
