"""Synthetic, hand-calculated tests for the canonical IOI metric."""

import unittest

import torch

from circuitlab.metrics import logit_diff


class LogitDiffTests(unittest.TestCase):
    def test_batched_final_position_scores_and_mean(self):
        # The large values before the final sequence position must be ignored.
        logits = torch.tensor(
            [
                [[100.0, 100.0, 100.0, 100.0, 100.0], [50.0, 50.0, 50.0, 50.0, 50.0], [0.0, 1.5, 4.0, 0.0, 0.0]],
                [[100.0, 100.0, 100.0, 100.0, 100.0], [50.0, 50.0, 50.0, 50.0, 50.0], [-1.0, 0.0, 0.0, 0.0, 2.0]],
            ]
        )

        result = logit_diff(logits, io_token_ids=[2, 0], subject_token_ids=[1, 4])

        torch.testing.assert_close(result["per_example"], torch.tensor([2.5, -3.0]))
        torch.testing.assert_close(result["mean"], torch.tensor(-0.25))

    def test_accepts_tensor_token_ids_on_a_batched_input(self):
        logits = torch.zeros((3, 2, 4))
        logits[:, -1, 1] = torch.tensor([1.0, 2.0, 3.0])
        logits[:, -1, 3] = torch.tensor([0.5, 1.0, 4.0])

        result = logit_diff(logits, torch.tensor([1, 1, 1]), torch.tensor([3, 3, 3]))

        torch.testing.assert_close(result["per_example"], torch.tensor([0.5, 1.0, -1.0]))
        torch.testing.assert_close(result["mean"], torch.tensor(1.0 / 6.0))

    def test_rejects_invalid_token_id_shapes_and_values(self):
        logits = torch.zeros((2, 1, 3))
        with self.assertRaisesRegex(ValueError, "shape"):
            logit_diff(logits, [[0], [1]], [1, 2])
        with self.assertRaisesRegex(ValueError, "outside"):
            logit_diff(logits, [0, 3], [1, 2])


if __name__ == "__main__":
    unittest.main()
