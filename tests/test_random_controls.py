"""Deterministic, non-model tests for same-size random circuit controls."""

import unittest
from types import SimpleNamespace

import torch

from circuitlab.evaluation.circuit import _final_position_multi_head_hook, sample_random_circuits
from circuitlab.interpretability.patching import attention_head_output_hook_name


class RandomCircuitSamplingTests(unittest.TestCase):
    def test_sampling_is_reproducible_unique_and_excludes_candidate(self) -> None:
        candidate = ["L0H0", "L9H9", "L11H11"]
        first = sample_random_circuits(candidate, n_random=20, seed=42)
        second = sample_random_circuits(candidate, n_random=20, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({tuple(circuit) for circuit in first}), 20)
        self.assertTrue(all(len(circuit) == len(candidate) for circuit in first))
        self.assertTrue(all(set(circuit) != set(candidate) for circuit in first))

    def test_different_seed_normally_changes_samples(self) -> None:
        candidate = ["L1H1", "L2H2"]
        self.assertNotEqual(
            sample_random_circuits(candidate, n_random=5, seed=1),
            sample_random_circuits(candidate, n_random=5, seed=2),
        )

    def test_full_circuit_has_no_non_candidate_control(self) -> None:
        all_heads = [f"L{layer}H{head}" for layer in range(12) for head in range(12)]
        with self.assertRaisesRegex(ValueError, "only 0 non-candidate"):
            sample_random_circuits(all_heads, n_random=1, seed=42)


class SimultaneousCircuitHookTests(unittest.TestCase):
    def test_multiple_heads_patch_only_the_final_token(self) -> None:
        hook_name = attention_head_output_hook_name(0)
        clean_result = torch.zeros((2, 3, 12, 4))
        clean_result[:, -1, 1, :] = 1.0
        clean_result[:, -1, 3, :] = 3.0
        _, hook = _final_position_multi_head_hook({hook_name: clean_result}, layer=0, heads=[1, 3])

        output = hook(torch.zeros((2, 3, 12, 4)), SimpleNamespace(name=hook_name))

        torch.testing.assert_close(output[:, -1, 1, :], torch.ones((2, 4)))
        torch.testing.assert_close(output[:, -1, 3, :], torch.full((2, 4), 3.0))
        torch.testing.assert_close(output[:, :-1, [1, 3], :], torch.zeros((2, 2, 2, 4)))
        torch.testing.assert_close(output[:, :, 0, :], torch.zeros((2, 3, 4)))


if __name__ == "__main__":
    unittest.main()
