"""Boundary validation tests for the autonomous-agent public API."""

import json
import unittest

import torch

import lab_tools
from circuitlab.datasets import IOIDataset


def _valid_dataset() -> IOIDataset:
    return IOIDataset(
        clean_prompts=["clean"],
        corrupted_prompts=["corrupt"],
        clean_tokens=torch.tensor([[1, 2]]),
        corrupted_tokens=torch.tensor([[1, 3]]),
        io_names=["Mary"],
        subject_names=["John"],
        io_token_ids=torch.tensor([1]),
        subject_token_ids=torch.tensor([2]),
        seed=0,
        metadata={"task": "ioi", "dataset_id": "test", "prompt_lengths": [2]},
    )


class LabToolsBoundaryTests(unittest.TestCase):
    def test_component_format_and_gpt2_bounds_are_rejected_before_execution(self) -> None:
        for component in ("L12H0", "L0H12", "L0H-1", "foo", 9):
            with self.subTest(component=component):
                with self.assertRaises((TypeError, ValueError)):
                    lab_tools.patch(component, ds=None)

    def test_dotted_and_lowercase_heads_are_normalized_to_stable_names(self) -> None:
        self.assertEqual(lab_tools._validate_components(["9.9", "l10h0", "L7H3"]), ["L9H9", "L10H0", "L7H3"])
        with self.assertRaises(ValueError):
            lab_tools._validate_components(["9.9", "L9H9"])

    def test_invalid_dataset_and_positions_are_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "IOIDataset"):
            lab_tools.run_baseline(ds="not a dataset")
        with self.assertRaisesRegex(NotImplementedError, "positions='final'"):
            lab_tools.patch("L0H0", positions="middle", ds=_valid_dataset())

    def test_unsupported_modes_and_noncausal_receivers_are_rejected(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "mode='mean'"):
            lab_tools.ablate("L0H0", ds=_valid_dataset(), mode="zero")
        with self.assertRaisesRegex(ValueError, "strictly later"):
            lab_tools.path_patch("L2H0", "L2H1", ds=_valid_dataset())

    def test_seed_and_random_count_validation(self) -> None:
        with self.assertRaisesRegex(TypeError, "seed"):
            lab_tools.make_dataset("ioi", n=1, seed="42")
        with self.assertRaisesRegex(ValueError, "n_random"):
            lab_tools.random_circuit_control("L0H0", ds=_valid_dataset(), n_random=0)

    def test_budget_rejects_invalid_configuration_and_is_json_serializable(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_passes"):
            lab_tools.budget(max_passes=-1, reset=True)
        with self.assertRaisesRegex(ValueError, "requires reset=True"):
            lab_tools.budget(max_passes=10)
        snapshot = lab_tools.budget()
        json.dumps(snapshot)
        self.assertEqual(set(snapshot), {"passes_used", "passes_left", "max_passes", "by_operation"})

    def test_malformed_dataset_shape_is_rejected(self) -> None:
        dataset = _valid_dataset()
        dataset.corrupted_tokens = torch.tensor([[1], [2]])
        with self.assertRaisesRegex(ValueError, "equal shape"):
            lab_tools.faithfulness("L0H0", ds=dataset)


if __name__ == "__main__":
    unittest.main()
