"""Validation for reproducible, token-aligned IOI dataset generation."""

import unittest

import torch

import lab_tools
from circuitlab.datasets import IOIDataset, make_ioi_dataset
from circuitlab.datasets.ioi import TEMPLATES
from circuitlab.model import get_model


class IOIDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Dataset construction loads GPT-2 only to tokenize prompts; no forward runs.
        cls.dataset = make_ioi_dataset(n=24, seed=42)

    def test_length_and_public_tool(self):
        self.assertGreaterEqual(len(TEMPLATES), 10)
        self.assertLessEqual(len(TEMPLATES), 15)
        self.assertEqual(len(self.dataset), 24)
        self.assertEqual(len(self.dataset.clean_prompts), 24)
        self.assertEqual(len(self.dataset.corrupted_prompts), 24)
        self.assertIsInstance(lab_tools.make_dataset("ioi", n=2, seed=3), IOIDataset)

    def test_same_seed_is_identical(self):
        repeat = make_ioi_dataset(n=24, seed=42)
        self.assertEqual(self.dataset.clean_prompts, repeat.clean_prompts)
        self.assertEqual(self.dataset.corrupted_prompts, repeat.corrupted_prompts)
        self.assertEqual(self.dataset.io_names, repeat.io_names)
        self.assertTrue(torch.equal(self.dataset.clean_tokens, repeat.clean_tokens))
        self.assertTrue(torch.equal(self.dataset.corrupted_tokens, repeat.corrupted_tokens))
        self.assertTrue(torch.equal(self.dataset.io_token_ids, repeat.io_token_ids))

    def test_different_seeds_normally_produce_different_examples(self):
        different = make_ioi_dataset(n=24, seed=43)
        self.assertNotEqual(self.dataset.clean_prompts, different.clean_prompts)

    def test_io_and_subject_are_distinct(self):
        corrupted_names = self.dataset.metadata["corrupted_io_names"]
        for io_name, subject_name, corrupt_io_name in zip(
            self.dataset.io_names, self.dataset.subject_names, corrupted_names
        ):
            self.assertNotEqual(io_name, subject_name)
            self.assertNotEqual(io_name, corrupt_io_name)
            self.assertNotEqual(subject_name, corrupt_io_name)

    def test_token_tensor_shapes(self):
        self.assertEqual(self.dataset.clean_tokens.ndim, 2)
        self.assertEqual(self.dataset.corrupted_tokens.ndim, 2)
        self.assertEqual(tuple(self.dataset.clean_tokens.shape), tuple(self.dataset.corrupted_tokens.shape))
        self.assertEqual(self.dataset.clean_tokens.shape[0], len(self.dataset))
        self.assertEqual(tuple(self.dataset.io_token_ids.shape), (len(self.dataset),))
        self.assertEqual(tuple(self.dataset.subject_token_ids.shape), (len(self.dataset),))

    def test_answer_tokens_are_valid_leading_space_gpt2_ids(self):
        model = get_model()
        for names, token_ids in (
            (self.dataset.io_names, self.dataset.io_token_ids),
            (self.dataset.subject_names, self.dataset.subject_token_ids),
        ):
            for name, token_id in zip(names, token_ids.tolist()):
                self.assertGreaterEqual(token_id, 0)
                self.assertLess(token_id, model.cfg.d_vocab)
                self.assertEqual(model.to_string([token_id]), f" {name}")


if __name__ == "__main__":
    unittest.main()
