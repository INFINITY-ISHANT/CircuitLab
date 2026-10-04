"""Append-only run-record and public-tool logging tests."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import torch

import lab_tools
from circuitlab.datasets import IOIDataset
from circuitlab.run_logging import append_run_record


def _dataset() -> IOIDataset:
    return IOIDataset(
        clean_prompts=["clean"], corrupted_prompts=["corrupt"],
        clean_tokens=torch.tensor([[1, 2]]), corrupted_tokens=torch.tensor([[1, 3]]),
        io_names=["Mary"], subject_names=["John"], io_token_ids=torch.tensor([1]), subject_token_ids=torch.tensor([2]),
        seed=42, metadata={"task": "ioi", "dataset_id": "ioi-42-1", "prompt_lengths": [2]},
    )


class RunLoggingTests(unittest.TestCase):
    def test_records_are_valid_jsonl_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            common = dict(tool="patch", task="ioi", seed=42, components=["L0H0"], positions="final", dataset_size=1,
                          metric="ioi_logit_difference", result_summary={"rows": 1}, passes_used=3, parameters={})
            first = append_run_record(**common, path=path)
            second = append_run_record(**common, path=path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 2)
            self.assertNotEqual(first["run_id"], second["run_id"])
            self.assertEqual({line["run_id"] for line in lines}, {first["run_id"], second["run_id"]})
            self.assertNotIn("tensor", json.dumps(lines).lower())

    def test_public_patch_writes_a_bounded_record_to_requested_path(self) -> None:
        frame = pd.DataFrame([{"component": "L0H0", "effect": 0.5, "recovery": 0.1, "passes_used": 3}])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runs.jsonl"
            with patch("circuitlab.interpretability.patching.run_activation_patching", return_value=frame):
                result = lab_tools.patch("L0H0", ds=_dataset(), log_path=path)
            self.assertIs(result, frame)
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(record["tool"], "patch")
            self.assertEqual(record["components"], ["L0H0"])
            self.assertEqual(record["dataset_size"], 1)
            self.assertEqual(record["result_summary"]["rows"], 1)
            self.assertNotIn("per_example", record["result_summary"])

    def test_tensors_are_rejected_in_record_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "tensors"):
                append_run_record(
                    tool="patch", task="ioi", seed=42, components=["L0H0"], positions="final", dataset_size=1,
                    metric="metric", result_summary={"bad": torch.tensor([1])}, passes_used=0, parameters={},
                    path=Path(directory) / "runs.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
