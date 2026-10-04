"""The agent layer must turn objects and DataFrames into plain JSON, with no GPU involved."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

import agent_tools
import lab_tools


def _fake_dataset(seed: int = 42, n: int = 100) -> SimpleNamespace:
    return SimpleNamespace(dataset_id=f"ioi-{seed}-{n}")


class AgentToolsTests(unittest.TestCase):
    def setUp(self) -> None:
        agent_tools._DATASETS.clear()

    def test_make_dataset_returns_only_an_id_and_registers_the_object(self) -> None:
        with mock.patch.object(lab_tools, "make_dataset", return_value=_fake_dataset()):
            result = agent_tools.make_dataset("ioi", n=100, seed=42)
        self.assertEqual(result["dataset_id"], "ioi-42-100")
        self.assertIn("ioi-42-100", agent_tools._DATASETS)
        json.dumps(result)

    def test_unknown_id_is_rebuilt_from_its_name_and_garbage_is_rejected(self) -> None:
        with mock.patch.object(lab_tools, "make_dataset", return_value=_fake_dataset(7, 20)) as made:
            ds = agent_tools._dataset("ioi-7-20")
        made.assert_called_once_with("ioi", n=20, seed=7)
        self.assertEqual(ds.dataset_id, "ioi-7-20")
        with self.assertRaises(ValueError):
            agent_tools._dataset("banana")

    def test_patch_returns_json_records_with_nan_and_numpy_cleaned(self) -> None:
        frame = pd.DataFrame(
            {"component": ["L9H9", "L0H0"], "effect": [np.float32(1.23456789), 0.0],
             "recovery": [0.9, float("nan")], "layer": [np.int64(9), np.int64(0)]}
        )
        agent_tools._DATASETS["ioi-42-100"] = _fake_dataset()
        with mock.patch.object(lab_tools, "patch", return_value=frame) as patched:
            records = agent_tools.patch(["L9H9", "L0H0"], "ioi-42-100")
        patched.assert_called_once()
        self.assertEqual(records[0]["component"], "L9H9")
        self.assertIsNone(records[1]["recovery"])
        self.assertEqual(records[0]["layer"], 9)
        json.dumps(records, allow_nan=False)

    def test_faithfulness_bundles_the_random_control(self) -> None:
        agent_tools._DATASETS["ioi-42-100"] = _fake_dataset()
        faith = {"circuit": ["L9H9"], "n": 100, "faithfulness": 0.8123456, "forward_passes": 3}
        control = {"candidate_faithfulness": 0.8, "random_mean": 0.02, "random_std": 0.03,
                   "percentile": 100.0, "beats_random": True, "raw_results_path": "x", "raw": list(range(50))}
        with mock.patch.object(lab_tools, "faithfulness", return_value=faith), \
                mock.patch.object(lab_tools, "random_circuit_control", return_value=control):
            result = agent_tools.faithfulness(["L9H9"], "ioi-42-100")
        self.assertEqual(result["faithfulness"], 0.8123)
        self.assertTrue(result["random_control"]["beats_random"])
        self.assertNotIn("raw", result["random_control"])

    def test_score_vs_truth_reports_unavailable_instead_of_raising(self) -> None:
        with mock.patch.object(lab_tools, "score_vs_truth", side_effect=ValueError("truth not verified")):
            result = agent_tools.score_vs_truth(["L9H9"])
        self.assertFalse(result["available"])
        self.assertIn("truth not verified", result["error"])

    def test_baseline_drops_the_long_per_example_list(self) -> None:
        raw = {"accuracy": 0.96, "mean_logit_diff": 2.7633, "per_example_logit_diffs": [1.0] * 100, "n": 100}
        with mock.patch.object(lab_tools, "run_baseline", return_value=dict(raw)):
            result = agent_tools.run_baseline("ioi")
        self.assertNotIn("per_example_logit_diffs", result)
        self.assertEqual(result["accuracy"], 0.96)


class PathPatchSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        agent_tools._DATASETS.clear()
        agent_tools._DATASETS["ioi-42-100"] = _fake_dataset()

    def test_sweeps_only_given_senders_joint_only_sorted_and_cut(self) -> None:
        effects = {"L0H1": 0.1, "L1H2": 0.9, "L2H3": 0.5, "L3H4": 0.7}

        def fake(sender, receivers, ds=None, **kwargs):
            return pd.DataFrame([{"receiver": "ALL", "path_effect": effects[sender], "recovery": effects[sender] / 2}])

        with mock.patch.object(lab_tools, "path_patch", side_effect=fake) as patched:
            result = agent_tools.path_patch_sweep(["L9H9", "L9H6"], list(effects), "ioi-42-100", top_k=2)
        self.assertEqual(patched.call_count, 4)
        self.assertTrue(all(call.kwargs.get("joint_only") is True for call in patched.call_args_list))
        self.assertEqual([r["sender"] for r in result], ["L1H2", "L3H4"])
        self.assertEqual(result[0]["path_effect"], 0.9)

    def test_rejects_bad_arguments_without_running_anything(self) -> None:
        many = [f"L0H{i % 12}" for i in range(25)]
        bad = [
            ([], ["L0H0"], 10),
            (["L9H9"], [], 10),
            (["L9H9"], ["L0H0"], 0),
            (["L9H9"], many, 10),
        ]
        with mock.patch.object(lab_tools, "path_patch") as patched:
            for receivers, senders, top_k in bad:
                with self.assertRaises(ValueError):
                    agent_tools.path_patch_sweep(receivers, senders, "ioi-42-100", top_k=top_k)
        patched.assert_not_called()


class LiteratureBlocklistTests(unittest.TestCase):
    RESULTS = [{"title": "IOI", "arxiv_id": "2211.00593v1"}, {"title": "Other", "arxiv_id": "1706.03762v5"}]

    def test_blocked_ids_are_dropped_by_prefix(self) -> None:
        with mock.patch.dict(os.environ, {"CIRCUITLAB_BLOCK_ARXIV": "2211.00593, 2304.14997"}), \
                mock.patch.object(lab_tools, "search_literature", return_value=list(self.RESULTS)):
            kept = agent_tools.search_literature("ioi")
        self.assertEqual([r["arxiv_id"] for r in kept], ["1706.03762v5"])

    def test_nothing_is_dropped_when_unset(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("CIRCUITLAB_BLOCK_ARXIV", None)
            with mock.patch.object(lab_tools, "search_literature", return_value=list(self.RESULTS)):
                self.assertEqual(agent_tools.search_literature("ioi"), self.RESULTS)


if __name__ == "__main__":
    unittest.main()
