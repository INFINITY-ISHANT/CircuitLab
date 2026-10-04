"""Set-based IOI circuit truth scoring tests without altering source truth."""

import unittest

import lab_tools
from circuitlab.datasets import load_ioi_truth, validate_ioi_truth
from circuitlab.evaluation.circuit import score_circuit_against_truth


SYNTHETIC_TRUTH = {
    "task": "ioi",
    "status": "verified",
    "source": "synthetic test fixture",
    "heads": [
        {"layer": 9, "head": 9, "component": "L9H9", "role": "name_mover", "source": "fixture"},
        {"layer": 10, "head": 0, "component": "L10H0", "role": "backup_name_mover", "source": "fixture"},
        {"layer": 5, "head": 5, "component": "L5H5", "role": "induction", "source": "fixture"},
    ],
}


class TruthScoringTests(unittest.TestCase):
    def test_exact_tp_fp_fn_precision_recall_and_f1(self) -> None:
        result = score_circuit_against_truth(["L9H9", "L10H0", "L0H0"], SYNTHETIC_TRUTH)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 1, 1))
        self.assertAlmostEqual(result["precision"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 2 / 3)
        self.assertAlmostEqual(result["f1"], 2 / 3)
        self.assertFalse(result["success"])
        self.assertEqual(result["role_breakdown"]["name_mover"]["recall"], 1.0)

    def test_duplicates_do_not_change_set_score_and_threshold_is_inclusive(self) -> None:
        truth = SYNTHETIC_TRUTH | {
            "heads": SYNTHETIC_TRUTH["heads"][:2]
            + [
                {"layer": 1, "head": 1, "component": "L1H1", "role": "induction", "source": "fixture"},
                {"layer": 1, "head": 2, "component": "L1H2", "role": "induction", "source": "fixture"},
                {"layer": 1, "head": 3, "component": "L1H3", "role": "induction", "source": "fixture"},
            ]
        }
        result = score_circuit_against_truth(["L9H9", "L10H0", "L1H1", "L1H2", "L0H0", "L0H0"], truth)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (4, 1, 1))
        self.assertAlmostEqual(result["precision"], 0.8)
        self.assertAlmostEqual(result["recall"], 0.8)
        self.assertTrue(result["success"])

    def test_invalid_head_identity_is_rejected(self) -> None:
        invalid = SYNTHETIC_TRUTH | {
            "heads": [{"layer": 12, "head": 0, "component": "L12H0", "role": "name_mover", "source": "fixture"}]
        }
        with self.assertRaisesRegex(ValueError, r"\[0, 11\]"):
            validate_ioi_truth(invalid)

    def test_verified_repository_truth_loads_and_scores_a_circuit(self) -> None:
        truth = load_ioi_truth()
        self.assertEqual(truth["status"], "verified")
        self.assertEqual(len(truth["heads"]), 26)
        score = lab_tools.score_vs_truth(["L9H9"])
        self.assertEqual((score["tp"], score["fp"], score["fn"]), (1, 0, 25))
        self.assertFalse(score["success"])


if __name__ == "__main__":
    unittest.main()
