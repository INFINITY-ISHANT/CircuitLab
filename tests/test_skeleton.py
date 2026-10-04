"""Basic public entry-point checks."""

import unittest

import lab_tools
from circuitlab import get_model


class PublicApiTests(unittest.TestCase):
    def test_model_entry_point_reuses_central_loader(self):
        from model_loader import get_model as root_get_model

        self.assertIs(get_model, root_get_model)

    def test_literature_search_rejects_empty_queries_before_any_network_call(self):
        for query in ("", "   ", "an of"):
            with self.assertRaises(ValueError):
                lab_tools.search_literature(query)

    def test_present_claim_labels_the_result_as_a_hypothesis(self):
        result = lab_tools.present_claim(["L9H9"], ["R1-01"], ["2211.00593"], "test")
        self.assertEqual(result["status"], "presented")
        self.assertEqual(result["label"], "agent-generated hypothesis")

    def test_budget_exposes_a_json_compatible_snapshot(self):
        snapshot = lab_tools.budget()
        self.assertEqual(set(snapshot), {"passes_used", "passes_left", "max_passes", "by_operation"})


if __name__ == "__main__":
    unittest.main()
