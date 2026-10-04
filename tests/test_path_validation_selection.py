"""Source-only selection tests for the path-patching validation script."""

import unittest

from experiments.validate_path_patching import select_documented_path_pair


class DocumentedPathPairTests(unittest.TestCase):
    def test_selects_only_source_listed_earlier_sender_and_name_mover_receiver(self) -> None:
        truth = {
            "heads": [
                {"layer": 5, "head": 5, "component": "L5H5", "role": "Induction"},
                {"layer": 9, "head": 9, "component": "L9H9", "role": "Name Mover"},
            ]
        }
        self.assertEqual(select_documented_path_pair(truth), ("L5H5", "L9H9"))

    def test_refuses_to_invent_pair_when_source_roles_are_insufficient(self) -> None:
        truth = {"heads": [{"layer": 9, "head": 9, "component": "L9H9", "role": "Name Mover"}]}
        with self.assertRaisesRegex(ValueError, "no sender/receiver pair was inferred"):
            select_documented_path_pair(truth)


if __name__ == "__main__":
    unittest.main()
