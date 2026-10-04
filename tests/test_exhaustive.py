"""Non-model tests for exhaustive patching orchestration."""

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.exhaustive import RESULT_COLUMNS, _read_resumable_rows, _write_outputs, all_head_components, result_paths


class ExhaustiveExperimentTests(unittest.TestCase):
    def test_component_inventory_is_full_and_ordered(self) -> None:
        components = all_head_components()
        self.assertEqual(len(components), 144)
        self.assertEqual(components[0], "L0H0")
        self.assertEqual(components[-1], "L11H11")
        self.assertEqual(all_head_components(max_layers=2), [f"L{layer}H{head}" for layer in range(2) for head in range(12)])

    def test_raw_preserves_order_and_ranked_orders_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            raw_path, ranked_path = result_paths(8, 42, Path(directory))
            base = {column: 0 for column in RESULT_COLUMNS}
            first = base | {"component": "L0H0", "effect": -0.5, "cumulative_forward_passes": 3}
            second = base | {"component": "L0H1", "effect": 1.5, "cumulative_forward_passes": 4}
            _write_outputs([first, second], raw_path, ranked_path)
            resumed = _read_resumable_rows(raw_path, ["L0H0", "L0H1"], resume=True)
            self.assertEqual([row["component"] for row in resumed], ["L0H0", "L0H1"])
            self.assertEqual(pd.read_csv(ranked_path).iloc[0]["component"], "L0H1")


if __name__ == "__main__":
    unittest.main()
