"""Non-model tests for source-backed, resumable benchmark orchestration."""

import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from experiments.benchmark import _exhaustive_circuit, _trace_runs, run_benchmark


class BenchmarkRunnerTests(unittest.TestCase):
    def test_trace_loader_reads_seeded_circuits_and_costs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            path.write_text(json.dumps({"runs": [{"seed": 0, "circuit": ["L9H9", "L1H0"], "forward_passes": 12}]}), encoding="utf-8")
            self.assertEqual(_trace_runs(path), {0: {"circuit": ["L1H0", "L9H9"], "forward_passes": 12}})

    def test_exhaustive_selection_requires_complete_sweep_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            rows = [
                {"component": f"L{layer}H{head}", "layer": layer, "head": head, "effect": layer * 12 + head,
                 "cumulative_forward_passes": 292}
                for layer in range(12) for head in range(12)
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            circuit, passes = _exhaustive_circuit(path, 2)
            self.assertEqual(circuit, ["L11H11", "L11H10"])
            self.assertEqual(passes, 292)

    def test_unavailable_rows_are_replaced_on_resume_not_duplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            first, _ = run_benchmark(n=1, results_dir=output)
            second, _ = run_benchmark(n=1, results_dir=output)
            self.assertEqual(len(first), 15)
            self.assertEqual(len(second), 15)
            self.assertEqual(len({(row["strategy"], row["seed"]) for row in second}), 15)
            self.assertTrue(all(row["status"] == "unavailable" for row in second))


if __name__ == "__main__":
    unittest.main()
