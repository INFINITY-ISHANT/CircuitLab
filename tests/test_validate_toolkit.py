"""Unit checks for per-subsystem reporting in the toolkit validation runner."""

import unittest

from experiments.validate_toolkit import run_check


class ValidationHarnessTests(unittest.TestCase):
    def test_failed_check_reports_exception_without_raising(self) -> None:
        result, value = run_check("broken", lambda: (_ for _ in ()).throw(ValueError("details")))
        self.assertFalse(result.passed)
        self.assertIsNone(value)
        self.assertEqual(result.detail, "ValueError: details")

    def test_passing_check_preserves_value(self) -> None:
        result, value = run_check("working", lambda: 7)
        self.assertTrue(result.passed)
        self.assertEqual(value, 7)


if __name__ == "__main__":
    unittest.main()
