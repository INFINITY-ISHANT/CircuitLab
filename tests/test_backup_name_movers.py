"""Source-role selection tests for the backup-name-mover experiment."""

import unittest

from experiments.backup_name_movers import documented_name_mover_heads


class BackupNameMoverRoleTests(unittest.TestCase):
    def test_reads_only_explicit_primary_and_backup_roles(self) -> None:
        truth = {
            "heads": [
                {"component": "L9H9", "role": "Name Movers"},
                {"component": "L10H0", "role": "primary_name_mover"},
                {"component": "L11H1", "role": "Backup Name Movers"},
                {"component": "L5H5", "role": "induction"},
            ]
        }
        self.assertEqual(documented_name_mover_heads(truth), (["L10H0", "L9H9"], ["L11H1"]))

    def test_missing_documented_roles_fails_without_inference(self) -> None:
        with self.assertRaisesRegex(ValueError, "No heads were inferred"):
            documented_name_mover_heads({"heads": [{"component": "L9H9", "role": "induction"}]})


if __name__ == "__main__":
    unittest.main()
