import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from arc_prize.rearc_manifest import decode_grid, encode_grid, prepare_rearc_dataset


def example(color: int) -> dict:
    return {"input": [[color, 0]], "output": [[0], [color]]}


class PrepareReARCDatasetTest(unittest.TestCase):
    def test_prepares_deterministic_split_and_excludes_sparse_family(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "source" / "tasks"
            source.mkdir(parents=True)
            (source / "eligible.json").write_text(
                json.dumps([example(index % 10) for index in range(10)])
            )
            (source / "sparse.json").write_text(json.dumps([example(1), example(2)]))

            first = prepare_rearc_dataset(root / "source", root / "prepared-a", seed=7)
            second = prepare_rearc_dataset(root / "source", root / "prepared-b", seed=7)
            shutil.copytree(root / "source", root / "relocated-source")
            relocated = prepare_rearc_dataset(
                root / "relocated-source", root / "prepared-c", seed=7
            )

            self.assertEqual(first.manifest["num_families"], 1)
            self.assertEqual(first.manifest["num_excluded_families"], 1)
            self.assertEqual(first.manifest["num_examples"], 10)
            self.assertEqual(first.manifest["fingerprint"], second.manifest["fingerprint"])
            self.assertEqual(first.manifest["fingerprint"], relocated.manifest["fingerprint"])
            family = first.manifest["families"][0]
            self.assertEqual(len(family["validation_indices"]), 1)
            self.assertEqual(len(family["train_indices"]), 9)
            self.assertTrue(
                set(family["validation_indices"]).isdisjoint(family["train_indices"])
            )

            connection = sqlite3.connect(first.database_path)
            counts = dict(
                connection.execute("SELECT split, COUNT(*) FROM examples GROUP BY split")
            )
            connection.close()
            self.assertEqual(counts, {"train": 9, "validation": 1})

    def test_rejects_invalid_color(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            source = root / "tasks"
            source.mkdir(parents=True)
            values = [example(index) for index in range(6)]
            values[-1]["output"] = [[10]]
            (source / "bad.json").write_text(json.dumps(values))
            with self.assertRaisesRegex(ValueError, "invalid color"):
                prepare_rearc_dataset(root, root / "prepared")

    def test_grid_binary_round_trip(self) -> None:
        grid = [[0, 1, 9], [2, 3, 4]]
        self.assertEqual(decode_grid(encode_grid(grid)), grid)


if __name__ == "__main__":
    unittest.main()
