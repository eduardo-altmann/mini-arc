import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

from arc_prize.rearc_manifest import prepare_rearc_dataset

if torch is not None:
    from arc_prize.rearc_dataset import BalancedReARCDataset


def examples(offset: int) -> list[dict]:
    return [
        {"input": [[(offset + index) % 10]], "output": [[(offset + index + 1) % 10]]}
        for index in range(12)
    ]


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class BalancedReARCDatasetTest(unittest.TestCase):
    def test_shapes_balance_determinism_and_validation_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            tasks = root / "source" / "tasks"
            tasks.mkdir(parents=True)
            (tasks / "family-a.json").write_text(json.dumps(examples(0)))
            (tasks / "family-b.json").write_text(json.dumps(examples(3)))
            prepared = prepare_rearc_dataset(root / "source", root / "prepared", seed=11)

            dataset = BalancedReARCDataset(
                prepared.output_dir,
                split="validation",
                samples_per_family=3,
                seed=19,
            )
            first = dataset[0]
            repeated = dataset[0]
            self.assertEqual(first["family_id"], repeated["family_id"])
            self.assertEqual(first["target_index"], repeated["target_index"])
            self.assertEqual(first["demo_indices"], repeated["demo_indices"])
            self.assertEqual(tuple(first["grids"].shape), (9, 12, 12))
            self.assertEqual(tuple(first["masks"].shape), (9, 12, 12))
            self.assertEqual(tuple(first["output"].shape), (12, 12))
            self.assertNotIn(first["target_index"], first["demo_indices"])

            manifest_family = next(
                family
                for family in prepared.manifest["families"]
                if family["family_id"] == first["family_id"]
            )
            self.assertIn(first["target_index"], manifest_family["validation_indices"])
            self.assertTrue(
                set(first["demo_indices"]).issubset(manifest_family["train_indices"])
            )
            counts = {family["family_id"]: 0 for family in prepared.manifest["families"]}
            for index in range(len(dataset)):
                counts[dataset[index]["family_id"]] += 1
            self.assertEqual(set(counts.values()), {3})


if __name__ == "__main__":
    unittest.main()
