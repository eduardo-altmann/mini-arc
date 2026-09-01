import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from arc_prize.train_rearc import atomic_file_copy, atomic_torch_save


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AtomicCheckpointTest(unittest.TestCase):
    def test_replaces_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "latest.pt"
            atomic_torch_save({"step": 1}, checkpoint)
            atomic_torch_save({"step": 2}, checkpoint)
            self.assertEqual(torch.load(checkpoint, weights_only=False), {"step": 2})
            self.assertEqual(list(checkpoint.parent.glob(".latest.pt.*")), [])

            mirror = Path(temporary_dir) / "persistent" / "latest.pt"
            atomic_file_copy(checkpoint, mirror)
            self.assertEqual(torch.load(mirror, weights_only=False), {"step": 2})
            self.assertEqual(list(mirror.parent.glob(".latest.pt.*")), [])


if __name__ == "__main__":
    unittest.main()
