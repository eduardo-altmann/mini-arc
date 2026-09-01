import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from arc_prize.train_rearc import atomic_torch_save


@unittest.skipUnless(torch is not None, "PyTorch is not installed")
class AtomicCheckpointTest(unittest.TestCase):
    def test_replaces_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint = Path(temporary_dir) / "latest.pt"
            atomic_torch_save({"step": 1}, checkpoint)
            atomic_torch_save({"step": 2}, checkpoint)
            self.assertEqual(torch.load(checkpoint, weights_only=False), {"step": 2})
            self.assertEqual(list(checkpoint.parent.glob(".latest.pt.*")), [])


if __name__ == "__main__":
    unittest.main()
