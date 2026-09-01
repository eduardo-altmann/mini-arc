"""Balanced, deterministic PyTorch dataset for prepared RE-ARC examples."""

from __future__ import annotations

import hashlib
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

from arc_prize.data import ARCDatasetParams, pad_and_mask_grid
from arc_prize.rearc_manifest import decode_grid


Split = Literal["train", "validation"]


def _item_seed(seed: int, split: Split, epoch: int, index: int) -> int:
    digest = hashlib.sha256(f"{seed}:{split}:{epoch}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class BalancedReARCDataset(Dataset):
    """Assemble four demonstrations and a target with uniform family frequency.

    The prepared SQLite database remains read-only. Connections are opened lazily,
    making the dataset safe to use from DataLoader worker processes.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: Split,
        samples_per_family: int,
        seed: int = 42,
        config: ARCDatasetParams | None = None,
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"unsupported split: {split}")
        if samples_per_family < 1:
            raise ValueError("samples_per_family must be positive")

        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.database_path = self.dataset_dir / "examples.sqlite3"
        with (self.dataset_dir / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest: dict[str, Any] = json.load(handle)
        if not self.database_path.is_file():
            raise FileNotFoundError(self.database_path)

        self.split = split
        self.samples_per_family = samples_per_family
        self.seed = seed
        self.config = config or ARCDatasetParams(
            max_grid_size=self.manifest["max_grid_size"],
            max_train_grids=4,
            color_offset=1,
        )
        if self.config.max_train_grids != 4:
            raise ValueError("mini-arc-v12 requires exactly four training pairs")

        self.families = sorted(self.manifest["families"], key=lambda item: item["family_id"])
        if not self.families:
            raise ValueError("prepared dataset has no eligible families")
        self._connection: sqlite3.Connection | None = None
        self.epoch = 0

    @classmethod
    def for_steps(
        cls,
        dataset_dir: str | Path,
        *,
        split: Split,
        steps: int,
        batch_size: int,
        world_size: int,
        seed: int = 42,
    ) -> "BalancedReARCDataset":
        with (Path(dataset_dir).expanduser() / "manifest.json").open(encoding="utf-8") as handle:
            family_count = int(json.load(handle)["num_families"])
        required_samples = steps * batch_size * world_size
        return cls(
            dataset_dir,
            split=split,
            samples_per_family=max(1, math.ceil(required_samples / family_count)),
            seed=seed,
        )

    @property
    def fingerprint(self) -> str:
        return str(self.manifest["fingerprint"])

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.families) * self.samples_per_family

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connection"] = None
        return state

    def _database(self) -> sqlite3.Connection:
        if self._connection is None:
            uri = f"file:{self.database_path}?mode=ro"
            self._connection = sqlite3.connect(uri, uri=True)
        return self._connection

    def _load_examples(
        self, family_id: str, indices: list[int]
    ) -> dict[int, tuple[list[list[int]], list[list[int]]]]:
        placeholders = ",".join("?" for _ in indices)
        rows = self._database().execute(
            f"SELECT example_index, input_grid, output_grid FROM examples "
            f"WHERE family_id = ? AND example_index IN ({placeholders})",
            (family_id, *indices),
        )
        examples = {
            int(index): (decode_grid(input_blob), decode_grid(output_blob))
            for index, input_blob, output_blob in rows
        }
        if len(examples) != len(indices):
            raise RuntimeError(f"missing prepared examples for family {family_id}")
        return examples

    def __getitem__(self, index: int) -> dict[str, Any]:
        family = self.families[index % len(self.families)]
        family_id = family["family_id"]
        train_indices = family["train_indices"]
        validation_indices = family["validation_indices"]
        rng = random.Random(_item_seed(self.seed, self.split, self.epoch, index))

        if self.split == "train":
            target_index = rng.choice(train_indices)
            demo_pool = [candidate for candidate in train_indices if candidate != target_index]
        else:
            target_index = rng.choice(validation_indices)
            demo_pool = train_indices
        demo_indices = rng.sample(demo_pool, 4)
        selected_indices = [*demo_indices, target_index]
        examples = self._load_examples(family_id, selected_indices)

        grid_count = 2 * self.config.max_train_grids + 1
        grids = torch.zeros(
            grid_count,
            self.config.max_grid_size,
            self.config.max_grid_size,
            dtype=torch.int,
        )
        masks = torch.zeros_like(grids, dtype=torch.bool)
        for pair_index, example_index in enumerate(demo_indices):
            input_grid, output_grid = examples[example_index]
            grids[2 * pair_index], masks[2 * pair_index] = pad_and_mask_grid(
                input_grid, self.config
            )
            grids[2 * pair_index + 1], masks[2 * pair_index + 1] = pad_and_mask_grid(
                output_grid, self.config
            )

        target_input, target_output = examples[target_index]
        grids[-1], masks[-1] = pad_and_mask_grid(target_input, self.config)
        output, output_mask = pad_and_mask_grid(target_output, self.config)
        return {
            "family_id": family_id,
            "target_index": target_index,
            "demo_indices": demo_indices,
            "grids": grids,
            "masks": masks,
            "output": output,
            "output_mask": output_mask,
        }


def collate_balanced_rearc(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "family_id": [item["family_id"] for item in batch],
        "target_index": [item["target_index"] for item in batch],
        "demo_indices": [item["demo_indices"] for item in batch],
        "grids": torch.stack([item["grids"] for item in batch]),
        "masks": torch.stack([item["masks"] for item in batch]),
        "output": torch.stack([item["output"] for item in batch]),
        "output_mask": torch.stack([item["output_mask"] for item in batch]),
    }
