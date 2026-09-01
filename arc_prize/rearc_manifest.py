"""Validate RE-ARC JSON files and build a compact, reproducible dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PreparedDataset:
    output_dir: Path
    manifest_path: Path
    database_path: Path
    manifest: dict[str, Any]


def _validate_grid(grid: Any, *, context: str, max_size: int) -> list[list[int]]:
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"{context}: grid must be a non-empty list")
    if len(grid) > max_size:
        raise ValueError(f"{context}: grid height {len(grid)} exceeds {max_size}")
    if not isinstance(grid[0], list) or not grid[0]:
        raise ValueError(f"{context}: rows must be non-empty lists")

    width = len(grid[0])
    if width > max_size:
        raise ValueError(f"{context}: grid width {width} exceeds {max_size}")
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{context}: row {row_index} makes grid non-rectangular")
        for column_index, color in enumerate(row):
            if isinstance(color, bool) or not isinstance(color, int) or not 0 <= color <= 9:
                raise ValueError(
                    f"{context}: invalid color at ({row_index}, {column_index}): {color!r}"
                )
    return grid


def encode_grid(grid: list[list[int]]) -> bytes:
    """Encode a validated grid as height, width, then row-major uint8 cells."""
    return bytes((len(grid), len(grid[0]), *(color for row in grid for color in row)))


def decode_grid(blob: bytes) -> list[list[int]]:
    if len(blob) < 3:
        raise ValueError("invalid encoded grid")
    height, width = blob[0], blob[1]
    cells = blob[2:]
    if height == 0 or width == 0 or len(cells) != height * width:
        raise ValueError("invalid encoded grid dimensions")
    return [list(cells[offset : offset + width]) for offset in range(0, len(cells), width)]


def _family_seed(seed: int, family_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{family_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _source_files(source_dir: Path) -> list[Path]:
    tasks_dir = source_dir / "tasks" if (source_dir / "tasks").is_dir() else source_dir
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise ValueError(f"no JSON task files found in {tasks_dir}")
    return files


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def prepare_rearc_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 42,
    validation_ratio: float = 0.1,
    max_grid_size: int = 12,
    min_examples: int = 6,
    overwrite: bool = False,
) -> PreparedDataset:
    """Validate source files and create ``manifest.json`` plus ``examples.sqlite3``."""
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    if min_examples < 6:
        raise ValueError("min_examples must be at least 6 (four demos, train target, val target)")

    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    database_path = output_dir / "examples.sqlite3"
    if not overwrite and (manifest_path.exists() or database_path.exists()):
        raise FileExistsError(
            f"prepared dataset already exists in {output_dir}; pass overwrite=True to replace it"
        )

    temporary_database = output_dir / f".examples.sqlite3.{os.getpid()}.tmp"
    if temporary_database.exists():
        temporary_database.unlink()

    connection = sqlite3.connect(temporary_database)
    families: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    total_examples = 0
    prepared_examples = 0
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE examples (
                family_id TEXT NOT NULL,
                example_index INTEGER NOT NULL,
                split TEXT NOT NULL CHECK(split IN ('train', 'validation')),
                input_grid BLOB NOT NULL,
                output_grid BLOB NOT NULL,
                PRIMARY KEY (family_id, example_index)
            ) WITHOUT ROWID;
            CREATE INDEX examples_split_family ON examples(split, family_id);
            """
        )

        for source_file in _source_files(source_dir):
            raw_bytes = source_file.read_bytes()
            try:
                examples = json.loads(raw_bytes)
            except json.JSONDecodeError as error:
                raise ValueError(f"{source_file}: invalid JSON: {error}") from error
            if not isinstance(examples, list):
                raise ValueError(f"{source_file}: top-level value must be an array")

            family_id = source_file.stem
            total_examples += len(examples)
            validated: list[tuple[bytes, bytes]] = []
            for index, example in enumerate(examples):
                context = f"{source_file.name}[{index}]"
                if not isinstance(example, dict) or "input" not in example or "output" not in example:
                    raise ValueError(f"{context}: expected an object with input and output")
                input_grid = _validate_grid(
                    example["input"], context=f"{context}.input", max_size=max_grid_size
                )
                output_grid = _validate_grid(
                    example["output"], context=f"{context}.output", max_size=max_grid_size
                )
                validated.append((encode_grid(input_grid), encode_grid(output_grid)))

            source_hash = hashlib.sha256(raw_bytes).hexdigest()
            if len(validated) < min_examples:
                excluded.append(
                    {
                        "family_id": family_id,
                        "num_examples": len(validated),
                        "reason": f"fewer than {min_examples} examples",
                        "sha256": source_hash,
                        "source_file": source_file.name,
                    }
                )
                continue

            indices = list(range(len(validated)))
            random.Random(_family_seed(seed, family_id)).shuffle(indices)
            validation_count = max(1, round(len(indices) * validation_ratio))
            validation_count = min(validation_count, len(indices) - 5)
            validation_indices = sorted(indices[:validation_count])
            validation_set = set(validation_indices)
            train_indices = sorted(index for index in indices if index not in validation_set)

            connection.executemany(
                "INSERT INTO examples VALUES (?, ?, ?, ?, ?)",
                (
                    (
                        family_id,
                        index,
                        "validation" if index in validation_set else "train",
                        input_blob,
                        output_blob,
                    )
                    for index, (input_blob, output_blob) in enumerate(validated)
                ),
            )
            prepared_examples += len(validated)
            families.append(
                {
                    "family_id": family_id,
                    "num_examples": len(validated),
                    "sha256": source_hash,
                    "source_file": source_file.name,
                    "train_indices": train_indices,
                    "validation_indices": validation_indices,
                }
            )

        connection.commit()
        connection.execute("PRAGMA optimize")
    except Exception:
        connection.close()
        if temporary_database.exists():
            temporary_database.unlink()
        raise
    else:
        connection.close()

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_dir": str(source_dir),
        "seed": seed,
        "validation_ratio": validation_ratio,
        "max_grid_size": max_grid_size,
        "min_examples": min_examples,
        "num_source_families": len(families) + len(excluded),
        "num_families": len(families),
        "num_excluded_families": len(excluded),
        "num_source_examples": total_examples,
        "num_examples": prepared_examples,
        "families": families,
        "excluded_families": excluded,
    }
    # The prepared artifact may be copied between a workstation, $HOME, and OAR
    # scratch. Keep the source path informational rather than making location
    # part of dataset identity.
    fingerprint_fields = {key: value for key, value in manifest.items() if key != "source_dir"}
    fingerprint_payload = json.dumps(
        fingerprint_fields, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest["fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()

    os.replace(temporary_database, database_path)
    _atomic_json(manifest_path, manifest)
    return PreparedDataset(output_dir, manifest_path, database_path, manifest)


def _summary_lines(manifest: dict[str, Any]) -> Iterable[str]:
    yield f"Prepared dataset: {manifest['num_families']} families"
    yield f"Examples: {manifest['num_examples']:,} / {manifest['num_source_examples']:,}"
    yield f"Excluded families: {manifest['num_excluded_families']}"
    yield f"Fingerprint: {manifest['fingerprint']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="RE-ARC directory containing tasks/*.json")
    parser.add_argument("--output", required=True, help="Prepared dataset output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    prepared = prepare_rearc_dataset(
        args.source,
        args.output,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
        overwrite=args.overwrite,
    )
    print("\n".join(_summary_lines(prepared.manifest)))
    print(f"Manifest: {prepared.manifest_path}")
    print(f"Database: {prepared.database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
