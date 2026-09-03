"""Evaluate a mini-arc-v12 checkpoint on ARC-AGI tasks, with optional TTT."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from arc_prize.data import ARCDatasetParams, pad_and_mask_grid
from arc_prize.model import ARCTransformerEncoderDecoderParams, ARCVisionEncoder


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected an object in {path}")
    return value


def _valid_grid(grid: Any, grid_dim: int) -> bool:
    return (
        isinstance(grid, list)
        and 0 < len(grid) <= grid_dim
        and all(
            isinstance(row, list)
            and len(row) == len(grid[0])
            and 0 < len(row) <= grid_dim
            and all(isinstance(color, int) and 0 <= color <= 9 for color in row)
            for row in grid
        )
    )


def _task_is_eligible(task: dict[str, Any], grid_dim: int, num_pairs: int) -> tuple[bool, str]:
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, list) or not isinstance(test, list) or not train or not test:
        return False, "missing train/test examples"
    if len(train) > num_pairs:
        return False, f"has {len(train)} demonstrations; model supports at most {num_pairs}"
    for example in [*train, *test]:
        if not isinstance(example, dict) or not _valid_grid(example.get("input"), grid_dim):
            return False, "invalid or oversized input grid"
        if "output" in example and not _valid_grid(example["output"], grid_dim):
            return False, "invalid or oversized output grid"
    return True, ""


def _prompt(
    demonstrations: list[dict[str, list[list[int]]]],
    query: list[list[int]],
    config: ARCDatasetParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_count = 2 * config.max_train_grids + 1
    grids = torch.zeros((grid_count, config.max_grid_size, config.max_grid_size), dtype=torch.int)
    masks = torch.zeros_like(grids, dtype=torch.bool)
    for index, example in enumerate(demonstrations):
        grids[2 * index], masks[2 * index] = pad_and_mask_grid(example["input"], config)
        grids[2 * index + 1], masks[2 * index + 1] = pad_and_mask_grid(example["output"], config)
    grids[-1], masks[-1] = pad_and_mask_grid(query, config)
    return grids, masks


def _target(grid: list[list[int]], config: ARCDatasetParams) -> torch.Tensor:
    return pad_and_mask_grid(grid, config)[0]


def _ttt_examples(
    train_examples: list[dict[str, list[list[int]]]],
    config: ARCDatasetParams,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Use every ordering of the demonstrations as a supervised adaptation item.

    The final element of each ordering is held out as its target; its input is
    the query and the preceding elements are demonstrations. This mirrors the
    original mini-arc fine-tuning construction while retaining only real pairs.
    """
    examples = []
    for ordering in itertools.permutations(train_examples):
        held_out = ordering[-1]
        grids, masks = _prompt(list(ordering[:-1]), held_out["input"], config)
        examples.append((grids, masks, _target(held_out["output"], config)))
    return examples


def _adapt_model(
    base_model: ARCVisionEncoder,
    train_examples: list[dict[str, list[list[int]]]],
    config: ARCDatasetParams,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
) -> ARCVisionEncoder:
    model = copy.deepcopy(base_model).to(device)
    examples = _ttt_examples(train_examples, config)
    if not examples or epochs == 0:
        return model.eval()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = torch.ones(model.num_classes, device=device)
    weights[0] = 0.2
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    model.train()
    for _ in range(epochs):
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            grids = torch.stack([item[0] for item in chunk]).to(device, non_blocking=True)
            masks = torch.stack([item[1] for item in chunk]).to(device, non_blocking=True)
            targets = torch.stack([item[2] for item in chunk]).to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(grids, masks)[0]
                loss = criterion(logits.reshape(-1, model.num_classes), targets.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    return model.eval()


def _predict(model: ARCVisionEncoder, grids: torch.Tensor, masks: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad(), autocast(device_type=device.type, enabled=device.type == "cuda"):
        return model.generate(grids.unsqueeze(0).to(device), masks.unsqueeze(0).to(device))[0][0].cpu()


def _crop_prediction(prediction: torch.Tensor) -> list[list[int]]:
    """Convert 0=padding, 1..10=ARC colors into a JSON ARC grid."""
    occupied = prediction != 0
    if not occupied.any():
        return [[0]]
    rows = occupied.any(dim=1).nonzero(as_tuple=True)[0]
    cols = occupied.any(dim=0).nonzero(as_tuple=True)[0]
    cropped = prediction[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] - 1
    return cropped.clamp_min(0).tolist()


def _metrics(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> dict[str, float | int]:
    if not targets:
        return {"queries_scored": 0, "cell_accuracy": 0.0, "exact_grid_accuracy": 0.0}
    cells = sum(target.numel() for target in targets)
    correct = sum((prediction == target).sum().item() for prediction, target in zip(predictions, targets))
    exact = sum(torch.equal(prediction, target) for prediction, target in zip(predictions, targets))
    return {
        "queries_scored": len(targets),
        "cell_accuracy": correct / cells,
        "exact_grid_accuracy": exact / len(targets),
    }


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary_name, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "vision_encoder":
        raise ValueError("checkpoint is not a vision_encoder checkpoint")
    params = ARCTransformerEncoderDecoderParams(**checkpoint["model_params"])
    model = ARCVisionEncoder(params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = ARCDatasetParams(max_grid_size=params.grid_dim, max_train_grids=params.num_train_pairs, color_offset=1)
    challenges = _load_json(args.challenges)
    solutions = _load_json(args.solutions) if args.solutions else {}

    direct_predictions: list[torch.Tensor] = []
    tuned_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    eligible = 0
    for task_index, (task_id, task) in enumerate(challenges.items()):
        if args.max_tasks is not None and task_index >= args.max_tasks:
            break
        eligible_task, reason = _task_is_eligible(task, params.grid_dim, params.num_train_pairs)
        if not eligible_task:
            skipped[task_id] = reason
            continue
        eligible += 1
        train_examples = task["train"]
        tuned_model = (
            _adapt_model(
                model,
                train_examples,
                config,
                device,
                epochs=args.ttt_epochs,
                learning_rate=args.ttt_learning_rate,
                weight_decay=args.ttt_weight_decay,
                batch_size=args.ttt_batch_size,
            )
            if args.ttt_epochs > 0
            else model
        )
        task_results = []
        task_solutions = solutions.get(task_id, [])
        for query_index, query in enumerate(task["test"]):
            grids, masks = _prompt(train_examples, query["input"], config)
            direct = _predict(model, grids, masks, device)
            tuned = _predict(tuned_model, grids, masks, device)
            record: dict[str, Any] = {
                "direct_prediction": _crop_prediction(direct),
                "ttt_prediction": _crop_prediction(tuned),
            }
            if query_index < len(task_solutions):
                target = _target(task_solutions[query_index], config)
                direct_predictions.append(direct)
                tuned_predictions.append(tuned)
                targets.append(target)
                record["target"] = task_solutions[query_index]
                record["direct_exact"] = bool(torch.equal(direct, target))
                record["ttt_exact"] = bool(torch.equal(tuned, target))
            task_results.append(record)
        results[task_id] = task_results
        del tuned_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_model_name": checkpoint.get("model_name"),
        "model_params": asdict(params),
        "ttt": {
            "epochs": args.ttt_epochs,
            "learning_rate": args.ttt_learning_rate,
            "weight_decay": args.ttt_weight_decay,
            "batch_size": args.ttt_batch_size,
        },
        "tasks_requested": len(challenges) if args.max_tasks is None else min(args.max_tasks, len(challenges)),
        "tasks_eligible": eligible,
        "tasks_skipped": skipped,
        "direct_metrics": _metrics(direct_predictions, targets),
        "ttt_metrics": _metrics(tuned_predictions, targets),
        "predictions": results,
    }
    _atomic_json_dump(report, Path(args.output))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ttt-epochs", type=int, default=2)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ttt-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ttt-batch-size", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.ttt_epochs < 0 or args.ttt_batch_size < 1:
        parser.error("TTT epochs must be non-negative and batch size must be positive")
    return args


def main() -> int:
    report = evaluate(_parse_args())
    print(json.dumps({key: report[key] for key in ("tasks_eligible", "tasks_skipped", "direct_metrics", "ttt_metrics")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
