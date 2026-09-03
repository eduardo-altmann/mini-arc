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


def _task_is_eligible(task: dict[str, Any], grid_dim: int) -> tuple[bool, str]:
    train = task.get("train")
    test = task.get("test")
    if not isinstance(train, list) or not isinstance(test, list) or not train or not test:
        return False, "missing train/test examples"
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
    for length in range(3, len(train_examples) + 1):
        for combination in itertools.combinations(train_examples, length):
            for ordering in itertools.permutations(combination):
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
    accuracy_cutoff: float,
) -> tuple[ARCVisionEncoder, int, int]:
    model = copy.deepcopy(base_model).to(device)
    examples = _ttt_examples(train_examples, config)
    if not examples or epochs == 0:
        return model.eval(), 0, len(examples)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weights = torch.ones(model.num_classes, device=device)
    weights[0] = 0.2
    criterion = nn.CrossEntropyLoss(weight=weights)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    model.train()
    epochs_run = 0
    for _ in range(epochs):
        correct = 0
        cells = 0
        order = torch.randperm(len(examples)).tolist()
        for start in range(0, len(order), batch_size):
            chunk = [examples[index] for index in order[start : start + batch_size]]
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
            predictions = logits.argmax(dim=-1)
            correct += (predictions == targets).sum().item()
            cells += targets.numel()
        epochs_run += 1
        if correct / cells >= accuracy_cutoff:
            break
    return model.eval(), epochs_run, len(examples)


def _predict(
    model: ARCVisionEncoder,
    grids: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
    *,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    with torch.no_grad(), autocast(device_type=device.type, enabled=device.type == "cuda"):
        target_batch = target.unsqueeze(0).to(device) if target is not None else None
        return model.generate(
            grids.unsqueeze(0).to(device),
            masks.unsqueeze(0).to(device),
            tgt=target_batch,
        )[0][0].cpu()


def _crop_prediction(prediction: torch.Tensor) -> list[list[int]]:
    """Convert 0=padding, 1..10=ARC colors into a JSON ARC grid."""
    occupied = prediction != 0
    if not occupied.any():
        return [[0]]
    rows = occupied.any(dim=1).nonzero(as_tuple=True)[0]
    cols = occupied.any(dim=0).nonzero(as_tuple=True)[0]
    cropped = prediction[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1] - 1
    return cropped.clamp_min(0).tolist()


def _metrics(
    predictions: list[torch.Tensor],
    targets: list[torch.Tensor],
    task_ids: list[str],
) -> dict[str, float | int]:
    if not targets:
        return {
            "puzzles_scored": 0,
            "queries_scored": 0,
            "score": 0,
            "score_percent": 0.0,
            "accuracy": 0.0,
            "closeness": 0,
            "closeness_percent": 0.0,
            "cell_accuracy": 0.0,
            "exact_grid_accuracy": 0.0,
        }
    if not (len(predictions) == len(targets) == len(task_ids)):
        raise ValueError("predictions, targets, and task IDs must have equal lengths")
    cells = sum(target.numel() for target in targets)
    correct = sum((prediction == target).sum().item() for prediction, target in zip(predictions, targets))
    exact_queries = sum(torch.equal(prediction, target) for prediction, target in zip(predictions, targets))
    task_cells: dict[str, int] = {}
    task_correct: dict[str, int] = {}
    task_exact: dict[str, bool] = {}
    for task_id, prediction, target in zip(task_ids, predictions, targets):
        task_cells[task_id] = task_cells.get(task_id, 0) + target.numel()
        task_correct[task_id] = task_correct.get(task_id, 0) + (prediction == target).sum().item()
        task_exact[task_id] = task_exact.get(task_id, True) and torch.equal(prediction, target)
    puzzle_count = len(task_cells)
    score = sum(task_exact.values())
    closeness = sum(task_correct[task_id] / task_cells[task_id] >= 0.95 for task_id in task_cells)
    return {
        "puzzles_scored": puzzle_count,
        "queries_scored": len(targets),
        "score": score,
        "score_percent": score / puzzle_count,
        "accuracy": correct / cells,
        "closeness": closeness,
        "closeness_percent": closeness / puzzle_count,
        "cell_accuracy": correct / cells,
        "exact_grid_accuracy": exact_queries / len(targets),
    }


def _load_task_ids(path: Path | None, challenges: dict[str, Any]) -> list[str]:
    if path is None:
        return list(challenges)
    task_ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError(f"duplicate task ID in {path}")
    missing = [task_id for task_id in task_ids if task_id not in challenges]
    if missing:
        raise ValueError(f"{len(missing)} requested task IDs are absent from challenges: {missing[:5]}")
    return task_ids


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
    trained_refinement_ratio = checkpoint.get("train_config", {}).get("refinement_ratio", 0.0)
    if args.refinement_rounds > 0 and trained_refinement_ratio <= 0.0:
        raise ValueError(
            "refinement was requested, but this checkpoint never trained its refinement branch"
        )
    params = ARCTransformerEncoderDecoderParams(**checkpoint["model_params"])
    model = ARCVisionEncoder(params).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    config = ARCDatasetParams(max_grid_size=params.grid_dim, max_train_grids=params.num_train_pairs, color_offset=1)
    challenges = _load_json(args.challenges)
    solutions = _load_json(args.solutions) if args.solutions else {}
    requested_task_ids = _load_task_ids(args.task_ids_file, challenges)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    direct_predictions: list[torch.Tensor] = []
    tuned_predictions: list[torch.Tensor] = []
    refined_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    scored_task_ids: list[str] = []
    results: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    eligible = 0
    for task_index, task_id in enumerate(requested_task_ids):
        if args.max_tasks is not None and task_index >= args.max_tasks:
            break
        task = challenges[task_id]
        eligible_task, reason = _task_is_eligible(task, params.grid_dim)
        task_solutions = solutions.get(task_id, [])
        if eligible_task and task_solutions:
            if len(task_solutions) != len(task["test"]):
                eligible_task = False
                reason = "solution count does not match test query count"
            elif any(not _valid_grid(solution, params.grid_dim) for solution in task_solutions):
                eligible_task = False
                reason = "invalid or oversized known solution grid"
        if not eligible_task:
            skipped[task_id] = reason
            continue
        eligible += 1
        all_train_examples = task["train"]
        train_examples = all_train_examples[: params.num_train_pairs]
        tuned_model, ttt_epochs_run, ttt_examples = (
            _adapt_model(
                model,
                train_examples,
                config,
                device,
                epochs=args.ttt_epochs,
                learning_rate=args.ttt_learning_rate,
                weight_decay=args.ttt_weight_decay,
                batch_size=args.ttt_batch_size,
                accuracy_cutoff=args.ttt_accuracy_cutoff,
            )
            if args.ttt_epochs > 0
            else (model, 0, 0)
        )
        task_results = []
        task_queries = task["test"][:1] if args.first_query_only else task["test"]
        for query_index, query in enumerate(task_queries):
            grids, masks = _prompt(train_examples, query["input"], config)
            direct = _predict(model, grids, masks, device)
            tuned = _predict(tuned_model, grids, masks, device)
            refined = tuned
            for _ in range(args.refinement_rounds):
                refined = _predict(tuned_model, grids, masks, device, target=refined)
            record: dict[str, Any] = {
                "direct_prediction": _crop_prediction(direct),
                "ttt_prediction": _crop_prediction(tuned),
                "demonstrations_available": len(all_train_examples),
                "demonstrations_used": len(train_examples),
                "ttt_examples": ttt_examples,
                "ttt_epochs_run": ttt_epochs_run,
            }
            if args.refinement_rounds > 0:
                record["refined_prediction"] = _crop_prediction(refined)
            if query_index < len(task_solutions):
                target = _target(task_solutions[query_index], config)
                direct_predictions.append(direct)
                tuned_predictions.append(tuned)
                refined_predictions.append(refined)
                targets.append(target)
                scored_task_ids.append(task_id)
                record["target"] = task_solutions[query_index]
                record["direct_exact"] = bool(torch.equal(direct, target))
                record["ttt_exact"] = bool(torch.equal(tuned, target))
                if args.refinement_rounds > 0:
                    record["refined_exact"] = bool(torch.equal(refined, target))
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
            "accuracy_cutoff": args.ttt_accuracy_cutoff,
        },
        "refinement_rounds": args.refinement_rounds,
        "first_query_only": args.first_query_only,
        "tasks_requested": (
            len(requested_task_ids)
            if args.max_tasks is None
            else min(args.max_tasks, len(requested_task_ids))
        ),
        "tasks_eligible": eligible,
        "tasks_skipped": skipped,
        "direct_metrics": _metrics(direct_predictions, targets, scored_task_ids),
        "ttt_metrics": _metrics(tuned_predictions, targets, scored_task_ids),
        "predictions": results,
    }
    if args.refinement_rounds > 0:
        report["refined_metrics"] = _metrics(refined_predictions, targets, scored_task_ids)
    _atomic_json_dump(report, Path(args.output))
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ttt-epochs", type=int, default=15)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-5)
    parser.add_argument("--ttt-weight-decay", type=float, default=1e-5)
    parser.add_argument("--ttt-batch-size", type=int, default=4)
    parser.add_argument("--ttt-accuracy-cutoff", type=float, default=0.995)
    parser.add_argument("--refinement-rounds", type=int, default=0)
    parser.add_argument("--task-ids-file", type=Path)
    parser.add_argument(
        "--first-query-only",
        action="store_true",
        help="score one query per puzzle, matching the original paper's evaluation artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.ttt_epochs < 0 or args.refinement_rounds < 0 or args.ttt_batch_size < 1:
        parser.error("TTT epochs must be non-negative and batch size must be positive")
    if not 0.0 < args.ttt_accuracy_cutoff <= 1.0:
        parser.error("TTT accuracy cutoff must be in (0, 1]")
    return args


def main() -> int:
    report = evaluate(_parse_args())
    summary_keys = ["tasks_eligible", "tasks_skipped", "direct_metrics", "ttt_metrics"]
    if "refined_metrics" in report:
        summary_keys.append("refined_metrics")
    print(json.dumps({key: report[key] for key in summary_keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
