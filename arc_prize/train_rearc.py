"""Train the checkpoint-safe mini-arc-v12 model on prepared RE-ARC data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import signal
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from arc_prize.model import ARCTransformerEncoderDecoderParams, ARCVisionEncoder
from arc_prize.rearc_dataset import BalancedReARCDataset, collate_balanced_rearc


MODEL_TYPE = "vision_encoder"
DEFAULT_MODEL_PROFILE = "reduced"


@dataclass(frozen=True)
class MiniARCV12Config:
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    train_steps_per_epoch: int = 1000
    validation_steps_per_epoch: int = 100
    max_epochs: int = 100
    patience: int = 15
    warmup_epochs: int = 8
    seed: int = 42
    num_workers: int = 0
    checkpoint_every_steps: int = 0
    refinement_ratio: float = 0.0

    def __post_init__(self) -> None:
        positive_fields = {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "train_steps_per_epoch": self.train_steps_per_epoch,
            "validation_steps_per_epoch": self.validation_steps_per_epoch,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "warmup_epochs": self.warmup_epochs,
        }
        invalid = {name: value for name, value in positive_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f"training values must be positive: {invalid}")
        if self.weight_decay < 0 or self.num_workers < 0 or self.checkpoint_every_steps < 0:
            raise ValueError("weight decay, workers, and checkpoint interval cannot be negative")
        if not 0.0 <= self.refinement_ratio <= 1.0:
            raise ValueError("refinement ratio must be in [0, 1]")


@dataclass(frozen=True)
class MiniARCModelSpec:
    """An architecture profile with an independent checkpoint identity."""

    name: str
    params: ARCTransformerEncoderDecoderParams


MODEL_SPECS = {
    "reduced": MiniARCModelSpec(
        name="mini-arc-v12",
        params=ARCTransformerEncoderDecoderParams(
            grid_dim=12,
            num_train_pairs=4,
            num_colors=10,
            num_encoder_layers=4,
            num_decoder_layers=0,
            num_heads=4,
            d_model=128,
            d_ff=512,
            dropout=0.1,
        ),
    ),
    "full": MiniARCModelSpec(
        name="mini-arc-v12-full",
        params=ARCTransformerEncoderDecoderParams(
            grid_dim=12,
            num_train_pairs=4,
            num_colors=10,
            num_encoder_layers=16,
            num_decoder_layers=0,
            num_heads=16,
            d_model=512,
            d_ff=3072,
            dropout=0.1,
        ),
    ),
    "full-refinement": MiniARCModelSpec(
        name="mini-arc-v12-full-refinement",
        params=ARCTransformerEncoderDecoderParams(
            grid_dim=12,
            num_train_pairs=4,
            num_colors=10,
            num_encoder_layers=16,
            num_decoder_layers=0,
            num_heads=16,
            d_model=512,
            d_ff=3072,
            dropout=0.1,
        ),
    ),
}


def model_spec(profile: str = DEFAULT_MODEL_PROFILE) -> MiniARCModelSpec:
    try:
        return MODEL_SPECS[profile]
    except KeyError as error:
        choices = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"unknown model profile {profile!r}; choose one of: {choices}") from error


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, local_rank, world_size, device


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _base_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu())


def atomic_torch_save(value: Any, path: str | Path) -> None:
    """Write a checkpoint in the destination directory, then atomically rename it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(file_descriptor)
    try:
        torch.save(value, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_file_copy(source: str | Path, destination: str | Path) -> None:
    """Copy a completed local checkpoint to persistent storage atomically."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(file_descriptor)
    try:
        shutil.copyfile(source, temporary_name)
        with open(temporary_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _archive_existing_checkpoints(checkpoint_dir: Path) -> None:
    existing = [checkpoint_dir / "latest.pt", checkpoint_dir / "best.pt"]
    existing = [path for path in existing if path.exists()]
    if not existing:
        return
    archive_dir = checkpoint_dir / "archive" / (
        time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns()}"
    )
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(path, archive_dir / path.name)
    print(f"Archived existing checkpoints in {archive_dir}", flush=True)


def _all_rank_states(
    rank: int, world_size: int, partial_train_metrics: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, float]]]:
    local_state = {"rng": _rng_state(), "partial_train_metrics": partial_train_metrics}
    if world_size == 1:
        return [local_state["rng"]], [local_state["partial_train_metrics"]]
    states: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(states, local_state)
    if any(state is None for state in states):
        raise RuntimeError(f"rank {rank}: failed to collect checkpoint RNG state")
    complete_states = [state for state in states if state is not None]
    return (
        [state["rng"] for state in complete_states],
        [state["partial_train_metrics"] for state in complete_states],
    )


def _checkpoint_state(
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    warmup_scheduler: optim.lr_scheduler.LinearLR,
    plateau_scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    config: MiniARCV12Config,
    dataset_fingerprint: str,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_validation_loss: float,
    epochs_without_improvement: int,
    history: list[dict[str, Any]],
    rng_states: list[dict[str, Any]],
    partial_train_metrics_by_rank: list[dict[str, float]],
    model_specification: MiniARCModelSpec,
) -> dict[str, Any]:
    return {
        "checkpoint_version": 1,
        "model_name": model_specification.name,
        "model_type": MODEL_TYPE,
        "model_params": asdict(model_specification.params),
        "train_config": asdict(config),
        "dataset_fingerprint": dataset_fingerprint,
        "model_state_dict": _base_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
        "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
        "rng_states": rng_states,
        "partial_train_metrics_by_rank": partial_train_metrics_by_rank,
    }


def _load_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    warmup_scheduler: optim.lr_scheduler.LinearLR,
    plateau_scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    dataset_fingerprint: str,
    config: MiniARCV12Config,
    rank: int,
    device: torch.device,
    model_specification: MiniARCModelSpec,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_name") != model_specification.name:
        raise ValueError(
            f"checkpoint is for {checkpoint.get('model_name')}, not {model_specification.name}"
        )
    if checkpoint.get("model_params") != asdict(model_specification.params):
        raise ValueError(
            f"checkpoint model parameters do not match {model_specification.name}"
        )
    if checkpoint.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError("checkpoint dataset fingerprint does not match the prepared dataset")
    saved_config = checkpoint.get("train_config", {})
    current_config = asdict(config)
    mutable_resume_fields = {
        "max_epochs",
        "patience",
        "checkpoint_every_steps",
        "num_workers",
    }
    mismatches = {
        key: (saved_config.get(key, 0.0 if key == "refinement_ratio" else None), value)
        for key, value in current_config.items()
        if key not in mutable_resume_fields
        and saved_config.get(key, 0.0 if key == "refinement_ratio" else None) != value
    }
    if mismatches:
        raise ValueError(f"checkpoint training configuration mismatch: {mismatches}")

    _base_model(model).load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
    plateau_scheduler.load_state_dict(checkpoint["plateau_scheduler_state_dict"])
    states = checkpoint.get("rng_states", [])
    if states:
        _restore_rng_state(states[min(rank, len(states) - 1)])
    return checkpoint


def _make_loaders(
    dataset_dir: Path,
    config: MiniARCV12Config,
    world_size: int,
    rank: int,
) -> tuple[
    BalancedReARCDataset,
    BalancedReARCDataset,
    DistributedSampler,
    DistributedSampler,
    DataLoader,
    DataLoader,
]:
    train_dataset = BalancedReARCDataset.for_steps(
        dataset_dir,
        split="train",
        steps=config.train_steps_per_epoch,
        batch_size=config.batch_size,
        world_size=world_size,
        seed=config.seed,
    )
    validation_dataset = BalancedReARCDataset.for_steps(
        dataset_dir,
        split="validation",
        steps=config.validation_steps_per_epoch,
        batch_size=config.batch_size,
        world_size=world_size,
        seed=config.seed + 1,
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=False,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
        drop_last=False,
    )
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_balanced_rearc,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_args)
    validation_loader = DataLoader(validation_dataset, sampler=validation_sampler, **loader_args)
    return (
        train_dataset,
        validation_dataset,
        train_sampler,
        validation_sampler,
        train_loader,
        validation_loader,
    )


def _reduce_metrics(values: list[float], device: torch.device, world_size: int) -> list[float]:
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def _refinement_target(target: torch.Tensor, retention_ratio: float, num_classes: int) -> torch.Tensor:
    """Create the partial target used by the original mini-arc refinement path."""
    retained = torch.rand(target.shape, device=target.device) < retention_ratio
    random_values = torch.randint(
        0, num_classes, target.shape, device=target.device, dtype=target.dtype
    )
    return torch.where(retained, target, random_values)


def train(
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    config: MiniARCV12Config,
    *,
    persistent_checkpoint_dir: str | Path | None = None,
    force_restart: bool = False,
    model_profile: str = DEFAULT_MODEL_PROFILE,
) -> None:
    rank, local_rank, world_size, device = _distributed_context()
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    persistent_dir = (
        Path(persistent_checkpoint_dir).expanduser().resolve()
        if persistent_checkpoint_dir is not None
        else None
    )
    if persistent_dir == checkpoint_dir:
        persistent_dir = None
    model_specification = model_spec(model_profile)
    latest_path = checkpoint_dir / "latest.pt"
    best_path = checkpoint_dir / "best.pt"

    random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed + rank)

    if force_restart and rank == 0:
        _archive_existing_checkpoints(checkpoint_dir)
        if persistent_dir is not None:
            _archive_existing_checkpoints(persistent_dir)
    elif rank == 0 and persistent_dir is not None:
        persistent_latest = persistent_dir / "latest.pt"
        persistent_best = persistent_dir / "best.pt"
        if not latest_path.exists() and persistent_latest.exists():
            atomic_file_copy(persistent_latest, latest_path)
        if not best_path.exists() and persistent_best.exists():
            atomic_file_copy(persistent_best, best_path)
    _barrier(world_size)

    (
        train_dataset,
        validation_dataset,
        train_sampler,
        validation_sampler,
        train_loader,
        validation_loader,
    ) = _make_loaders(dataset_dir, config, world_size, rank)

    model: nn.Module = ARCVisionEncoder(model_specification.params).to(device)
    if world_size > 1:
        # Direct steps leave tgt_embedding unused; refinement steps leave the
        # learned output_query unused. DDP must support either graph.
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    class_weights = torch.ones(11, device=device)
    class_weights[0] = 0.2
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = GradScaler(device.type, enabled=device.type == "cuda")
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=config.warmup_epochs
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    epoch = 0
    resume_step = 0
    global_step = 0
    best_validation_loss = math.inf
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []
    partial_train_metrics = {
        "loss_sum": 0.0,
        "correct": 0.0,
        "cells": 0.0,
        "examples": 0.0,
    }
    if latest_path.exists() and not force_restart:
        checkpoint = _load_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            warmup_scheduler=warmup_scheduler,
            plateau_scheduler=plateau_scheduler,
            dataset_fingerprint=train_dataset.fingerprint,
            config=config,
            rank=rank,
            device=device,
            model_specification=model_specification,
        )
        epoch = int(checkpoint["epoch"])
        resume_step = int(checkpoint["step_in_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = float(checkpoint["best_validation_loss"])
        epochs_without_improvement = int(checkpoint["epochs_without_improvement"])
        history = list(checkpoint["history"])
        saved_partial_metrics = checkpoint.get("partial_train_metrics_by_rank", [])
        if saved_partial_metrics:
            partial_train_metrics = dict(
                saved_partial_metrics[min(rank, len(saved_partial_metrics) - 1)]
            )
        if rank == 0:
            print(
                f"Resumed {latest_path} at epoch={epoch}, step={resume_step}, global_step={global_step}",
                flush=True,
            )
    elif rank == 0:
        print(f"Starting new {model_specification.name} training run", flush=True)

    stop_requested = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"Rank {rank} received signal {signum}; checkpointing at the next batch boundary", flush=True)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    def save_checkpoint(next_epoch: int, next_step: int, *, is_best: bool = False) -> None:
        rng_states, partial_metrics_by_rank = _all_rank_states(
            rank, world_size, partial_train_metrics
        )
        if rank == 0:
            state = _checkpoint_state(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                warmup_scheduler=warmup_scheduler,
                plateau_scheduler=plateau_scheduler,
                config=config,
                dataset_fingerprint=train_dataset.fingerprint,
                epoch=next_epoch,
                step_in_epoch=next_step,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                epochs_without_improvement=epochs_without_improvement,
                history=history,
                rng_states=rng_states,
                partial_train_metrics_by_rank=partial_metrics_by_rank,
                model_specification=model_specification,
            )
            atomic_torch_save(state, latest_path)
            if persistent_dir is not None:
                atomic_file_copy(latest_path, persistent_dir / "latest.pt")
            if is_best:
                atomic_torch_save(state, best_path)
                if persistent_dir is not None:
                    atomic_file_copy(best_path, persistent_dir / "best.pt")
            print(
                f"Saved latest checkpoint at epoch={next_epoch}, step={next_step}"
                + (" and updated best checkpoint" if is_best else ""),
                flush=True,
            )
        _barrier(world_size)

    if rank == 0:
        print(
            json.dumps(
                {
                    "device": str(device),
                    "world_size": world_size,
                    "families": len(train_dataset.families),
                    "model_name": model_specification.name,
                    "model_params": asdict(model_specification.params),
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "train_items": len(train_dataset),
                    "validation_items": len(validation_dataset),
                    "config": asdict(config),
                },
                indent=2,
            ),
            flush=True,
        )

    try:
        while epoch < config.max_epochs:
            train_dataset.set_epoch(epoch)
            train_sampler.set_epoch(epoch)
            model.train()
            train_loss_sum = float(partial_train_metrics["loss_sum"])
            train_correct = float(partial_train_metrics["correct"])
            train_cells = float(partial_train_metrics["cells"])
            train_examples = float(partial_train_metrics["examples"])
            epoch_start = time.time()
            completed_steps = resume_step

            for step, batch in enumerate(train_loader):
                if step < resume_step:
                    continue
                if step >= config.train_steps_per_epoch:
                    break
                grids = batch["grids"].to(device, non_blocking=True)
                masks = batch["masks"].to(device, non_blocking=True)
                targets = batch["output"].to(device, non_blocking=True).long()
                refinement_target = None
                if config.refinement_ratio > 0.0 and random.random() < config.refinement_ratio:
                    refinement_target = _refinement_target(
                        targets,
                        retention_ratio=random.uniform(0.0, 0.6),
                        num_classes=11,
                    )

                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(grids, masks, tgt=refinement_target)[0]
                    loss = criterion(logits.reshape(-1, 11), targets.reshape(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                predictions = logits.argmax(dim=-1)
                batch_examples = targets.shape[0]
                train_loss_sum += loss.item() * batch_examples
                train_correct += (predictions == targets).sum().item()
                train_cells += targets.numel()
                train_examples += batch_examples
                partial_train_metrics = {
                    "loss_sum": train_loss_sum,
                    "correct": train_correct,
                    "cells": train_cells,
                    "examples": train_examples,
                }
                global_step += 1
                completed_steps = step + 1

                local_stop = torch.tensor(int(stop_requested), device=device)
                if world_size > 1:
                    dist.all_reduce(local_stop, op=dist.ReduceOp.MAX)
                must_stop = bool(local_stop.item())
                interval = config.checkpoint_every_steps
                if interval > 0 and global_step % interval == 0 and not must_stop:
                    save_checkpoint(epoch, completed_steps)
                if must_stop:
                    save_checkpoint(epoch, completed_steps)
                    return

            resume_step = 0
            if completed_steps < config.train_steps_per_epoch:
                raise RuntimeError(
                    f"training loader supplied {completed_steps} steps, expected "
                    f"{config.train_steps_per_epoch}"
                )

            partial_train_metrics = {
                "loss_sum": 0.0,
                "correct": 0.0,
                "cells": 0.0,
                "examples": 0.0,
            }

            validation_dataset.set_epoch(0)
            validation_sampler.set_epoch(0)
            model.eval()
            validation_loss_sum = 0.0
            validation_correct = 0.0
            validation_cells = 0.0
            validation_exact = 0.0
            validation_examples = 0.0
            with torch.no_grad():
                for step, batch in enumerate(validation_loader):
                    if step >= config.validation_steps_per_epoch:
                        break
                    grids = batch["grids"].to(device, non_blocking=True)
                    masks = batch["masks"].to(device, non_blocking=True)
                    targets = batch["output"].to(device, non_blocking=True).long()
                    with autocast(device_type=device.type, enabled=device.type == "cuda"):
                        logits = model(grids, masks)[0]
                        loss = criterion(logits.reshape(-1, 11), targets.reshape(-1))
                    predictions = logits.argmax(dim=-1)
                    matches = predictions == targets
                    batch_examples = targets.shape[0]
                    validation_loss_sum += loss.item() * batch_examples
                    validation_correct += matches.sum().item()
                    validation_cells += targets.numel()
                    validation_exact += matches.flatten(1).all(dim=1).sum().item()
                    validation_examples += batch_examples

            (
                train_loss_sum,
                train_correct,
                train_cells,
                train_examples,
                validation_loss_sum,
                validation_correct,
                validation_cells,
                validation_exact,
                validation_examples,
            ) = _reduce_metrics(
                [
                    train_loss_sum,
                    train_correct,
                    train_cells,
                    train_examples,
                    validation_loss_sum,
                    validation_correct,
                    validation_cells,
                    validation_exact,
                    validation_examples,
                ],
                device,
                world_size,
            )
            validation_loss = validation_loss_sum / validation_examples
            metrics = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "train_loss": train_loss_sum / train_examples,
                "train_cell_accuracy": train_correct / train_cells,
                "validation_loss": validation_loss,
                "validation_cell_accuracy": validation_correct / validation_cells,
                "validation_exact_grid_accuracy": validation_exact / validation_examples,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "duration_seconds": time.time() - epoch_start,
            }
            history.append(metrics)

            improved = validation_loss < best_validation_loss
            if improved:
                best_validation_loss = validation_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch < config.warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(validation_loss)
            epoch += 1
            save_checkpoint(epoch, 0, is_best=improved)
            if rank == 0:
                print(json.dumps(metrics, sort_keys=True), flush=True)

            if epochs_without_improvement >= config.patience:
                if rank == 0:
                    print(
                        f"Early stopping after {epochs_without_improvement} epochs without improvement",
                        flush=True,
                    )
                break
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(Path.home() / "arc-checkpoints" / model_spec().name),
    )
    parser.add_argument(
        "--persistent-checkpoint-dir",
        help="optional mirror for each completed local latest/best checkpoint",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--validation-steps", type=int, default=100)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every-steps", type=int, default=0)
    parser.add_argument(
        "--refinement-ratio",
        type=float,
        default=0.0,
        help="fraction of training steps supplied a noisy partial target (paper uses 0.25)",
    )
    parser.add_argument("--force-restart", action="store_true")
    parser.add_argument(
        "--model-profile",
        choices=sorted(MODEL_SPECS),
        default=DEFAULT_MODEL_PROFILE,
        help="architecture/checkpoint identity; full is the original 16-layer 512-dim model",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = MiniARCV12Config(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_steps_per_epoch=args.train_steps,
        validation_steps_per_epoch=args.validation_steps,
        max_epochs=args.max_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        seed=args.seed,
        num_workers=args.num_workers,
        checkpoint_every_steps=args.checkpoint_every_steps,
        refinement_ratio=args.refinement_ratio,
    )
    train(
        args.dataset_dir,
        args.checkpoint_dir,
        config,
        persistent_checkpoint_dir=args.persistent_checkpoint_dir,
        force_restart=args.force_restart,
        model_profile=args.model_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
