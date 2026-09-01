#!/usr/bin/env bash
#OAR -n mini-arc-v12
#OAR -l host=1/gpu=2,walltime=4:00:00
#OAR -O mini-arc-v12-%jobid%.out
#OAR -E mini-arc-v12-%jobid%.err

set -euo pipefail

ARC_WORKSPACE="${ARC_WORKSPACE:-$HOME/arc-agi}"
MINI_ARC_SOURCE="${MINI_ARC_SOURCE:-$ARC_WORKSPACE/mini-arc}"
REARC_SOURCE="${REARC_SOURCE:-$ARC_WORKSPACE/re-arc/re_arc_5k_12x12}"
PREPARED_DATASET="${PREPARED_DATASET:-$HOME/arc-data/re_arc_5k_12x12_balanced}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/arc-checkpoints/mini-arc-v12}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
JOB_ID="${OAR_JOB_ID:-manual-$$}"
JOB_ROOT="/tmp/$USER/mini-arc-v12-$JOB_ID"

case "$JOB_ROOT" in
    "/tmp/$USER/mini-arc-v12-"*) ;;
    *) echo "Refusing unsafe job directory: $JOB_ROOT" >&2; exit 2 ;;
esac

if [[ ! -d "$MINI_ARC_SOURCE/arc_prize" ]]; then
    echo "mini-arc source not found: $MINI_ARC_SOURCE" >&2
    exit 2
fi
if [[ ! -d "$REARC_SOURCE/tasks" && ! -f "$PREPARED_DATASET/manifest.json" ]]; then
    echo "Neither raw nor prepared RE-ARC data was found" >&2
    exit 2
fi

mkdir -p "$JOB_ROOT/mini-arc" "$JOB_ROOT/data" "$CHECKPOINT_DIR"
rsync -a --exclude='.git/' --exclude='__pycache__/' "$MINI_ARC_SOURCE/" "$JOB_ROOT/mini-arc/"

cd "$JOB_ROOT/mini-arc"

if [[ ! -f "$PREPARED_DATASET/manifest.json" || ! -f "$PREPARED_DATASET/examples.sqlite3" ]]; then
    echo "Preparing persistent RE-ARC dataset in $PREPARED_DATASET"
    "$PYTHON_BIN" -m arc_prize.rearc_manifest \
        --source "$REARC_SOURCE" \
        --output "$PREPARED_DATASET"
fi
rsync -a "$PREPARED_DATASET/" "$JOB_ROOT/data/"

NPROC="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$NPROC" -lt 1 ]]; then
    echo "No CUDA devices are visible inside the OAR allocation" >&2
    exit 3
fi

echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "GPUs: $NPROC"
echo "Scratch: $JOB_ROOT"
echo "Checkpoints: $CHECKPOINT_DIR"
nvidia-smi

trainer_pid=""
forward_termination() {
    if [[ -n "$trainer_pid" ]]; then
        kill -TERM "$trainer_pid" 2>/dev/null || true
        wait "$trainer_pid" || true
    fi
    exit 143
}
trap forward_termination TERM INT

TRAIN_ARGS=(
    --dataset-dir "$JOB_ROOT/data"
    --checkpoint-dir "$CHECKPOINT_DIR"
    --batch-size "${BATCH_SIZE:-32}"
    --learning-rate "${LEARNING_RATE:-0.0001}"
    --weight-decay "${WEIGHT_DECAY:-0.0001}"
    --train-steps "${TRAIN_STEPS:-1000}"
    --validation-steps "${VALIDATION_STEPS:-100}"
    --max-epochs "${MAX_EPOCHS:-100}"
    --patience "${PATIENCE:-15}"
    --seed "${SEED:-42}"
    --num-workers "${NUM_WORKERS:-0}"
    --checkpoint-every-steps "${CHECKPOINT_EVERY_STEPS:-0}"
)
if [[ "${FORCE_RESTART:-0}" == "1" ]]; then
    TRAIN_ARGS+=(--force-restart)
fi

"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC" \
    -m arc_prize.train_rearc "${TRAIN_ARGS[@]}" &
trainer_pid=$!
wait "$trainer_pid"
trainer_pid=""

if [[ ! -f "$CHECKPOINT_DIR/latest.pt" ]]; then
    echo "Training exited without a persistent latest checkpoint; preserving scratch" >&2
    exit 4
fi

rm -rf -- "$JOB_ROOT"
echo "Training finished; removed job-specific scratch directory $JOB_ROOT"
