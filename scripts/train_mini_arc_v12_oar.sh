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
PERSISTENT_CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/arc-checkpoints/mini-arc-v12}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif}"
JOB_ID="${OAR_JOB_ID:-manual-$$}"
JOB_ROOT="/tmp/$USER/mini-arc-v12-$JOB_ID"
LOCAL_IMAGE="$JOB_ROOT/mini-arc-v12.sif"
LOCAL_CHECKPOINT_DIR="$JOB_ROOT/checkpoints"

case "$JOB_ROOT" in
    "/tmp/$USER/mini-arc-v12-"*) ;;
    *) echo "Refusing unsafe job directory: $JOB_ROOT" >&2; exit 2 ;;
esac

if [[ ! -d "$MINI_ARC_SOURCE/arc_prize" ]]; then
    echo "mini-arc source not found: $MINI_ARC_SOURCE" >&2
    exit 2
fi
if [[ ! -f "$APPTAINER_IMAGE" ]]; then
    echo "Apptainer image not found: $APPTAINER_IMAGE" >&2
    exit 2
fi
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on this node" >&2
    exit 2
fi
if [[ ! -d "$REARC_SOURCE/tasks" && ! -f "$PREPARED_DATASET/manifest.json" ]]; then
    echo "Neither raw nor prepared RE-ARC data was found" >&2
    exit 2
fi
if [[ -e "$PREPARED_DATASET/manifest.json" && ! -e "$PREPARED_DATASET/examples.sqlite3" ]] ||
   [[ ! -e "$PREPARED_DATASET/manifest.json" && -e "$PREPARED_DATASET/examples.sqlite3" ]]; then
    echo "Prepared dataset is incomplete: $PREPARED_DATASET" >&2
    exit 2
fi

mkdir -p \
    "$JOB_ROOT/mini-arc" \
    "$JOB_ROOT/data" \
    "$LOCAL_CHECKPOINT_DIR" \
    "$PERSISTENT_CHECKPOINT_DIR" \
    "$PREPARED_DATASET"
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='data/re_arc_5k_12x12/' \
    --exclude='data/re_arc_5k_12x12_balanced/' \
    "$MINI_ARC_SOURCE/" "$JOB_ROOT/mini-arc/"
rsync -a --info=progress2 "$APPTAINER_IMAGE" "$LOCAL_IMAGE"

CONTAINER=(
    apptainer exec
    --nv
    --bind "$JOB_ROOT:$JOB_ROOT"
    --bind "$HOME:$HOME"
    "$LOCAL_IMAGE"
)

cd "$JOB_ROOT/mini-arc"

if [[ ! -f "$PREPARED_DATASET/manifest.json" || ! -f "$PREPARED_DATASET/examples.sqlite3" ]]; then
    echo "Preparing persistent RE-ARC dataset in $PREPARED_DATASET"
    "${CONTAINER[@]}" python -m arc_prize.rearc_manifest \
        --source "$REARC_SOURCE" \
        --output "$PREPARED_DATASET"
fi
rsync -a "$PREPARED_DATASET/" "$JOB_ROOT/data/"
if [[ "${FORCE_RESTART:-0}" != "1" ]]; then
    rsync -a \
        --include='latest.pt' \
        --include='best.pt' \
        --exclude='*' \
        "$PERSISTENT_CHECKPOINT_DIR/" "$LOCAL_CHECKPOINT_DIR/"
fi

NPROC="$("${CONTAINER[@]}" python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$NPROC" -lt 1 ]]; then
    echo "No CUDA devices are visible inside the OAR allocation" >&2
    exit 3
fi

echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "GPUs: $NPROC"
echo "Scratch: $JOB_ROOT"
echo "Local checkpoints: $LOCAL_CHECKPOINT_DIR"
echo "Persistent checkpoint mirror: $PERSISTENT_CHECKPOINT_DIR"
nvidia-smi
"${CONTAINER[@]}" python -c \
    'import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())'

trainer_pid=""
sync_results() {
    if [[ -d "$LOCAL_CHECKPOINT_DIR" ]]; then
        rsync -a "$LOCAL_CHECKPOINT_DIR/" "$PERSISTENT_CHECKPOINT_DIR/"
    fi
}
forward_termination() {
    if [[ -n "$trainer_pid" ]]; then
        kill -TERM "$trainer_pid" 2>/dev/null || true
        wait "$trainer_pid" || true
    fi
    sync_results
    exit 143
}
trap forward_termination TERM INT

TRAIN_ARGS=(
    --dataset-dir "$JOB_ROOT/data"
    --checkpoint-dir "$LOCAL_CHECKPOINT_DIR"
    --persistent-checkpoint-dir "$PERSISTENT_CHECKPOINT_DIR"
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

"${CONTAINER[@]}" python -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    -m arc_prize.train_rearc "${TRAIN_ARGS[@]}" &
trainer_pid=$!
wait "$trainer_pid"
trainer_pid=""
sync_results

if [[ ! -f "$PERSISTENT_CHECKPOINT_DIR/latest.pt" ]]; then
    echo "Training exited without a persistent latest checkpoint; preserving scratch" >&2
    exit 4
fi

rm -rf -- "$JOB_ROOT"
echo "Training finished; removed job-specific scratch directory $JOB_ROOT"
