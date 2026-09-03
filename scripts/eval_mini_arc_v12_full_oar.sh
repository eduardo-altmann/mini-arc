#!/usr/bin/env bash
#OAR -n mini-arc-v12-full-eval
#OAR -O mini-arc-v12-full-eval-%jobid%.out
#OAR -E mini-arc-v12-full-eval-%jobid%.err

# Evaluate the full checkpoint on ARC-AGI using one visible GPU. Resource and
# time selection are intentionally left to the oarsub command.
set -euo pipefail

ARC_WORKSPACE="${ARC_WORKSPACE:-$HOME/arc-agi}"
MINI_ARC_SOURCE="${MINI_ARC_SOURCE:-$ARC_WORKSPACE/mini-arc}"
APPTAINER_IMAGE="${APPTAINER_IMAGE:-$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HOME/arc-checkpoints/mini-arc-v12-full/best.pt}"
RESULTS_DIR="${RESULTS_DIR:-$HOME/arc-results/mini-arc-v12-full}"
CHALLENGES_PATH="${CHALLENGES_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_challenges.json}"
SOLUTIONS_PATH="${SOLUTIONS_PATH:-$MINI_ARC_SOURCE/data/arc-agi_evaluation_solutions.json}"
JOB_ID="${OAR_JOB_ID:-manual-$$}"
JOB_ROOT="/tmp/$USER/mini-arc-v12-eval-$JOB_ID"
LOCAL_IMAGE="$JOB_ROOT/mini-arc-v12.sif"
LOCAL_CHECKPOINT="$JOB_ROOT/checkpoint.pt"
LOCAL_RESULTS="$JOB_ROOT/results"

case "$JOB_ROOT" in
    "/tmp/$USER/mini-arc-v12-eval-"*) ;;
    *) echo "Refusing unsafe job directory: $JOB_ROOT" >&2; exit 2 ;;
esac

for path in "$MINI_ARC_SOURCE/arc_prize/eval_arc_agi.py" "$APPTAINER_IMAGE" "$CHECKPOINT_PATH" "$CHALLENGES_PATH" "$SOLUTIONS_PATH"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 2
    fi
done
if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on this node" >&2
    exit 2
fi

mkdir -p "$JOB_ROOT/mini-arc" "$JOB_ROOT/data" "$LOCAL_RESULTS" "$RESULTS_DIR"
rsync -a \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='data/re_arc_5k_12x12/' \
    --exclude='data/re_arc_5k_12x12_balanced/' \
    "$MINI_ARC_SOURCE/" "$JOB_ROOT/mini-arc/"
rsync -a "$APPTAINER_IMAGE" "$LOCAL_IMAGE"
rsync -a "$CHECKPOINT_PATH" "$LOCAL_CHECKPOINT"
rsync -a "$CHALLENGES_PATH" "$JOB_ROOT/data/challenges.json"
rsync -a "$SOLUTIONS_PATH" "$JOB_ROOT/data/solutions.json"

CONTAINER=(
    apptainer exec --nv
    --bind "$JOB_ROOT:$JOB_ROOT"
    --bind "$HOME:$HOME"
    "$LOCAL_IMAGE"
)

GPU_COUNT="$("${CONTAINER[@]}" python -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPU_COUNT" -lt 1 ]]; then
    echo "No CUDA device is visible inside the allocation" >&2
    exit 3
fi

echo "Job ID: $JOB_ID"
echo "Host: $(hostname)"
echo "Visible GPUs: $GPU_COUNT (evaluation uses cuda:0 only)"
echo "Checkpoint: $CHECKPOINT_PATH"
nvidia-smi

cd "$JOB_ROOT/mini-arc"
EVAL_ARGS=(
    --checkpoint "$LOCAL_CHECKPOINT"
    --challenges "$JOB_ROOT/data/challenges.json"
    --solutions "$JOB_ROOT/data/solutions.json"
    --output "$LOCAL_RESULTS/arc-agi-evaluation.json"
    --ttt-epochs "${TTT_EPOCHS:-2}"
    --ttt-learning-rate "${TTT_LEARNING_RATE:-0.00001}"
    --ttt-weight-decay "${TTT_WEIGHT_DECAY:-0.00001}"
    --ttt-batch-size "${TTT_BATCH_SIZE:-4}"
)
if [[ -n "${MAX_TASKS:-}" ]]; then
    EVAL_ARGS+=(--max-tasks "$MAX_TASKS")
fi
"${CONTAINER[@]}" python -m arc_prize.eval_arc_agi "${EVAL_ARGS[@]}"

rsync -a "$LOCAL_RESULTS/" "$RESULTS_DIR/"
rm -rf -- "$JOB_ROOT"
echo "Evaluation finished; report: $RESULTS_DIR/arc-agi-evaluation.json"
