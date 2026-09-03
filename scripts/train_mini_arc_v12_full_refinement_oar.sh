#!/usr/bin/env bash
#OAR -n mini-arc-v12-full-refinement
#OAR -l host=1/gpu=8,walltime=8:00:00
#OAR -O mini-arc-v12-full-refinement-%jobid%.out
#OAR -E mini-arc-v12-full-refinement-%jobid%.err

# Train the paper-style full model with 25% noisy-target refinement steps.
# This uses a separate checkpoint identity and never overwrites the direct-only run.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PROFILE=full-refinement
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/arc-checkpoints/mini-arc-v12-full-refinement}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-100}"
export MAX_EPOCHS="${MAX_EPOCHS:-150}"
export PATIENCE="${PATIENCE:-150}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-0}"
export REFINEMENT_RATIO="${REFINEMENT_RATIO:-0.25}"

exec "$script_dir/train_mini_arc_v12_oar.sh"
