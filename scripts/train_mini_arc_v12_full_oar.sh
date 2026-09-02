#!/usr/bin/env bash
#OAR -n mini-arc-v12-full
#OAR -l host=1/gpu=8,walltime=8:00:00
#OAR -O mini-arc-v12-full-%jobid%.out
#OAR -E mini-arc-v12-full-%jobid%.err

# Wrapper for the original 16-layer, 512-dimensional mini-arc-v12.  Keep its
# persistent state distinct from the reduced baseline.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_PROFILE=full
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/arc-checkpoints/mini-arc-v12-full}"
# Keep unattended OAR submissions compatible with the established full-model
# checkpoint, even if the scheduler does not preserve submission-shell extras.
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-100}"
export MAX_EPOCHS="${MAX_EPOCHS:-100}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-200}"

exec "$script_dir/train_mini_arc_v12_oar.sh"
