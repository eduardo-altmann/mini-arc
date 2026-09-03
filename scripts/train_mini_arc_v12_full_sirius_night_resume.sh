#!/usr/bin/env bash
#OAR -n mini-arc-v12-full-resume
#OAR -q default
#OAR -t exotic
#OAR -t night
#OAR -p host='sirius-1.lyon.grid5000.fr'
#OAR -l host=1/gpu=8
#OAR -r "2026-09-03 19:05:00,2026-09-04 08:55:00"
#OAR -O mini-arc-v12-full-resume-%jobid%.out
#OAR -E mini-arc-v12-full-resume-%jobid%.err

# Resume the existing full-model checkpoint during one policy-compliant night.
# The generic launcher stages to /tmp and persists checkpoints in $HOME.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$HOME/arc-checkpoints/mini-arc-v12-full}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export VALIDATION_STEPS="${VALIDATION_STEPS:-100}"
export MAX_EPOCHS="${MAX_EPOCHS:-100}"
export PATIENCE="${PATIENCE:-100}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-200}"

exec "$script_dir/train_mini_arc_v12_full_oar.sh"
