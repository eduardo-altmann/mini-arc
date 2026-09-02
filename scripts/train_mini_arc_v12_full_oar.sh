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

exec "$script_dir/train_mini_arc_v12_oar.sh"
