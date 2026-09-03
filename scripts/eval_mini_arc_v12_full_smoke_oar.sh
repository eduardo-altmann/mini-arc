#!/usr/bin/env bash
#OAR -n mini-arc-v12-full-eval-smoke
#OAR -O mini-arc-v12-full-eval-smoke-%jobid%.out
#OAR -E mini-arc-v12-full-eval-smoke-%jobid%.err

# Validate checkpoint loading, direct inference, and TTT on five ARC-AGI tasks.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_TASKS="${MAX_TASKS:-5}"
export TTT_EPOCHS="${TTT_EPOCHS:-2}"
export RESULTS_DIR="${RESULTS_DIR:-$HOME/arc-results/mini-arc-v12-full/smoke}"

exec "$script_dir/eval_mini_arc_v12_full_oar.sh"
