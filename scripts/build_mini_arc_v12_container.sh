#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_IMAGE="${1:-$HOME/containers/mini-arc-v12-pytorch2.4.1-cuda12.1.sif}"
OUTPUT_DIR="$(dirname "$OUTPUT_IMAGE")"
TEMP_IMAGE="$OUTPUT_DIR/.mini-arc-v12-$USER-$$.sif"
CACHE_DIR=""

if ! command -v apptainer >/dev/null 2>&1; then
    echo "apptainer is not available on this machine" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
if [[ -e "$OUTPUT_IMAGE" ]]; then
    echo "Refusing to overwrite existing image: $OUTPUT_IMAGE" >&2
    exit 2
fi

if [[ -n "${OAR_JOB_ID:-}" ]]; then
    CACHE_DIR="/tmp/$USER/mini-arc-v12-apptainer-$OAR_JOB_ID"
    mkdir -p "$CACHE_DIR/cache" "$CACHE_DIR/tmp"
    export APPTAINER_CACHEDIR="$CACHE_DIR/cache"
    export APPTAINER_TMPDIR="$CACHE_DIR/tmp"
fi

cleanup() {
    if [[ -e "$TEMP_IMAGE" ]]; then
        rm -f -- "$TEMP_IMAGE"
    fi
    if [[ -n "$CACHE_DIR" && "$CACHE_DIR" == "/tmp/$USER/mini-arc-v12-apptainer-"* ]]; then
        rm -rf -- "$CACHE_DIR"
    fi
}
trap cleanup EXIT

apptainer build "$TEMP_IMAGE" "$REPO_ROOT/containers/mini-arc-v12.def"
apptainer exec "$TEMP_IMAGE" python -c \
    'import sqlite3, torch; print("torch", torch.__version__, "cuda build", torch.version.cuda)'
mv "$TEMP_IMAGE" "$OUTPUT_IMAGE"
cleanup
trap - EXIT

echo "Built $OUTPUT_IMAGE"
echo "On the GPU node, verify with:"
echo "  apptainer exec --nv $OUTPUT_IMAGE python -c 'import torch; print(torch.cuda.device_count())'"
