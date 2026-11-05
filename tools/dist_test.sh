#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${PORT:-29500}" \
    --nproc_per_node="$GPUS" \
    "$SCRIPT_DIR/test.py" "$CONFIG" "$CHECKPOINT" --launcher pytorch "$@"
