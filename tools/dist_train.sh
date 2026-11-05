#!/usr/bin/env bash
set -euo pipefail

CONFIG=$1
GPUS=$2
shift 2
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec torchrun \
    --nnodes="${NNODES:-1}" \
    --node_rank="${NODE_RANK:-0}" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${PORT:-29500}" \
    --nproc_per_node="$GPUS" \
    "$SCRIPT_DIR/train.py" "$CONFIG" --launcher pytorch "$@"
