#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_K="${TOP_K:-5}" DATASET_NAME="narrativeqa" "$SCRIPT_DIR/run_selfrag_multi_gpu.sh" "$@"
