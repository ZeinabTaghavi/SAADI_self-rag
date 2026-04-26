#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GENERATION_TOP_K="${GENERATION_TOP_K:-${TOP_K:-5}}" "$SCRIPT_DIR/run_all_rag_evaluations.sh" "$@"
