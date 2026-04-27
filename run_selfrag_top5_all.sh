#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SCRIPT_NAME="$(basename "$0" .sh)"
LOG_DIR="${ROOT_DIR}/scripts/logs"
LOG_FILE="${LOG_DIR}/${SCRIPT_NAME}.log"
mkdir -p "${LOG_DIR}"
if [[ "${SAADI_SCRIPT_LOGGING:-0}" != "1" ]]; then
  set +e
  SAADI_SCRIPT_LOGGING=1 bash "${SCRIPT_PATH}" "$@" 2>&1 | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  set -e
  exit "${status}"
fi

export PYTHONPATH="${PYTHONPATH:-${ROOT_DIR}/src}"
HF_CACHE_ROOT="${SAADI_HF_CACHE_ROOT:-/mnt/cache/taghavi}"
export HF_HOME="${HF_HOME:-${HF_CACHE_ROOT}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"


TOP_K="${TOP_K:-5}" \
DATASET_SEQUENCE="${DATASET_SEQUENCE:-loogle,qasper,quality,narrativeqa,novelhopqa}" \
"$SCRIPT_DIR/run_selfrag_multi_gpu.sh" "$@"
