#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default usage:
#   GPU_IDS=4,5,6,7 ./run_selfrag_multi_gpu.sh
#
# LooGLE one-document smoke test:
#   GPU_IDS=4,5,6,7 MAX_DOCS=1 QA_N=3 ./run_selfrag_multi_gpu.sh

DATASET_NAME="${DATASET_NAME:-loogle}"
DEFAULT_YAML="${DEFAULT_YAML:-$SCRIPT_DIR/configs/selfrag/loogle_selfrag.yaml}"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_selfrag_experiment.py}"

GPU_IDS="${GPU_IDS:-4,5,6,7}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
MODEL_NAME="${MODEL_NAME:-selfrag/selfrag_llama2_7b}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$SCRIPT_DIR/.cache}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_home}"

# Optional fast-test overrides.
RUN_NAME="${RUN_NAME:-}"
MAX_DOCS="${MAX_DOCS:-}"
QA_N="${QA_N:-}"
QA_SELECTION_METHOD="${QA_SELECTION_METHOD:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
NDOCS="${NDOCS:-}"
THRESHOLD="${THRESHOLD:-}"
MODE="${MODE:-}"
RESUME="${RESUME:-0}"

if [[ -z "${GPU_IDS// }" ]]; then
  echo "GPU_IDS is empty. Example: GPU_IDS=4,5,6,7 ./run_selfrag_multi_gpu.sh" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_YAML" ]]; then
  echo "DEFAULT_YAML does not exist: $DEFAULT_YAML" >&2
  exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "RUNNER does not exist: $RUNNER" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
TP_SIZE="${#GPU_ARRAY[@]}"
if [[ "$TP_SIZE" -lt 1 ]]; then
  echo "Could not parse any GPU ids from GPU_IDS=$GPU_IDS" >&2
  exit 1
fi

mkdir -p "$DOWNLOAD_DIR" "$HF_HOME" "$HF_HOME/hub" "$HF_HOME/transformers" "$SCRIPT_DIR/.tmp"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false

TMP_YAML="$SCRIPT_DIR/.tmp/${DATASET_NAME}_selfrag_runtime.yaml"

DEFAULT_YAML="$DEFAULT_YAML" \
TMP_YAML="$TMP_YAML" \
RUN_NAME="$RUN_NAME" \
MODEL_NAME="$MODEL_NAME" \
DTYPE="$DTYPE" \
TP_SIZE="$TP_SIZE" \
GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION" \
GPU_IDS="$GPU_IDS" \
DOWNLOAD_DIR="$DOWNLOAD_DIR" \
MAX_DOCS="$MAX_DOCS" \
QA_N="$QA_N" \
QA_SELECTION_METHOD="$QA_SELECTION_METHOD" \
MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
NDOCS="$NDOCS" \
THRESHOLD="$THRESHOLD" \
MODE="$MODE" \
python3 - <<'PY'
import os
from pathlib import Path
import yaml

default_yaml = Path(os.environ["DEFAULT_YAML"])
tmp_yaml = Path(os.environ["TMP_YAML"])

with open(default_yaml, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle) or {}

cfg.setdefault("dataset", {})
cfg.setdefault("model", {})
cfg.setdefault("generation", {})
cfg.setdefault("selfrag", {})

if os.environ["RUN_NAME"]:
    cfg["run_name"] = os.environ["RUN_NAME"]

cfg["model"]["name"] = os.environ["MODEL_NAME"]
cfg["model"]["dtype"] = os.environ["DTYPE"]
cfg["model"]["tensor_parallel_size"] = int(os.environ["TP_SIZE"])
cfg["model"]["gpu_memory_utilization"] = float(os.environ["GPU_MEMORY_UTILIZATION"])
cfg["model"]["cuda_visible_devices"] = os.environ["GPU_IDS"]
cfg["model"]["download_dir"] = os.environ["DOWNLOAD_DIR"]

if os.environ["MAX_DOCS"]:
    cfg["dataset"]["max_docs"] = int(os.environ["MAX_DOCS"])
if os.environ["QA_N"]:
    raw = os.environ["QA_N"].strip()
    cfg["dataset"]["qa_n"] = int(raw) if raw.isdigit() else raw
if os.environ["QA_SELECTION_METHOD"]:
    cfg["dataset"]["qa_selection_method"] = os.environ["QA_SELECTION_METHOD"]
if os.environ["MAX_NEW_TOKENS"]:
    cfg["generation"]["max_new_tokens"] = int(os.environ["MAX_NEW_TOKENS"])
if os.environ["NDOCS"]:
    ndocs = int(os.environ["NDOCS"])
    cfg["selfrag"]["ndocs"] = ndocs
    cfg.setdefault("retrieval", {})
    cfg["retrieval"]["retrieve_k"] = ndocs
if os.environ["THRESHOLD"]:
    cfg["selfrag"]["threshold"] = float(os.environ["THRESHOLD"])
if os.environ["MODE"]:
    cfg["selfrag"]["mode"] = os.environ["MODE"]

tmp_yaml.parent.mkdir(parents=True, exist_ok=True)
with open(tmp_yaml, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=False)
PY

CMD=(
  python3 "$RUNNER"
  --dataset-name "$DATASET_NAME"
  --default-yaml "$TMP_YAML"
)

if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
fi

echo "cwd=$SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "tensor_parallel_size=$TP_SIZE"
echo "DEFAULT_YAML=$DEFAULT_YAML"
echo "TMP_YAML=$TMP_YAML"
echo "DOWNLOAD_DIR=$DOWNLOAD_DIR"
echo "HF_HOME=$HF_HOME"
echo

"${CMD[@]}"
