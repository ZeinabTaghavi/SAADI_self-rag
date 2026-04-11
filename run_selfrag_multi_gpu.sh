#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure these before running, or override them inline:
#   GPU_IDS=4,5,6,7 QUERY="..." DOCUMENT_FILE=sample_doc.txt ./run_selfrag_multi_gpu.sh
GPU_IDS="${GPU_IDS:-4,5,6,7}"
MODEL_NAME="${MODEL_NAME:-selfrag/selfrag_llama2_7b}"
QUERY="${QUERY:-What is the difference between llamas and alpacas?}"
DOCUMENT_FILE="${DOCUMENT_FILE:-}"
DOCUMENT_TEXT="${DOCUMENT_TEXT:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-half}"
CHUNK_SIZE="${CHUNK_SIZE:-120}"
CHUNK_OVERLAP="${CHUNK_OVERLAP:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-80}"
THRESHOLD="${THRESHOLD:-0.4}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$SCRIPT_DIR/.cache}"
HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_home}"

if [[ -z "${GPU_IDS// }" ]]; then
  echo "GPU_IDS is empty. Example: GPU_IDS=4,5,6,7 ./run_selfrag_multi_gpu.sh" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
TP_SIZE="${#GPU_ARRAY[@]}"

mkdir -p "$DOWNLOAD_DIR" "$HF_HOME"
mkdir -p "$HF_HOME/hub" "$HF_HOME/transformers"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TOKENIZERS_PARALLELISM=false

CMD=(
  python3 "$SCRIPT_DIR/test.py"
  --model-name "$MODEL_NAME"
  --query "$QUERY"
  --cuda-visible-devices "$GPU_IDS"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
  --chunk-size "$CHUNK_SIZE"
  --chunk-overlap "$CHUNK_OVERLAP"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --threshold "$THRESHOLD"
  --download-dir "$DOWNLOAD_DIR"
)

if [[ -n "$DOCUMENT_FILE" ]]; then
  CMD+=(--document-file "$DOCUMENT_FILE")
elif [[ -n "$DOCUMENT_TEXT" ]]; then
  CMD+=(--document "$DOCUMENT_TEXT")
fi

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "tensor_parallel_size=$TP_SIZE"
echo "download_dir=$DOWNLOAD_DIR"
echo "HF_HOME=$HF_HOME"
echo

"${CMD[@]}"
