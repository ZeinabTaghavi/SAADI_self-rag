#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/selfrag_runs}"
EVALS_ROOT="${EVALS_ROOT:-${ROOT_DIR}/selfRAG_evaluations}"
METHOD_NAME="${METHOD_NAME:-selfrag}"
GENERATION_TOP_K="${GENERATION_TOP_K:-10}"
INCLUDE_INCOMPLETE="${INCLUDE_INCOMPLETE:-0}"

DATASETS=(
  "loogle:test"
  "narrativeqa:test"
  "novelhopqa:test"
  "qasper:test"
  "quality:validation"
)

usage() {
  cat <<'EOF'
Run compact RAG evaluation for every completed run under selfrag_runs/<dataset>/.

Defaults:
  RUNS_ROOT=./selfrag_runs
  EVALS_ROOT=./selfRAG_evaluations
  METHOD_NAME=selfrag
  GENERATION_TOP_K=10

Examples:
  ./run_all_rag_evaluations.sh

  RUNS_ROOT=/path/to/selfrag_runs EVALS_ROOT=/path/to/selfRAG_evaluations METHOD_NAME=selfrag ./run_all_rag_evaluations.sh

  INCLUDE_INCOMPLETE=1 ./run_all_rag_evaluations.sh

Notes:
  By default, incomplete run directories are skipped when rag/qa_predictions.jsonl
  is missing. Set INCLUDE_INCOMPLETE=1 to run the evaluator anyway and let the
  manifest document missing metrics.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "${RUNS_ROOT}" ]]; then
  echo "No runs root found at ${RUNS_ROOT}" >&2
  exit 1
fi

mkdir -p "${EVALS_ROOT}"

evaluated=0
skipped=0

for dataset_entry in "${DATASETS[@]}"; do
  dataset="${dataset_entry%%:*}"
  split="${dataset_entry#*:}"
  dataset_dir="${RUNS_ROOT}/${dataset}"
  mkdir -p "${EVALS_ROOT}/${dataset}"

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[skip] ${dataset}: no directory at ${dataset_dir}"
    skipped=$((skipped + 1))
    continue
  fi

  found_dataset_run=0
  while IFS= read -r -d '' run_dir; do
    found_dataset_run=1
    run_name="$(basename "${run_dir}")"
    predictions_file="${run_dir}/rag/qa_predictions.jsonl"

    if [[ "${INCLUDE_INCOMPLETE}" != "1" && ! -f "${predictions_file}" ]]; then
      echo "[skip] ${dataset}/${run_name}: missing rag/qa_predictions.jsonl"
      skipped=$((skipped + 1))
      continue
    fi

    output_dir="${EVALS_ROOT}/${dataset}/${run_name}"
    echo "[eval] ${dataset}/${run_name} -> ${output_dir}"

    python3 "${ROOT_DIR}/evaluate_rag_run.py" \
      --run-dir "${run_dir}" \
      --output-dir "${output_dir}" \
      --method-name "${METHOD_NAME}" \
      --dataset-name "${dataset}" \
      --split "${split}" \
      --ks 5 10 \
      --generation-top-k "${GENERATION_TOP_K}"

    evaluated=$((evaluated + 1))
  done < <(find "${dataset_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if [[ "${found_dataset_run}" == "0" ]]; then
    echo "[skip] ${dataset}: no run directories under ${dataset_dir}"
    skipped=$((skipped + 1))
  fi
done

echo "Done. evaluated=${evaluated} skipped=${skipped}"
