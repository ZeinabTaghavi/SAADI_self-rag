#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_NAME="${METHOD_NAME:-selfrag}"
GENERATION_TOP_K="${GENERATION_TOP_K:-${TOP_K:-10}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/selfrag_${GENERATION_TOP_K}_runs}"
EVALS_ROOT="${EVALS_ROOT:-${ROOT_DIR}/selfrag_${GENERATION_TOP_K}_evaluations}"
INCLUDE_INCOMPLETE="${INCLUDE_INCOMPLETE:-0}"
DISABLE_BERT_SCORE="${DISABLE_BERT_SCORE:-0}"
ALLOW_MISSING_BERT_SCORE="${ALLOW_MISSING_BERT_SCORE:-0}"
BERT_SCORE_MODEL="${BERT_SCORE_MODEL:-roberta-large}"
BERT_SCORE_LANG="${BERT_SCORE_LANG:-en}"
BERT_SCORE_BATCH_SIZE="${BERT_SCORE_BATCH_SIZE:-16}"
BERT_SCORE_DEVICE="${BERT_SCORE_DEVICE:-}"
BERT_SCORE_RESCALE_WITH_BASELINE="${BERT_SCORE_RESCALE_WITH_BASELINE:-0}"

DATASETS=(
  "loogle:test"
  "narrativeqa:test"
  "novelhopqa:test"
  "qasper:test"
  "quality:validation"
)

usage() {
  cat <<'EOF'
Run compact RAG evaluation for every completed run under selfrag_<k>_runs/<dataset>/.

Defaults:
  GENERATION_TOP_K=10
  RUNS_ROOT=./selfrag_10_runs
  EVALS_ROOT=./selfrag_10_evaluations
  METHOD_NAME=selfrag
  BERT_SCORE_MODEL=roberta-large

Examples:
  ./run_all_rag_evaluations.sh

  GENERATION_TOP_K=5 ./run_all_rag_evaluations.sh

  RUNS_ROOT=/path/to/selfrag_5_runs EVALS_ROOT=/path/to/selfrag_5_evaluations METHOD_NAME=selfrag GENERATION_TOP_K=5 ./run_all_rag_evaluations.sh

  INCLUDE_INCOMPLETE=1 ./run_all_rag_evaluations.sh

  DISABLE_BERT_SCORE=1 ./run_all_rag_evaluations.sh

  ALLOW_MISSING_BERT_SCORE=1 ./run_all_rag_evaluations.sh

  BERT_SCORE_DEVICE=cuda:0 BERT_SCORE_BATCH_SIZE=8 ./run_all_rag_evaluations.sh

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
  echo "[info] creating empty runs root at ${RUNS_ROOT}"
fi
mkdir -p "${RUNS_ROOT}" "${EVALS_ROOT}"

bert_score_args=(
  --bert-score-model "${BERT_SCORE_MODEL}"
  --bert-score-lang "${BERT_SCORE_LANG}"
  --bert-score-batch-size "${BERT_SCORE_BATCH_SIZE}"
)
if [[ -n "${BERT_SCORE_DEVICE//[[:space:]]/}" ]]; then
  bert_score_args+=(--bert-score-device "${BERT_SCORE_DEVICE}")
fi
if [[ "${BERT_SCORE_RESCALE_WITH_BASELINE}" == "1" ]]; then
  bert_score_args+=(--bert-score-rescale-with-baseline)
fi
if [[ "${DISABLE_BERT_SCORE}" == "1" ]]; then
  bert_score_args=(--disable-bert-score)
elif [[ "${ALLOW_MISSING_BERT_SCORE}" == "1" ]]; then
  bert_score_args+=(--allow-missing-bert-score)
else
  if ! python3 -c 'import bert_score' >/dev/null 2>&1; then
    echo "BERTScore is enabled, but the active python3 cannot import bert_score." >&2
    echo "Install it in this environment with: python3 -m pip install 'bert-score>=0.3.13'" >&2
    echo "For smoke runs, set DISABLE_BERT_SCORE=1. To keep null BERTScore fields, set ALLOW_MISSING_BERT_SCORE=1." >&2
    exit 1
  fi
fi

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
      --generation-top-k "${GENERATION_TOP_K}" \
      "${bert_score_args[@]}"

    evaluated=$((evaluated + 1))
  done < <(find "${dataset_dir}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

  if [[ "${found_dataset_run}" == "0" ]]; then
    echo "[skip] ${dataset}: no run directories under ${dataset_dir}"
    skipped=$((skipped + 1))
  fi
done

echo "Done. evaluated=${evaluated} skipped=${skipped}"
