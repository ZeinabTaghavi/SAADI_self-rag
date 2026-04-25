# Compact RAG Evaluation

Use `evaluate_rag_run.py` after a retrieval plus generation run has already finished. The script reads existing run artifacts and writes only compact metric files; it does not rerun retrieval or generation.

## One-command usage

Evaluate every completed run under `selfrag_runs/<dataset>/`:

```bash
./run_all_rag_evaluations.sh
```

The all-runs wrapper covers `loogle`, `narrativeqa`, `novelhopqa`, `qasper`, and `quality`. It skips incomplete run folders unless you set:

```bash
INCLUDE_INCOMPLETE=1 ./run_all_rag_evaluations.sh
```

Evaluate one run directly:

```bash
./run_rag_evaluation.sh \
  --run-dir selfrag_runs/novelhopqa/novelhopqa_smoke \
  --labels-file selfrag_runs/novelhopqa/novelhopqa_smoke/selection/qa_entries.json \
  --output-dir selfrag_runs/novelhopqa/novelhopqa_smoke/evaluation \
  --method-name selfrag \
  --dataset-name novelhopqa \
  --split test \
  --ks 5 10 \
  --generation-top-k 10
```

`--labels-file` may be omitted when the run has `selection/qa_entries.json`. Use `--answers-file` when reference answers live separately from retrieval labels.

## Inputs Read

By default, the evaluator looks inside `--run-dir` for:

- generation predictions: `rag/qa_predictions.jsonl`
- ranked retrieval outputs: `retrieval/retrieval_outputs.jsonl`, `retrieval/retrieval_results.jsonl`, `rag/retrieval_outputs.jsonl`, or `rag/retrieval_results.jsonl`
- fallback traces: `rag/rag_run_traces.jsonl`
- labels: `selection/qa_entries.json`
- resource usage: `profiling/resource_usage.jsonl`

If your run uses different filenames, pass `--predictions-file`, `--retrieval-file`, or `--traces-file` explicitly. Retrieval metrics preserve the ranked order in the selected retrieval source and deduplicate retrieved ids by first occurrence before scoring.

## Labels

The preferred relevance fields are:

- `gold_chunk_ids`
- `silver_chunk_ids`
- `silver_chunk_groups`

Metrics are computed separately for `gold`, `silver_loose`, `silver_strict`, and `union` views. The main `retrieval_metrics` block uses one primary relevance definition: `gold` if available, otherwise `silver_loose`, otherwise `silver_strict`.

For current in-repo runner outputs that only contain `doc_id`, the evaluator uses `doc_id` as a gold relevance id and records that assumption in `evaluation_manifest.json`. Add `--disable-doc-id-label-fallback` to require explicit gold/silver fields.

## Outputs

The output directory contains:

- `metrics_summary.json`
- `metrics_per_query.jsonl`
- `leaderboard_row.json`
- `evaluation_manifest.json`

Generation metrics are computed from already-generated predictions. The manifest records `generation_top_k = 10`; the evaluator does not invent generation@5 when the run generated from top-10 context.
