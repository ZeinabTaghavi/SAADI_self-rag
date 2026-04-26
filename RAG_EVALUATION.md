# Compact RAG Evaluation

Use `evaluate_rag_run.py` after a retrieval plus generation run has already finished. The script reads existing run artifacts and writes only compact metric files; it does not rerun retrieval or generation.

## One-command usage

Evaluate every completed run under `selfrag_<top_k>_runs/<dataset>/` and write results under `selfrag_<top_k>_evaluations/<dataset>/<run_name>/`.
By default this means top-10 runs and evaluations:

```bash
./run_all_rag_evaluations.sh
```

For top-5 runs and evaluations:

```bash
GENERATION_TOP_K=5 ./run_all_rag_evaluations.sh
```

or:

```bash
./run_all_rag_evaluations_top5.sh
```

The all-runs wrapper covers `loogle`, `narrativeqa`, `novelhopqa`, `qasper`, and `quality`. It skips incomplete run folders unless you set:

```bash
INCLUDE_INCOMPLETE=1 ./run_all_rag_evaluations.sh
```

To choose a different evaluation root:

```bash
EVALS_ROOT=/path/to/selfrag_10_evaluations ./run_all_rag_evaluations.sh
```

Evaluate one run directly:

```bash
./run_rag_evaluation.sh \
  --run-dir selfrag_10_runs/novelhopqa/novelhopqa_smoke \
  --labels-file selfrag_10_runs/novelhopqa/novelhopqa_smoke/selection/qa_entries.json \
  --output-dir selfrag_10_evaluations/novelhopqa/novelhopqa_smoke \
  --method-name selfrag \
  --dataset-name novelhopqa \
  --split test \
  --ks 5 10 \
  --generation-top-k 10 \
  --bert-score-model roberta-large \
  --bert-score-lang en
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

Generation metrics are computed from already-generated predictions. The manifest records `generation_top_k`; the evaluator does not invent generation@5 when the run generated from top-10 context. Keep top-5 and top-10 runs in separate roots such as `selfrag_5_runs` and `selfrag_10_runs`.

The RAG metric block reports:

- `exact_match`
- `token_f1`
- `rouge_l`
- `bertscore_precision`
- `bertscore_recall`
- `bertscore_f1`

BERTScore uses best-over-references selection per query, mirroring the existing EM/F1/ROUGE behavior. By default it uses `roberta-large` with `lang=en`. For quick smoke tests, disable it with:

```bash
DISABLE_BERT_SCORE=1 ./run_all_rag_evaluations.sh
```

If BERTScore is enabled, the evaluator now requires numeric BERTScore values instead of silently writing `null` when the dependency or model cannot be loaded. Install the dependency in the same Python environment used for evaluation:

```bash
python3 -m pip install 'bert-score>=0.3.13'
```

If you intentionally want the old soft-missing behavior, set:

```bash
ALLOW_MISSING_BERT_SCORE=1 ./run_all_rag_evaluations.sh
```

For GPU-controlled runs, set:

```bash
BERT_SCORE_DEVICE=cuda:0 BERT_SCORE_BATCH_SIZE=8 ./run_all_rag_evaluations.sh
```
