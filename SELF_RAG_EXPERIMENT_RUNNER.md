# SELF-RAG Experiment Runner

This repository now includes a standalone experiment runner that stays inside the SELF-RAG project and writes only full RAG outputs plus supporting artifacts for later analysis.

## Entry point

```bash
python3 run_selfrag_experiment.py \
  --dataset-name YOUR_DATASET_NAME \
  --default-yaml /path/to/default_experiment.yaml
```

Optional:

```bash
python3 run_selfrag_experiment.py \
  --dataset-name YOUR_DATASET_NAME \
  --default-yaml /path/to/default_experiment.yaml \
  --resume
```

## Output layout

Each run is written to:

```text
selfrag_runs/<dataset_name>/<run_name>/
```

with these artifacts:

```text
config/default_experiment.yaml
config/selfrag_run.yaml
selection/selected_doc_ids.json
selection/qa_entries.json
corpus/documents.jsonl
rag/qa_predictions.jsonl
rag/rag_run_traces.jsonl
profiling/query_times.jsonl
profiling/resource_usage.jsonl
run_manifest.json
```

## YAML mapping

The runner resolves defaults from the input YAML as closely as practical. It looks for common fields under:

- `dataset.*` or `data.*`
- `model.*`
- `generation.*` or `decoding.*`
- `retrieval.*`
- `selfrag.*`
- `profiling.*`

Examples of supported dataset fields:

- `dataset.source_type`
- `dataset.source`
- `dataset.qa_path`
- `dataset.docs_path`
- `dataset.split`
- `dataset.question_field`
- `dataset.answers_field`
- `dataset.query_id_field`
- `dataset.doc_id_field`
- `dataset.contexts_field`
- `dataset.max_questions`
- `dataset.max_docs`
- `dataset.question_ids`
- `dataset.task`

Examples of supported model / runtime fields:

- `model.name`
- `model.tokenizer_name`
- `model.download_dir`
- `model.dtype`
- `model.tensor_parallel_size`
- `model.gpu_memory_utilization`
- `model.cuda_visible_devices`

Examples of supported retrieval fields:

- `retrieval.backend`
- `retrieval.model_name_or_path`
- `retrieval.passages`
- `retrieval.passages_embeddings`
- `retrieval.n_docs`

Supported retrieval backends:

- `provided_contexts`: use contexts already stored on each QA entry, such as `ctxs`, `top_contexts`, `docs`, or the configured `dataset.contexts_field`
- `contriever`: use the in-repo Contriever retriever from `retrieval_lm/passage_retrieval.py`

## LooGLE

This repo now includes a native `loogle` dataset loader for the standalone runner, based on the same loading shape as your external loader but implemented locally inside SELF-RAG.

Reference config:

`[configs/selfrag/loogle_selfrag.yaml](/Users/hslu-n0008110/Library/CloudStorage/OneDrive-HochschuleLuzern/Desktop/SAADI_self-rag/configs/selfrag/loogle_selfrag.yaml)`

Run it like this:

```bash
python3 run_selfrag_experiment.py \
  --dataset-name loogle \
  --default-yaml /Users/hslu-n0008110/Library/CloudStorage/OneDrive-HochschuleLuzern/Desktop/SAADI_self-rag/configs/selfrag/loogle_selfrag.yaml
```

Fields intentionally carried over from your external `loogle_retrieval_ablation.yaml`:

- `dataset.split`
- `dataset.config_name`
- `dataset.qa_n`
- `dataset.qa_selection_method`
- `sample.max_documents` mapped to `dataset.max_docs`
- `retrieval.retrieve_k` mapped to `selfrag.ndocs`
- `model.tensor_parallel_size`
- `model.dtype`
- `model.gpu_memory_utilization`

Fields intentionally omitted because they do not directly drive this SELF-RAG runner:

- `ingest.*`
- `retrieval.retriever`
- `retrieval.mode`
- `retrieval.scope`
- `evaluation.*`
- `model.backend`
- `model.alias`
- chat-specific stop strings and unrelated sampling fields

## Important notes

- No separate retrieval evaluation artifacts are written.
- Internal retrieval behavior is stored inside `rag/rag_run_traces.jsonl`.
- The runner records YAML fields that did not map cleanly in `run_manifest.json`.
- The current implementation uses a marker-based scoring path for compatibility with newer `vllm` builds that restrict very large `logprobs` requests.
- `short_form` is the primary supported SELF-RAG mode. A simplified `long_form` mapping is supported through the same retrieval-conditioned generation path, and any such mapping choice is recorded in the manifest notes.
