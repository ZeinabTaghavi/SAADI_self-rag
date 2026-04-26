#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METRIC_KEYS = ("recall", "mrr", "ndcg", "hit_rate")
BERTSCORE_KEYS = ("bertscore_precision", "bertscore_recall", "bertscore_f1")
NO_BERTSCORE_PAIRS_ERROR = "No non-empty prediction/reference pairs were available for BERTScore."
EFFICIENCY_FIELDS = (
    "retrieval_latency_ms",
    "generation_latency_ms",
    "total_latency_ms",
    "context_tokens",
    "answer_tokens",
    "peak_gpu_memory_mb",
)


def read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    payload = json.loads(line)
                    if isinstance(payload, dict):
                        rows.append(payload)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("data", "rows", "examples", "queries", "results", "predictions"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def first_present(row: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def listify(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def stringify_ids(value: Any) -> List[str]:
    out: List[str] = []
    for item in listify(value):
        if item is None:
            continue
        if isinstance(item, dict):
            raw = first_present(item, ("chunk_id", "id", "doc_id", "document_id", "passage_id"))
            if raw is None:
                continue
            item = raw
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def normalize_groups(value: Any) -> List[List[str]]:
    groups: List[List[str]] = []
    for group in listify(value):
        ids = stringify_ids(group)
        if ids:
            groups.append(ids)
    return groups


def dedupe_preserve_order(ids: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in ids:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def query_id(row: Dict[str, Any], fallback: int) -> str:
    raw = first_present(row, ("query_id", "qid", "question_id", "id"))
    return str(raw if raw is not None else fallback)


def discover_file(run_dir: Path, explicit: Optional[str], candidates: Sequence[str]) -> Optional[Path]:
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_absolute() else (Path.cwd() / path).resolve()
    for rel in candidates:
        path = run_dir / rel
        if path.exists():
            return path
    return None


def load_rows_by_query(path: Optional[Path]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    if path is None or not path.exists():
        return {}, []
    rows = read_json_or_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for index, row in enumerate(rows):
        qid = query_id(row, index)
        if qid not in by_id:
            order.append(qid)
        by_id[qid] = row
    return by_id, order


def extract_score(item: Dict[str, Any]) -> Optional[float]:
    raw = first_present(item, ("score", "retrieval_score", "similarity", "prob", "final_score"))
    try:
        return None if raw is None else float(raw)
    except Exception:
        return None


def extract_ids_from_items(items: Any) -> Tuple[List[str], List[Optional[float]]]:
    ids: List[str] = []
    scores: List[Optional[float]] = []
    if not isinstance(items, list):
        return ids, scores
    for item in items:
        score: Optional[float] = None
        raw_id: Any = None
        if isinstance(item, dict):
            raw_id = first_present(item, ("chunk_id", "id", "doc_id", "document_id", "passage_id"))
            score = extract_score(item)
            if raw_id is None and isinstance(item.get("passage"), dict):
                passage = item["passage"]
                raw_id = first_present(passage, ("chunk_id", "id", "doc_id", "document_id", "passage_id"))
                score = score if score is not None else extract_score(passage)
        else:
            raw_id = item
        if raw_id is None:
            continue
        text = str(raw_id).strip()
        if text:
            ids.append(text)
            scores.append(score)
    return ids, scores


def extract_retrieval(row: Optional[Dict[str, Any]]) -> Tuple[List[str], List[Optional[float]], Optional[str]]:
    if not row:
        return [], [], None
    for key in (
        "retrieved_ids",
        "retrieved_ids_top10",
        "retrieved_chunk_ids",
        "retrieved_doc_ids",
        "ranked_chunk_ids",
        "ranked_doc_ids",
    ):
        ids = stringify_ids(row.get(key))
        if ids:
            scores = listify(row.get("retrieved_scores") or row.get("scores"))
            score_values: List[Optional[float]] = []
            for value in scores[: len(ids)]:
                try:
                    score_values.append(float(value))
                except Exception:
                    score_values.append(None)
            score_values.extend([None] * (len(ids) - len(score_values)))
            deduped: List[str] = []
            deduped_scores: List[Optional[float]] = []
            seen = set()
            for item_id, score in zip(ids, score_values):
                if item_id not in seen:
                    seen.add(item_id)
                    deduped.append(item_id)
                    deduped_scores.append(score)
            return deduped, deduped_scores, key
    for key in (
        "retrieval_results",
        "retrieved_contexts",
        "top_contexts",
        "contexts",
        "ctxs",
        "docs",
        "passages",
        "passages_used",
        "candidate_traces",
    ):
        ids, scores = extract_ids_from_items(row.get(key))
        if ids:
            deduped: List[str] = []
            deduped_scores: List[Optional[float]] = []
            seen = set()
            for item_id, score in zip(ids, scores):
                if item_id not in seen:
                    seen.add(item_id)
                    deduped.append(item_id)
                    deduped_scores.append(score)
            return deduped, deduped_scores, key
    return [], [], None


def normalize_answer(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_tokens(text: Any) -> List[str]:
    normalized = normalize_answer(text)
    return normalized.split() if normalized else []


def exact_match(prediction: Any, references: Sequence[Any]) -> Optional[float]:
    refs = [normalize_answer(ref) for ref in references if normalize_answer(ref)]
    if not refs:
        return None
    pred = normalize_answer(prediction)
    return 1.0 if pred in refs else 0.0


def token_f1(prediction: Any, references: Sequence[Any]) -> Optional[float]:
    ref_tokens = [answer_tokens(ref) for ref in references if answer_tokens(ref)]
    if not ref_tokens:
        return None
    pred_tokens = answer_tokens(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in ref_tokens:
        common = defaultdict(int)
        for token in ref:
            common[token] += 1
        overlap = 0
        for token in pred_tokens:
            if common[token] > 0:
                overlap += 1
                common[token] -= 1
        if overlap == 0:
            score = 0.0
        else:
            precision = overlap / len(pred_tokens)
            recall = overlap / len(ref)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: Any, references: Sequence[Any]) -> Optional[float]:
    ref_tokens = [answer_tokens(ref) for ref in references if answer_tokens(ref)]
    if not ref_tokens:
        return None
    pred_tokens = answer_tokens(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in ref_tokens:
        lcs = lcs_len(pred_tokens, ref)
        if lcs == 0:
            score = 0.0
        else:
            precision = lcs / len(pred_tokens)
            recall = lcs / len(ref)
            score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def compute_bert_scores(
    items: Sequence[Tuple[str, str, Sequence[Any]]],
    *,
    model_type: str,
    lang: str,
    batch_size: int,
    device: Optional[str],
    rescale_with_baseline: bool,
) -> Tuple[Dict[str, Dict[str, Optional[float]]], Optional[str]]:
    empty = {qid: {key: None for key in BERTSCORE_KEYS} for qid, _, _ in items}
    pair_predictions: List[str] = []
    pair_references: List[str] = []
    pair_query_ids: List[str] = []
    for qid, prediction, references in items:
        refs = [str(ref).strip() for ref in references if str(ref).strip()]
        pred = str(prediction).strip()
        if not pred or not refs:
            continue
        for ref in refs:
            pair_query_ids.append(qid)
            pair_predictions.append(pred)
            pair_references.append(ref)

    if not pair_predictions:
        return empty, NO_BERTSCORE_PAIRS_ERROR

    try:
        from bert_score import score as bert_score_score  # type: ignore
    except Exception as exc:
        return empty, f"BERTScore could not be imported: {exc}"

    kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "rescale_with_baseline": rescale_with_baseline,
        "verbose": False,
    }
    if model_type:
        kwargs["model_type"] = model_type
    if lang:
        kwargs["lang"] = lang
    if device:
        kwargs["device"] = device

    try:
        precision, recall, f1 = bert_score_score(pair_predictions, pair_references, **kwargs)
    except Exception as exc:
        return empty, f"BERTScore computation failed: {exc}"

    scores = dict(empty)
    for qid, p_value, r_value, f_value in zip(pair_query_ids, precision.tolist(), recall.tolist(), f1.tolist()):
        candidate = {
            "bertscore_precision": float(p_value),
            "bertscore_recall": float(r_value),
            "bertscore_f1": float(f_value),
        }
        current = scores.get(qid, {key: None for key in BERTSCORE_KEYS})
        current_f1 = current.get("bertscore_f1")
        if current_f1 is None or candidate["bertscore_f1"] > current_f1:
            scores[qid] = candidate
    return scores, None


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(ranked_ids[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if idcg == 0 else dcg / idcg


def flat_retrieval_metrics(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> Dict[str, float]:
    relevant = set(relevant_ids)
    top = ranked_ids[:k]
    hits = [rank for rank, item in enumerate(top, start=1) if item in relevant]
    return {
        "recall": (len(set(top) & relevant) / len(relevant)) if relevant else 0.0,
        "mrr": (1.0 / hits[0]) if hits else 0.0,
        "ndcg": ndcg_at_k(top, list(relevant), k),
        "hit_rate": 1.0 if hits else 0.0,
    }


def strict_group_metrics(ranked_ids: Sequence[str], groups: Sequence[Sequence[str]], k: int) -> Dict[str, Optional[float]]:
    top = set(ranked_ids[:k])
    valid_groups = [set(group) for group in groups if group]
    fully_hit = [group for group in valid_groups if group.issubset(top)]
    return {
        "recall": (len(fully_hit) / len(valid_groups)) if valid_groups else 0.0,
        "mrr": None,
        "ndcg": None,
        "hit_rate": 1.0 if fully_hit else 0.0,
    }


def percentile(values: Sequence[float], pct: float) -> Optional[float]:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * pct
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return clean[int(position)]
    return clean[low] * (high - position) + clean[high] * (position - low)


def summary_stats(values: Sequence[Any]) -> Dict[str, Optional[float]]:
    clean: List[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p95": percentile(clean, 0.95),
    }


def safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except Exception:
        return None
    return number if math.isfinite(number) else None


def field_float(row: Dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    return safe_float(first_present(row, keys))


def field_float_from_rows(rows: Sequence[Dict[str, Any]], keys: Sequence[str]) -> Optional[float]:
    for row in rows:
        value = field_float(row, keys)
        if value is not None:
            return value
    return None


def load_gpu_memory_by_query(resource_path: Optional[Path]) -> Dict[str, float]:
    if resource_path is None or not resource_path.exists():
        return {}
    rows = read_json_or_jsonl(resource_path)
    values: Dict[str, float] = {}
    for index, row in enumerate(rows):
        qid_raw = first_present(row, ("query_id", "qid", "question_id", "id"))
        if qid_raw is None:
            continue
        qid = str(qid_raw)
        memory = field_float(row, ("peak_gpu_memory_mb", "max_gpu_memory_mb", "gpu_peak_memory_mb", "gpu_memory_mb"))
        if memory is None:
            continue
        values[qid] = max(values.get(qid, memory), memory)
    return values


def package_versions() -> Dict[str, Optional[str]]:
    packages = ["python", "bert-score", "numpy", "rouge-score", "torch", "transformers", "vllm", "datasets"]
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for package in packages:
        if package == "python":
            continue
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def required_bert_score_error_message(error: str) -> str:
    return (
        f"{error}\n\n"
        "BERTScore is enabled, so the evaluation cannot be reported with null BERTScore fields. "
        "Install the dependency in the same Python environment used to run this script, for example:\n"
        "  python3 -m pip install 'bert-score>=0.3.13'\n\n"
        "For smoke tests, pass --disable-bert-score. To preserve the old soft-missing behavior, "
        "pass --allow-missing-bert-score."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate compact retrieval and generation metrics for an existing RAG run.")
    parser.add_argument("--run-dir", required=True, help="Existing run directory containing raw retrieval/generation outputs.")
    parser.add_argument("--labels-file", help="Labels file with query ids, reference answers, and gold/silver relevance ids.")
    parser.add_argument("--answers-file", help="Optional separate reference-answer file keyed by query_id.")
    parser.add_argument("--output-dir", required=True, help="Directory for compact evaluation artifacts.")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10], help="Retrieval cutoffs to report, for example: --ks 5 10")
    parser.add_argument("--generation-top-k", type=int, default=10, help="Top-k context used by the already-generated answers.")
    parser.add_argument("--predictions-file", help="Optional generated-answer file. Defaults to rag/qa_predictions.jsonl when present.")
    parser.add_argument("--retrieval-file", help="Optional raw ranked retrieval output file.")
    parser.add_argument("--traces-file", help="Optional RAG trace file used as a fallback for retrieved ids.")
    parser.add_argument("--disable-bert-score", action="store_true", help="Skip BERTScore computation.")
    parser.add_argument("--bert-score-model", default="roberta-large", help="Model used by bert-score. Default: roberta-large.")
    parser.add_argument("--bert-score-lang", default="en", help="Language passed to bert-score. Default: en.")
    parser.add_argument("--bert-score-batch-size", type=int, default=16)
    parser.add_argument("--bert-score-device", help="Optional bert-score device, for example cuda:0 or cpu.")
    parser.add_argument(
        "--bert-score-rescale-with-baseline",
        action="store_true",
        help="Use bert-score baseline rescaling when available.",
    )
    parser.add_argument(
        "--allow-missing-bert-score",
        action="store_true",
        help=(
            "Continue and write null BERTScore fields if BERTScore import or computation fails. "
            "By default, enabled BERTScore must produce numeric values."
        ),
    )
    parser.add_argument(
        "--disable-doc-id-label-fallback",
        action="store_true",
        help="Do not use doc_id as a gold relevance id when explicit gold/silver fields are absent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ks = sorted(set(int(k) for k in args.ks if int(k) > 0))
    if 5 not in ks or 10 not in ks:
        raise ValueError("This evaluator must report retrieval at both @5 and @10; pass --ks 5 10.")
    run_name = run_dir.name

    labels_path = discover_file(run_dir, args.labels_file, ("selection/qa_entries.json", "labels.jsonl", "labels.json"))
    answers_path = discover_file(run_dir, args.answers_file, ())
    predictions_path = discover_file(
        run_dir,
        args.predictions_file,
        ("rag/qa_predictions.jsonl", "qa_predictions.jsonl", "predictions.jsonl", "rag/predictions.jsonl"),
    )
    retrieval_path = discover_file(
        run_dir,
        args.retrieval_file,
        (
            "retrieval/retrieval_outputs.jsonl",
            "retrieval/retrieval_results.jsonl",
            "rag/retrieval_outputs.jsonl",
            "rag/retrieval_results.jsonl",
            "retrieval_outputs.jsonl",
            "retrieval_results.jsonl",
        ),
    )
    traces_path = discover_file(run_dir, args.traces_file, ("rag/rag_run_traces.jsonl", "rag_run_traces.jsonl"))
    resource_path = discover_file(run_dir, None, ("profiling/resource_usage.jsonl", "resource_usage.jsonl"))

    labels_by_id, labels_order = load_rows_by_query(labels_path)
    predictions_by_id, predictions_order = load_rows_by_query(predictions_path)
    retrieval_by_id, retrieval_order = load_rows_by_query(retrieval_path)
    traces_by_id, traces_order = load_rows_by_query(traces_path)
    answers_by_id, _ = load_rows_by_query(answers_path)
    gpu_by_query = load_gpu_memory_by_query(resource_path)

    assumptions: List[str] = [
        "Evaluated existing predictions only; no retrieval or generation was rerun.",
        f"Generation metrics correspond to generation_top_k = {args.generation_top_k}.",
        "Retrieved ids are deduplicated while preserving first occurrence before metric computation.",
    ]
    missing_reasons: Dict[str, str] = {}
    retrieval_source_notes: Dict[str, int] = defaultdict(int)

    if retrieval_path is None:
        assumptions.append("No explicit retrieval output file was found; retrieved ids were inferred from traces/prediction rows when possible.")
    if traces_path is not None and retrieval_path is None:
        assumptions.append("If candidate_traces are used, their order may reflect SELF-RAG candidate scoring rather than raw retriever order.")
    if labels_path is None:
        missing_reasons["retrieval_metrics"] = "No labels file was found or provided."
    if predictions_path is None:
        missing_reasons["rag_metrics"] = "No generated prediction file was found or provided."

    explicit_relevance_seen = False
    doc_id_fallback_used = False

    query_ids: List[str] = []
    seen_query_ids = set()
    for source_order in (predictions_order, retrieval_order, traces_order, labels_order):
        for qid in source_order:
            if qid not in seen_query_ids:
                seen_query_ids.add(qid)
                query_ids.append(qid)

    per_query_rows: List[Dict[str, Any]] = []
    bert_score_items: List[Tuple[str, str, Sequence[Any]]] = []
    metrics_by_view: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    primary_scores: Dict[str, List[float]] = defaultdict(list)
    generation_scores: Dict[str, List[float]] = defaultdict(list)
    efficiency_values: Dict[str, List[float]] = defaultdict(list)
    eligible_counts: Dict[str, int] = defaultdict(int)

    global_has_gold = any(stringify_ids(row.get("gold_chunk_ids")) for row in labels_by_id.values())
    global_has_silver_loose = any(stringify_ids(row.get("silver_chunk_ids")) for row in labels_by_id.values())
    global_has_silver_strict = any(normalize_groups(row.get("silver_chunk_groups")) for row in labels_by_id.values())
    if global_has_gold:
        primary_relevance = "gold"
    elif global_has_silver_loose:
        primary_relevance = "silver_loose"
    elif global_has_silver_strict:
        primary_relevance = "silver_strict"
    else:
        primary_relevance = "gold"

    for qid in query_ids:
        label = labels_by_id.get(qid, {})
        prediction = predictions_by_id.get(qid, {})
        retrieval_row = retrieval_by_id.get(qid) or traces_by_id.get(qid) or prediction
        retrieved_ids, retrieved_scores, retrieval_source = extract_retrieval(retrieval_row)
        retrieval_source_notes[str(retrieval_source or "missing")] += 1

        gold_ids = stringify_ids(label.get("gold_chunk_ids"))
        silver_loose_ids = stringify_ids(label.get("silver_chunk_ids"))
        silver_strict_groups = normalize_groups(label.get("silver_chunk_groups"))
        if gold_ids or silver_loose_ids or silver_strict_groups:
            explicit_relevance_seen = True
        if not (gold_ids or silver_loose_ids or silver_strict_groups) and not args.disable_doc_id_label_fallback:
            doc_id = first_present(label, ("doc_id", "document_id"))
            if doc_id is not None:
                gold_ids = [str(doc_id)]
                doc_id_fallback_used = True

        reference_answers = listify(
            first_present(answers_by_id.get(qid, {}), ("reference_answers", "answers", "answer", "gold_answers"))
        ) or listify(first_present(label, ("reference_answers", "answers", "answer", "gold_answers")))
        prediction_text = first_present(prediction, ("prediction", "final_prediction", "answer", "generated_answer", "output"))
        if prediction_text is not None and reference_answers:
            bert_score_items.append((qid, str(prediction_text), reference_answers))

        views: Dict[str, Any] = {
            "gold": dedupe_preserve_order(gold_ids),
            "silver_loose": dedupe_preserve_order(silver_loose_ids),
            "silver_strict": silver_strict_groups,
        }
        if views["gold"]:
            eligible_counts["gold"] += 1
        if views["silver_loose"]:
            eligible_counts["silver_loose"] += 1
        if views["silver_strict"]:
            eligible_counts["silver_strict"] += 1
        if views["gold"] or views["silver_strict"]:
            eligible_counts["union"] += 1
        gold_hit: Dict[int, bool] = {}
        strict_hit: Dict[int, bool] = {}
        per_query_primary: Dict[str, Optional[float]] = {}

        for k in ks:
            if views["gold"]:
                scores = flat_retrieval_metrics(retrieved_ids, views["gold"], k)
                gold_hit[k] = bool(scores["hit_rate"])
                for key, value in scores.items():
                    metrics_by_view["gold"][f"{key}@{k}"].append(value)
            else:
                gold_hit[k] = False

            if views["silver_loose"]:
                scores = flat_retrieval_metrics(retrieved_ids, views["silver_loose"], k)
                for key, value in scores.items():
                    metrics_by_view["silver_loose"][f"{key}@{k}"].append(value)

            if views["silver_strict"]:
                scores = strict_group_metrics(retrieved_ids, views["silver_strict"], k)
                strict_hit[k] = bool(scores["hit_rate"])
                for key, value in scores.items():
                    if value is not None:
                        metrics_by_view["silver_strict"][f"{key}@{k}"].append(value)
            else:
                strict_hit[k] = False

            if views["gold"] or views["silver_strict"]:
                union_hit = 1.0 if (gold_hit[k] or strict_hit[k]) else 0.0
                metrics_by_view["union"][f"hit_rate@{k}"].append(union_hit)

            primary_target = views.get(primary_relevance)
            primary_metric_values: Dict[str, Optional[float]]
            if primary_relevance in ("gold", "silver_loose") and primary_target:
                primary_metric_values = flat_retrieval_metrics(retrieved_ids, primary_target, k)  # type: ignore[arg-type]
            elif primary_relevance == "silver_strict" and primary_target:
                primary_metric_values = strict_group_metrics(retrieved_ids, primary_target, k)  # type: ignore[arg-type]
            else:
                primary_metric_values = {key: None for key in METRIC_KEYS}
            for key in METRIC_KEYS:
                per_query_primary[f"{key}@{k}"] = primary_metric_values[key]
                if primary_metric_values[key] is not None:
                    primary_scores[f"{key}@{k}"].append(float(primary_metric_values[key]))

        em = exact_match(prediction_text, reference_answers) if prediction_text is not None else None
        f1 = token_f1(prediction_text, reference_answers) if prediction_text is not None else None
        rouge = rouge_l(prediction_text, reference_answers) if prediction_text is not None else None
        for key, value in (("exact_match", em), ("token_f1", f1), ("rouge_l", rouge)):
            if value is not None:
                generation_scores[key].append(value)

        efficiency_rows = [prediction, retrieval_row or {}, traces_by_id.get(qid, {})]
        efficiency = {
            "retrieval_latency_ms": field_float_from_rows(efficiency_rows, ("retrieval_latency_ms",)),
            "generation_latency_ms": field_float_from_rows(efficiency_rows, ("generation_latency_ms",)),
            "total_latency_ms": field_float_from_rows(efficiency_rows, ("total_latency_ms",)),
            "context_tokens": field_float_from_rows(efficiency_rows, ("context_tokens", "context_token_count")),
            "answer_tokens": field_float_from_rows(efficiency_rows, ("answer_tokens", "answer_token_count")),
            "peak_gpu_memory_mb": field_float_from_rows(efficiency_rows, ("peak_gpu_memory_mb", "max_gpu_memory_mb", "gpu_peak_memory_mb")),
        }
        if efficiency["peak_gpu_memory_mb"] is None:
            efficiency["peak_gpu_memory_mb"] = gpu_by_query.get(qid)
        for key, value in efficiency.items():
            if value is not None:
                efficiency_values[key].append(value)

        relevant_for_output: Any
        if primary_relevance == "silver_strict":
            relevant_for_output = views["silver_strict"]
        else:
            relevant_for_output = views.get(primary_relevance, [])

        out_row: Dict[str, Any] = {
            "query_id": qid,
            "doc_id": first_present(label, ("doc_id", "document_id")) or first_present(prediction, ("doc_id", "document_id")),
            "question": first_present(label, ("question", "query", "instruction")) or first_present(prediction, ("question", "query", "instruction")),
            "exact_match": em,
            "token_f1": f1,
            "rouge_l": rouge,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "retrieval_latency_ms": efficiency["retrieval_latency_ms"],
            "generation_latency_ms": efficiency["generation_latency_ms"],
            "total_latency_ms": efficiency["total_latency_ms"],
            "context_tokens": efficiency["context_tokens"],
            "answer_tokens": efficiency["answer_tokens"],
            "retrieved_ids_top10": retrieved_ids[:10],
            "relevant_ids": relevant_for_output,
        }
        for k in ks:
            for key in METRIC_KEYS:
                out_row[f"{key}@{k}"] = per_query_primary.get(f"{key}@{k}")
        if any(score is not None for score in retrieved_scores[:10]):
            out_row["retrieved_scores_top10"] = retrieved_scores[:10]
        per_query_rows.append(out_row)

    if args.disable_bert_score:
        assumptions.append("BERTScore computation was disabled by --disable-bert-score.")
        missing_reasons["bert_score"] = "BERTScore computation was disabled."
    else:
        bert_scores, bert_error = compute_bert_scores(
            bert_score_items,
            model_type=args.bert_score_model,
            lang=args.bert_score_lang,
            batch_size=args.bert_score_batch_size,
            device=args.bert_score_device,
            rescale_with_baseline=args.bert_score_rescale_with_baseline,
        )
        if bert_error is not None:
            missing_reasons["bert_score"] = bert_error
            if bert_error != NO_BERTSCORE_PAIRS_ERROR and not args.allow_missing_bert_score:
                raise SystemExit(required_bert_score_error_message(bert_error))
        else:
            assumptions.append(
                "BERTScore was computed with best-over-references selection per query "
                f"using model={args.bert_score_model}, lang={args.bert_score_lang}, "
                f"rescale_with_baseline={args.bert_score_rescale_with_baseline}."
            )
        for row in per_query_rows:
            qid = str(row.get("query_id", ""))
            values = bert_scores.get(qid, {key: None for key in BERTSCORE_KEYS})
            for key in BERTSCORE_KEYS:
                row[key] = values.get(key)
                if row[key] is not None:
                    generation_scores[key].append(float(row[key]))

    if doc_id_fallback_used:
        assumptions.append("Explicit gold/silver relevance ids were absent for at least one query, so doc_id was used as a gold relevance id.")
    if not explicit_relevance_seen and not doc_id_fallback_used:
        missing_reasons["retrieval_metrics"] = "No gold_chunk_ids, silver_chunk_ids, silver_chunk_groups, or doc_id fallback labels were available."
    if not any(generation_scores.values()):
        missing_reasons["rag_metrics"] = "Reference answers or generated predictions were unavailable."
    for field in EFFICIENCY_FIELDS:
        if not efficiency_values[field]:
            missing_reasons[field] = f"No numeric {field} values were found."

    retrieval_metrics = {f"{key}@{k}": (statistics.fmean(primary_scores[f"{key}@{k}"]) if primary_scores[f"{key}@{k}"] else None) for k in ks for key in METRIC_KEYS}
    retrieval_metrics_by_relevance: Dict[str, Dict[str, Optional[float]]] = {}
    for view, values in metrics_by_view.items():
        retrieval_metrics_by_relevance[view] = {}
        for k in ks:
            for key in METRIC_KEYS:
                metric_name = f"{key}@{k}"
                retrieval_metrics_by_relevance[view][metric_name] = statistics.fmean(values[metric_name]) if values[metric_name] else None

    rag_metrics = {
        "exact_match": statistics.fmean(generation_scores["exact_match"]) if generation_scores["exact_match"] else None,
        "token_f1": statistics.fmean(generation_scores["token_f1"]) if generation_scores["token_f1"] else None,
        "rouge_l": statistics.fmean(generation_scores["rouge_l"]) if generation_scores["rouge_l"] else None,
        "bertscore_precision": statistics.fmean(generation_scores["bertscore_precision"]) if generation_scores["bertscore_precision"] else None,
        "bertscore_recall": statistics.fmean(generation_scores["bertscore_recall"]) if generation_scores["bertscore_recall"] else None,
        "bertscore_f1": statistics.fmean(generation_scores["bertscore_f1"]) if generation_scores["bertscore_f1"] else None,
    }
    efficiency_summary = {field: summary_stats(efficiency_values[field]) for field in EFFICIENCY_FIELDS}

    summary = {
        "method_name": args.method_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "run_name": run_name,
        "n_queries": len(per_query_rows),
        "k_values": ks,
        "primary_relevance": primary_relevance,
        "generation_top_k": args.generation_top_k,
        "retrieval_metrics": retrieval_metrics,
        "retrieval_metrics_by_relevance": retrieval_metrics_by_relevance,
        "rag_metrics": rag_metrics,
        "efficiency": efficiency_summary,
    }

    leaderboard = {
        "method_name": args.method_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "run_name": run_name,
        "generation_top_k": args.generation_top_k,
        "exact_match": rag_metrics["exact_match"],
        "token_f1": rag_metrics["token_f1"],
        "rouge_l": rag_metrics["rouge_l"],
        "bertscore_precision": rag_metrics["bertscore_precision"],
        "bertscore_recall": rag_metrics["bertscore_recall"],
        "bertscore_f1": rag_metrics["bertscore_f1"],
        "retrieval_latency_mean_ms": efficiency_summary["retrieval_latency_ms"]["mean"],
        "generation_latency_mean_ms": efficiency_summary["generation_latency_ms"]["mean"],
        "total_latency_mean_ms": efficiency_summary["total_latency_ms"]["mean"],
        "context_tokens_mean": efficiency_summary["context_tokens"]["mean"],
        "peak_gpu_memory_mean_mb": efficiency_summary["peak_gpu_memory_mb"]["mean"],
    }
    for k in ks:
        for key in METRIC_KEYS:
            leaderboard[f"{key}@{k}"] = retrieval_metrics.get(f"{key}@{k}")

    output_files = {
        "metrics_summary": output_dir / "metrics_summary.json",
        "metrics_per_query": output_dir / "metrics_per_query.jsonl",
        "leaderboard_row": output_dir / "leaderboard_row.json",
        "evaluation_manifest": output_dir / "evaluation_manifest.json",
    }
    input_files = {
        "run_dir": run_dir,
        "labels_file": labels_path,
        "answers_file": answers_path,
        "predictions_file": predictions_path,
        "retrieval_file": retrieval_path,
        "traces_file": traces_path,
        "resource_usage_file": resource_path,
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files_used": {key: str(value) for key, value in input_files.items() if value is not None},
        "output_files_written": {key: str(value) for key, value in output_files.items()},
        "relevance_source_used": {
            "primary_relevance": primary_relevance,
            "explicit_relevance_fields_seen": explicit_relevance_seen,
            "doc_id_fallback_used": doc_id_fallback_used,
            "eligible_query_counts": dict(eligible_counts),
        },
        "primary_relevance_choice": primary_relevance,
        "generation_evaluated_from_top_k": args.generation_top_k,
        "retrieval_source_fields": dict(retrieval_source_notes),
        "assumptions": assumptions,
        "missing_metrics_and_reasons": missing_reasons,
        "command_used": " ".join(sys.argv),
        "package_versions": package_versions(),
        "bert_score_config": {
            "enabled": not args.disable_bert_score,
            "model": args.bert_score_model,
            "lang": args.bert_score_lang,
            "batch_size": args.bert_score_batch_size,
            "device": args.bert_score_device,
            "rescale_with_baseline": args.bert_score_rescale_with_baseline,
            "best_over_references": True,
        },
    }

    write_json(output_files["metrics_summary"], summary)
    write_jsonl(output_files["metrics_per_query"], per_query_rows)
    write_json(output_files["leaderboard_row"], leaderboard)
    write_json(output_files["evaluation_manifest"], manifest)

    print(str(output_dir))


if __name__ == "__main__":
    main()
