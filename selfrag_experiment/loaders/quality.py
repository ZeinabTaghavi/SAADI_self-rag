from __future__ import annotations

from typing import Any, Dict, List, Tuple


DATASET_ID = "tasksource/QuALITY"
VALID_SPLITS = {"train", "validation"}
SPLIT_ALIASES = {
    "default": "validation",
    "dev": "validation",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "validation",
    "train": "train",
}


def coerce_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    return None


def _datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required for the in-repo QuALITY loader.") from exc
    return load_dataset


def _normalize_split(split: str | None) -> str:
    raw = str(split or "validation").strip().lower()
    normalized = SPLIT_ALIASES.get(raw, raw)
    if normalized not in VALID_SPLITS:
        raise ValueError(f"Unsupported QuALITY split: {split!r}")
    return normalized


def _load_quality_split(split_name: str):
    load_dataset = _datasets()
    try:
        return load_dataset(DATASET_ID, split=split_name)
    except TypeError:
        return load_dataset(DATASET_ID)[split_name]


def _row_doc_id(row: Dict[str, Any], fallback_index: int) -> str:
    for key in ("article_id", "document_id", "doc_id"):
        value = row.get(key)
        if isinstance(value, (int, str)) and str(value).strip():
            return f"article:{str(value).strip()}"
    value = row.get("title")
    if isinstance(value, str) and value.strip():
        return f"article:{'_'.join(value.strip().split())}"
    return f"article:{fallback_index}"


def _options_list(row: Dict[str, Any]) -> List[str]:
    raw = row.get("options")
    if isinstance(raw, list):
        return [text for text in (coerce_to_text(item) for item in raw) if text]
    return []


def _gold_option_index(row: Dict[str, Any], options: List[str]) -> int | None:
    for key in ("gold_label", "writer_label"):
        value = row.get(key)
        try:
            idx = int(value)
        except Exception:
            continue
        if 1 <= idx <= len(options):
            return idx - 1
        if 0 <= idx < len(options):
            return idx
    return None


def _metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("title", "source", "author", "topic", "year"):
        value = coerce_to_text(row.get(key))
        if value:
            out[key] = value
    difficult = row.get("difficult")
    try:
        if difficult is not None:
            out["difficult"] = int(difficult)
    except Exception:
        pass
    return out


def load_quality_records(
    *,
    split: str = "validation",
    config_name: str | None = "default",
    qa_n: int | str = "all",
    qa_selection_method: str = "first",
    max_documents: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    import random

    _ = config_name
    split_name = _normalize_split(split)
    rows = _load_quality_split(split_name)

    documents: List[Dict[str, Any]] = []
    qa_entries: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_doc_ids = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue

        doc_id = _row_doc_id(row, fallback_index=index)
        title = str(row.get("title", "") or "")
        article_text = coerce_to_text(row.get("article")) or ""
        if doc_id not in seen_doc_ids and article_text:
            seen_doc_ids.add(doc_id)
            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": article_text,
                }
            )

        question = coerce_to_text(row.get("question"))
        if not question:
            continue
        options = _options_list(row)
        gold_idx = _gold_option_index(row, options)
        if gold_idx is None or gold_idx >= len(options):
            continue
        answer_text = options[gold_idx]
        if not answer_text:
            continue

        entry: Dict[str, Any] = {
            "query_id": str(row.get("question_unique_id") or f"{doc_id}:q{index}"),
            "doc_id": doc_id,
            "question": question,
            "reference_answers": [answer_text],
            "retrieval_spans": [],
            "provided_contexts": [
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": article_text,
                }
            ],
            "choices": options,
            "gold_option_index": gold_idx,
            "gold_option_label": gold_idx + 1,
            "raw_entry": row,
        }
        meta = _metadata(row)
        if meta:
            entry["metadata"] = meta
        qa_entries.append(entry)

    if max_documents is not None:
        documents = documents[: int(max_documents)]
        allowed_doc_ids = {doc["doc_id"] for doc in documents}
        qa_entries = [item for item in qa_entries if item["doc_id"] in allowed_doc_ids]
        notes.append(f"Applied max_documents={int(max_documents)} from the YAML-style sample settings.")

    total = len(qa_entries)
    if isinstance(qa_n, str):
        qa_n = total if qa_n.lower() == "all" else int(qa_n)
    qa_n = min(int(qa_n), total)
    indices = list(range(total))
    if qa_n < total:
        if qa_selection_method == "random":
            indices = random.sample(indices, qa_n)
            notes.append("QA selection used random sampling, matching the external loader behavior.")
        else:
            indices = indices[:qa_n]
    qa_entries = [qa_entries[index] for index in indices]

    return qa_entries, documents, notes
