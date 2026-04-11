from __future__ import annotations

from typing import Any, Dict, List, Tuple


def coerce_to_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = []
        for item in value:
            text = coerce_to_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts) if parts else None
    if isinstance(value, dict):
        for key in ("text", "answer", "answers"):
            text = coerce_to_text(value.get(key))
            if text:
                return text
    return None


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required for the in-repo NarrativeQA loader.") from exc
    return load_dataset, get_dataset_config_names


def _resolve_config(requested: str | None, dataset_name: str = "deepmind/narrativeqa") -> str | None:
    _, get_dataset_config_names = _datasets()
    try:
        configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_narrativeqa_dataset(*, config_name: str | None):
    load_dataset, _ = _datasets()
    cfg = _resolve_config(config_name)
    try:
        return load_dataset("deepmind/narrativeqa", name=cfg) if cfg else load_dataset("deepmind/narrativeqa")
    except ValueError as exc:
        if "Feature type 'List' not found" in str(exc):
            raise RuntimeError(
                "Loading deepmind/narrativeqa failed because the installed datasets version is too old. "
                "Upgrade to datasets>=4 and retry."
            ) from exc
        raise


def load_narrativeqa_records(
    *,
    split: str = "test",
    config_name: str | None = "default",
    qa_n: int | str = "all",
    qa_selection_method: str = "first",
    max_documents: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    import random

    ds = _load_narrativeqa_dataset(config_name=config_name)
    documents: List[Dict[str, Any]] = []
    qa_entries: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_doc_ids = set()

    for row in ds[split]:
        doc = row.get("document") or {}
        doc_id = doc.get("id") or row.get("document_id") or row.get("doc_id")
        if not doc_id:
            continue
        doc_id = str(doc_id)
        text = coerce_to_text(doc.get("text") if isinstance(doc, dict) else doc) or ""
        title = ""
        if isinstance(doc, dict):
            title = str(doc.get("title", "") or "")

        if doc_id not in seen_doc_ids and text:
            seen_doc_ids.add(doc_id)
            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                }
            )

        question = row.get("question")
        if isinstance(question, dict):
            question = question.get("text")
        if not isinstance(question, str) or not question.strip():
            continue

        answers = row.get("answers")
        answer_text = coerce_to_text(answers)
        if not answer_text:
            continue

        qa_entries.append(
            {
                "query_id": str(len(qa_entries)),
                "doc_id": doc_id,
                "question": question.strip(),
                "reference_answers": [answer_text.strip()],
                "retrieval_spans": [],
                "provided_contexts": [
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "text": text,
                    }
                ],
                "raw_entry": row,
            }
        )

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
