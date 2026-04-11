from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from selfrag_experiment.loaders.loogle import load_loogle_records
from selfrag_experiment.loaders.narrativeqa import load_narrativeqa_records
from selfrag_experiment.loaders.novelhopqa import load_novelhopqa_records
from selfrag_experiment.loaders.qasper import load_qasper_records
from selfrag_experiment.loaders.quality import load_quality_records


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if file_path.suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        raise ValueError(f"Unsupported JSON structure in {path}.")
    if file_path.suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    raise ValueError(f"Unsupported local dataset format for {path}. Expected .json or .jsonl.")


def _load_hf_records(loader_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except Exception as exc:  # pragma: no cover - optional import
        raise RuntimeError("datasets is not available, so Hugging Face datasets cannot be loaded.") from exc
    source = loader_cfg["source"]
    subset = loader_cfg.get("subset")
    split = loader_cfg.get("split") or "train"
    data_files = loader_cfg.get("data_files")
    dataset = hf_load_dataset(source, subset, split=split, data_files=data_files)
    return [dict(row) for row in dataset]


def _listify_answers(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_context(context: Dict[str, Any], fallback_index: int) -> Dict[str, Any]:
    doc_id = context.get("id", context.get("doc_id", context.get("document_id", f"context_{fallback_index}")))
    title = context.get("title", "")
    text = context.get("text", context.get("contents", context.get("paragraph", "")))
    return {
        "doc_id": str(doc_id),
        "title": "" if title is None else str(title),
        "text": "" if text is None else str(text),
    }


def _extract_contexts(record: Dict[str, Any], loader_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    field_candidates: List[Optional[str]] = [
        loader_cfg.get("contexts_field"),
        "ctxs",
        "top_contexts",
        "docs",
        "contexts",
    ]
    for field_name in field_candidates:
        if not field_name or field_name not in record or record[field_name] is None:
            continue
        raw_contexts = record[field_name]
        if isinstance(raw_contexts, list):
            return [_normalize_context(item, index) for index, item in enumerate(raw_contexts) if isinstance(item, dict)]
    return []


def _build_documents_from_records(
    records: Iterable[Dict[str, Any]],
    loader_cfg: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    document_id_field = loader_cfg["document_id_field"]
    text_field = loader_cfg["document_text_field"]
    title_field = loader_cfg["document_title_field"]

    ordered: List[Dict[str, Any]] = []
    by_id: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(records):
        raw_id = row.get(document_id_field, row.get("id", row.get("doc_id", f"doc_{index}")))
        doc_id = str(raw_id)
        text = row.get(text_field, row.get("text", row.get("contents")))
        if text is None:
            continue
        title = row.get(title_field, row.get("title", ""))
        entry = {
            "doc_id": doc_id,
            "title": "" if title is None else str(title),
            "text": str(text),
        }
        if doc_id not in by_id:
            by_id[doc_id] = entry
            ordered.append(entry)
    return ordered, by_id


def load_and_normalize_dataset(resolved_cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    loader_cfg = resolved_cfg["dataset_loader"]
    notes: List[str] = []

    if str(loader_cfg.get("source_type", "")).lower() == "loogle":
        qa_entries, documents, loogle_notes = load_loogle_records(
            split=str(loader_cfg.get("split") or "test"),
            config_name=loader_cfg.get("config_name"),
            qa_n=loader_cfg.get("qa_n", "all"),
            qa_selection_method=str(loader_cfg.get("qa_selection_method") or "first"),
            max_documents=loader_cfg.get("max_docs"),
        )
        notes.extend(loogle_notes)
        max_questions = loader_cfg.get("max_questions")
        if isinstance(max_questions, int) and max_questions > 0:
            qa_entries = qa_entries[:max_questions]
        if loader_cfg["question_ids"]:
            allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
            qa_entries = [item for item in qa_entries if item["query_id"] in allowed_ids]
            notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")
        return qa_entries, documents, notes

    if str(loader_cfg.get("source_type", "")).lower() == "narrativeqa":
        qa_entries, documents, nqa_notes = load_narrativeqa_records(
            split=str(loader_cfg.get("split") or "test"),
            config_name=loader_cfg.get("config_name"),
            qa_n=loader_cfg.get("qa_n", "all"),
            qa_selection_method=str(loader_cfg.get("qa_selection_method") or "first"),
            max_documents=loader_cfg.get("max_docs"),
        )
        notes.extend(nqa_notes)
        max_questions = loader_cfg.get("max_questions")
        if isinstance(max_questions, int) and max_questions > 0:
            qa_entries = qa_entries[:max_questions]
        if loader_cfg["question_ids"]:
            allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
            qa_entries = [item for item in qa_entries if item["query_id"] in allowed_ids]
            notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")
        return qa_entries, documents, notes

    if str(loader_cfg.get("source_type", "")).lower() == "qasper":
        qa_entries, documents, qasper_notes = load_qasper_records(
            split=str(loader_cfg.get("split") or "test"),
            config_name=loader_cfg.get("config_name"),
            qa_n=loader_cfg.get("qa_n", "all"),
            qa_selection_method=str(loader_cfg.get("qa_selection_method") or "first"),
            max_documents=loader_cfg.get("max_docs"),
        )
        notes.extend(qasper_notes)
        max_questions = loader_cfg.get("max_questions")
        if isinstance(max_questions, int) and max_questions > 0:
            qa_entries = qa_entries[:max_questions]
        if loader_cfg["question_ids"]:
            allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
            qa_entries = [item for item in qa_entries if item["query_id"] in allowed_ids]
            notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")
        return qa_entries, documents, notes

    if str(loader_cfg.get("source_type", "")).lower() == "quality":
        qa_entries, documents, quality_notes = load_quality_records(
            split=str(loader_cfg.get("split") or "validation"),
            config_name=loader_cfg.get("config_name"),
            qa_n=loader_cfg.get("qa_n", "all"),
            qa_selection_method=str(loader_cfg.get("qa_selection_method") or "first"),
            max_documents=loader_cfg.get("max_docs"),
        )
        notes.extend(quality_notes)
        max_questions = loader_cfg.get("max_questions")
        if isinstance(max_questions, int) and max_questions > 0:
            qa_entries = qa_entries[:max_questions]
        if loader_cfg["question_ids"]:
            allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
            qa_entries = [item for item in qa_entries if item["query_id"] in allowed_ids]
            notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")
        return qa_entries, documents, notes

    if str(loader_cfg.get("source_type", "")).lower() == "novelhopqa":
        qa_entries, documents, nhqa_notes = load_novelhopqa_records(
            split=str(loader_cfg.get("split") or "test"),
            config_name=loader_cfg.get("config_name"),
            qa_n=loader_cfg.get("qa_n", "all"),
            qa_selection_method=str(loader_cfg.get("qa_selection_method") or "first"),
            max_documents=loader_cfg.get("max_docs"),
            books_root=loader_cfg.get("books_root"),
        )
        notes.extend(nhqa_notes)
        max_questions = loader_cfg.get("max_questions")
        if isinstance(max_questions, int) and max_questions > 0:
            qa_entries = qa_entries[:max_questions]
        if loader_cfg["question_ids"]:
            allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
            qa_entries = [item for item in qa_entries if item["query_id"] in allowed_ids]
            notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")
        return qa_entries, documents, notes

    if loader_cfg["qa_path"]:
        qa_records = _load_json_or_jsonl(loader_cfg["qa_path"])
    elif loader_cfg["source_type"] == "hf":
        qa_records = _load_hf_records(loader_cfg)
    elif loader_cfg["source"]:
        qa_records = _load_json_or_jsonl(loader_cfg["source"])
    else:
        raise ValueError("No dataset source could be resolved from the YAML or CLI.")

    docs_records: List[Dict[str, Any]] = []
    if loader_cfg["docs_path"]:
        docs_records = _load_json_or_jsonl(loader_cfg["docs_path"])

    documents, docs_by_id = _build_documents_from_records(docs_records or qa_records, loader_cfg)

    normalized_qa: List[Dict[str, Any]] = []
    for index, row in enumerate(qa_records):
        question = row.get(loader_cfg["question_field"], row.get("question", row.get("instruction")))
        if question is None:
            continue
        query_id = row.get(loader_cfg["query_id_field"], row.get("id", f"q_{index}"))
        doc_id = row.get(loader_cfg["doc_id_field"], row.get("doc_id"))
        contexts = _extract_contexts(row, loader_cfg)
        if doc_id is None and contexts:
            doc_id = contexts[0]["doc_id"]
        if doc_id is None:
            doc_id = f"doc_for_{query_id}"
        normalized_qa.append(
            {
                "query_id": str(query_id),
                "doc_id": str(doc_id),
                "question": str(question),
                "reference_answers": _listify_answers(
                    row.get(loader_cfg["answers_field"], row.get("answers", row.get("answer", row.get("output"))))
                ),
                "provided_contexts": contexts,
                "raw_entry": row,
            }
        )

        if not contexts and str(doc_id) in docs_by_id:
            normalized_qa[-1]["provided_contexts"] = [docs_by_id[str(doc_id)]]

    if loader_cfg["question_ids"]:
        allowed_ids = {str(item) for item in loader_cfg["question_ids"]}
        normalized_qa = [item for item in normalized_qa if item["query_id"] in allowed_ids]
        notes.append(f"Filtered QA entries to the explicitly requested question_ids set ({len(allowed_ids)} ids).")

    max_questions = loader_cfg.get("max_questions")
    if isinstance(max_questions, int) and max_questions > 0:
        normalized_qa = normalized_qa[:max_questions]

    if not documents:
        inferred_docs: List[Dict[str, Any]] = []
        seen = set()
        for qa_entry in normalized_qa:
            for ctx in qa_entry["provided_contexts"]:
                if ctx["doc_id"] not in seen:
                    inferred_docs.append(ctx)
                    seen.add(ctx["doc_id"])
        documents = inferred_docs
        if documents:
            notes.append("No standalone documents source was provided; corpus/documents.jsonl was inferred from QA contexts.")

    max_docs = loader_cfg.get("max_docs")
    if isinstance(max_docs, int) and max_docs > 0:
        selected_doc_ids = {doc["doc_id"] for doc in documents[:max_docs]}
        documents = documents[:max_docs]
        normalized_qa = [item for item in normalized_qa if item["doc_id"] in selected_doc_ids]
        for qa_entry in normalized_qa:
            qa_entry["provided_contexts"] = [ctx for ctx in qa_entry["provided_contexts"] if ctx["doc_id"] in selected_doc_ids]
        notes.append(f"Applied max_docs={max_docs} to the selected corpus documents.")

    return normalized_qa, documents, notes
