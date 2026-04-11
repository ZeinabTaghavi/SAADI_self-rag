from __future__ import annotations

from importlib.metadata import version as pkg_version
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
    return None


def _datasets():
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("datasets package is required for the in-repo QASPER loader.") from exc
    return load_dataset, get_dataset_config_names


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        return int(raw.split(".", 1)[0]) if raw else None
    except Exception:
        return None


def _resolve_config(requested: str | None, dataset_name: str = "allenai/qasper") -> str | None:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs: Dict[str, Any] = {}
    if major is None or major < 4:
        kwargs["trust_remote_code"] = True
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            configs = get_dataset_config_names(dataset_name, **kwargs) or []
        except TypeError:
            configs = get_dataset_config_names(dataset_name) or []
    except Exception:
        configs = []
    if not configs:
        return None
    if requested in configs:
        return requested
    return configs[0]


def _load_qasper_dataset(*, config_name: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()
    cfg = _resolve_config(config_name)
    if major is not None and major >= 4:
        modern_kwargs = {"revision": "refs/convert/parquet"}
        try:
            return load_dataset("allenai/qasper", name=cfg, **modern_kwargs) if cfg else load_dataset("allenai/qasper", **modern_kwargs)
        except TypeError:
            return load_dataset("allenai/qasper", name=cfg) if cfg else load_dataset("allenai/qasper")
        except Exception as exc:
            if "Dataset scripts are no longer supported" in str(exc) and "qasper.py" in str(exc):
                raise RuntimeError(
                    "Loading allenai/qasper failed with datasets>=4 because parquet fallback could not be resolved. "
                    "Use local files or run with datasets<4."
                ) from exc
            raise

    legacy_kwargs = {"trust_remote_code": True}
    try:
        try:
            return load_dataset("allenai/qasper", name=cfg, **legacy_kwargs) if cfg else load_dataset("allenai/qasper", **legacy_kwargs)
        except TypeError:
            return load_dataset("allenai/qasper", name=cfg) if cfg else load_dataset("allenai/qasper")
    except RuntimeError as exc:
        if "trust_remote_code" in str(exc):
            raise RuntimeError(
                "Loading allenai/qasper requires script loading. Use datasets<4 with trust_remote_code enabled."
            ) from exc
        raise


def _extract_document_text(row: Dict[str, Any]) -> str:
    full_text = (row.get("full_text") or {}).get("paragraphs", "")
    if isinstance(full_text, list):
        if full_text and isinstance(full_text[0], list):
            return "\n".join([p for section in full_text for p in section if isinstance(p, str)])
        return "\n".join([p for p in full_text if isinstance(p, str)])
    return coerce_to_text(full_text) or ""


def load_qasper_records(
    *,
    split: str = "test",
    config_name: str | None = "default",
    qa_n: int | str = "all",
    qa_selection_method: str = "first",
    max_documents: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    import random

    ds = _load_qasper_dataset(config_name=config_name)
    documents: List[Dict[str, Any]] = []
    qa_entries: List[Dict[str, Any]] = []
    notes: List[str] = []
    seen_doc_ids = set()

    for row in ds[split]:
        doc_id = str(row.get("id"))
        title = str(row.get("title", "") or "")
        text = _extract_document_text(row)
        if doc_id not in seen_doc_ids and text:
            seen_doc_ids.add(doc_id)
            documents.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "text": text,
                }
            )

        qas = row.get("qas") or {}
        questions = qas.get("question") or []
        answers_all = qas.get("answers") or []

        for question, answer_dict in zip(questions, answers_all):
            if not isinstance(question, str) or not question.strip():
                continue
            answer_texts: List[str] = []
            retrieval_spans: List[str] = []
            for answer in (answer_dict or {}).get("answer", []) or []:
                if not isinstance(answer, dict) or answer.get("unanswerable"):
                    continue
                if answer.get("extractive_spans"):
                    text_answer = " ".join([s for s in answer.get("extractive_spans", []) if isinstance(s, str)])
                elif isinstance(answer.get("free_form_answer"), str) and answer.get("free_form_answer", "").strip():
                    text_answer = answer["free_form_answer"].strip()
                elif answer.get("yes_no") is not None:
                    text_answer = "Yes" if bool(answer["yes_no"]) else "No"
                else:
                    text_answer = ""
                evidence = answer.get("evidence") or answer.get("highlighted_evidence") or []
                if text_answer:
                    answer_texts.append(text_answer)
                if isinstance(evidence, str) and evidence.strip():
                    retrieval_spans.append(evidence.strip())
                elif isinstance(evidence, list):
                    retrieval_spans.extend([item.strip() for item in evidence if isinstance(item, str) and item.strip()])

            if answer_texts or retrieval_spans:
                qa_entries.append(
                    {
                        "query_id": str(len(qa_entries)),
                        "doc_id": doc_id,
                        "question": question.strip(),
                        "reference_answers": answer_texts,
                        "retrieval_spans": retrieval_spans,
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
