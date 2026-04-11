from __future__ import annotations

import ast
import json
import os
from importlib.metadata import version as pkg_version
from typing import Any, Dict, Iterable, List, Tuple


DATASET_IDS = ("bigai-nlco/LooGLE", "bigainlco/LooGLE")
LEGACY_CONFIG_ALIASES = {
    "longdep_summarization": "summarization",
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
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("datasets package is required for the in-repo LooGLE loader.") from exc
    return load_dataset, get_dataset_config_names


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        return int(raw.split(".", 1)[0]) if raw else None
    except Exception:
        return None


def _normalize_requested_config(requested: str | None) -> str:
    raw = str(requested or "").strip()
    if not raw:
        return "shortdep_qa"
    return LEGACY_CONFIG_ALIASES.get(raw, raw)


def _config_candidates(requested: str | None) -> List[str]:
    normalized = _normalize_requested_config(requested)
    out = [normalized]
    for old_name, new_name in LEGACY_CONFIG_ALIASES.items():
        if normalized == new_name:
            out.append(old_name)
        elif normalized == old_name:
            out.append(new_name)
    return out


def _get_config_names(dataset_name: str) -> List[str]:
    _, get_dataset_config_names = _datasets()
    major = _datasets_version_major()
    kwargs: Dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        try:
            return list(get_dataset_config_names(dataset_name, **kwargs) or [])
        except TypeError:
            return list(get_dataset_config_names(dataset_name) or [])
    except Exception:
        return []


def _resolve_dataset_and_config(requested_config_name: str | None) -> Tuple[str, str | None]:
    desired_candidates = _config_candidates(requested_config_name)
    for dataset_name in DATASET_IDS:
        configs = _get_config_names(dataset_name)
        if not configs:
            continue
        for cfg in desired_candidates:
            if cfg in configs:
                return dataset_name, cfg
        if "shortdep_qa" in configs:
            return dataset_name, "shortdep_qa"
        return dataset_name, configs[0]
    return DATASET_IDS[0], desired_candidates[0] if desired_candidates else None


def _load_loogle_dataset(*, config_name: str | None):
    load_dataset, _ = _datasets()
    major = _datasets_version_major()
    preferred_name, preferred_cfg = _resolve_dataset_and_config(config_name)
    dataset_order = [preferred_name] + [name for name in DATASET_IDS if name != preferred_name]
    cfg_candidates: List[str | None] = [preferred_cfg]
    for alt in _config_candidates(config_name):
        if alt not in cfg_candidates:
            cfg_candidates.append(alt)

    kwargs_candidates: List[Dict[str, Any]] = []
    if major is not None and major >= 4:
        kwargs_candidates.append({"revision": "refs/convert/parquet"})
    kwargs_candidates.append({})
    if major is None or major < 4:
        kwargs_candidates.append({"trust_remote_code": True})

    def attempt_load(force_offline: bool):
        prev_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        prev_datasets_offline = os.environ.get("HF_DATASETS_OFFLINE")
        offline_download_config = None
        if force_offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            try:
                from datasets import DownloadConfig  # type: ignore

                offline_download_config = DownloadConfig(local_files_only=True)
            except Exception:
                offline_download_config = None
        last_exc: Exception | None = None
        try:
            for dataset_name in dataset_order:
                for cfg_name in cfg_candidates:
                    for kwargs in kwargs_candidates:
                        call_kwargs = dict(kwargs)
                        if force_offline and offline_download_config is not None and "download_config" not in call_kwargs:
                            call_kwargs["download_config"] = offline_download_config
                        try:
                            return (
                                load_dataset(dataset_name, name=cfg_name, **call_kwargs)
                                if cfg_name
                                else load_dataset(dataset_name, **call_kwargs)
                            )
                        except TypeError:
                            try:
                                return load_dataset(dataset_name, name=cfg_name) if cfg_name else load_dataset(dataset_name)
                            except Exception as exc:
                                last_exc = exc
                        except Exception as exc:
                            last_exc = exc
                            continue
            return last_exc
        finally:
            if force_offline:
                if prev_hub_offline is None:
                    os.environ.pop("HF_HUB_OFFLINE", None)
                else:
                    os.environ["HF_HUB_OFFLINE"] = prev_hub_offline
                if prev_datasets_offline is None:
                    os.environ.pop("HF_DATASETS_OFFLINE", None)
                else:
                    os.environ["HF_DATASETS_OFFLINE"] = prev_datasets_offline

    first_try = attempt_load(force_offline=False)
    if not isinstance(first_try, Exception):
        return first_try
    second_try = attempt_load(force_offline=True)
    if not isinstance(second_try, Exception):
        return second_try
    raise RuntimeError("Failed to load the LooGLE dataset from known dataset ids.") from second_try


def _iter_rows_for_split(ds: Any, split: str) -> Iterable[Dict[str, Any]]:
    rows = ds[split]
    for row in rows:
        if isinstance(row, dict):
            yield row


def _extract_document_text(row: Dict[str, Any]) -> str | None:
    for key in ("context", "input", "document", "text", "article", "passage"):
        text = coerce_to_text(row.get(key))
        if text:
            return text
    return None


def _extract_doc_id(row: Dict[str, Any], fallback_index: int) -> str:
    for key in ("doc_id", "document_id", "docid", "title"):
        value = row.get(key)
        if isinstance(value, (str, int)) and str(value).strip():
            return str(value).strip()
    return f"doc_{fallback_index}"


def _to_text_list(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        for item in value:
            out.extend(_to_text_list(item))
        return out
    text = coerce_to_text(value)
    if text:
        out.append(text)
    return out


def _parse_qa_pairs(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, str):
        txt = raw.strip()
        if not txt:
            return []
        try:
            parsed = json.loads(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(txt)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except Exception:
            pass
    return []


def load_loogle_records(
    *,
    split: str = "test",
    config_name: str | None = "shortdep_qa",
    qa_n: int | str = "all",
    qa_selection_method: str = "first",
    max_documents: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    import random

    ds = _load_loogle_dataset(config_name=config_name)
    documents: List[Dict[str, Any]] = []
    qa_entries: List[Dict[str, Any]] = []
    notes: List[str] = []

    for index, row in enumerate(_iter_rows_for_split(ds, split=split)):
        doc_id = _extract_doc_id(row, fallback_index=index)
        document_text = _extract_document_text(row) or ""
        documents.append(
            {
                "doc_id": doc_id,
                "title": str(row.get("title", "")) if row.get("title") is not None else "",
                "text": document_text,
            }
        )

        qa_pairs = _parse_qa_pairs(row.get("qa_pairs"))
        if qa_pairs:
            for pair in qa_pairs:
                question = coerce_to_text(pair.get("Q") or pair.get("question"))
                if not question:
                    continue
                answer_texts = _to_text_list(pair.get("A") or pair.get("answer"))
                spans = _to_text_list(pair.get("S") or pair.get("evidence"))
                if answer_texts or spans:
                    qa_entries.append(
                        {
                            "query_id": str(len(qa_entries)),
                            "doc_id": doc_id,
                            "question": question,
                            "reference_answers": answer_texts,
                            "retrieval_spans": spans,
                            "provided_contexts": [
                                {
                                    "doc_id": doc_id,
                                    "title": str(row.get("title", "")) if row.get("title") is not None else "",
                                    "text": document_text,
                                }
                            ],
                            "raw_entry": row,
                        }
                    )
            continue

        question = coerce_to_text(row.get("question") or row.get("Q") or row.get("query"))
        if not question:
            continue
        answer_texts = _to_text_list(row.get("answer") or row.get("A") or row.get("answers"))
        spans = _to_text_list(row.get("evidence") or row.get("S") or row.get("span"))
        if answer_texts or spans:
            qa_entries.append(
                {
                    "query_id": str(len(qa_entries)),
                    "doc_id": doc_id,
                    "question": question,
                    "reference_answers": answer_texts,
                    "retrieval_spans": spans,
                    "provided_contexts": [
                        {
                            "doc_id": doc_id,
                            "title": str(row.get("title", "")) if row.get("title") is not None else "",
                            "text": document_text,
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
