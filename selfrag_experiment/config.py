from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML object at {path}, got {type(data).__name__}.")
    return data


def dump_yaml_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _get_path(payload: Dict[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def first_value(payload: Dict[str, Any], candidates: Iterable[str], default: Any = None) -> Any:
    for candidate in candidates:
        value = _get_path(payload, candidate)
        if value is not None:
            return value
    return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_run_config(default_yaml: Dict[str, Any], dataset_name: str) -> Tuple[Dict[str, Any], List[str]]:
    config = deepcopy(default_yaml)
    notes: List[str] = []

    run_name = _string_or_none(
        first_value(
            config,
            [
                "run_name",
                "experiment.run_name",
                "experiment.name",
                "name",
            ],
        )
    )
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        notes.append("No run_name found in the YAML; generated a UTC timestamp-based run name.")

    dataset_loader = {
        "source_type": first_value(
            config,
            [
                "dataset.source_type",
                "dataset.loader",
                "dataset.format",
                "data.source_type",
                "data.loader",
            ],
            default="jsonl",
        ),
        "dataset_name": _string_or_none(first_value(config, ["dataset.name", "data.name"], default=dataset_name)),
        "source": _string_or_none(first_value(config, ["dataset.source", "data.source", "dataset.path", "data.path"])),
        "split": _string_or_none(first_value(config, ["dataset.split", "data.split"], default="train")),
        "subset": _string_or_none(first_value(config, ["dataset.subset", "data.subset"])),
        "config_name": _string_or_none(first_value(config, ["dataset.config_name", "data.config_name"])),
        "data_files": first_value(config, ["dataset.data_files", "data.data_files"]),
        "qa_path": _string_or_none(first_value(config, ["dataset.qa_path", "data.qa_path", "qa.path"])),
        "docs_path": _string_or_none(first_value(config, ["dataset.docs_path", "data.docs_path", "corpus.path"])),
        "question_field": first_value(config, ["dataset.question_field", "data.question_field"], default="question"),
        "answers_field": first_value(
            config,
            ["dataset.answers_field", "data.answers_field", "dataset.reference_answers_field"],
            default="answers",
        ),
        "query_id_field": first_value(config, ["dataset.query_id_field", "data.query_id_field"], default="id"),
        "doc_id_field": first_value(config, ["dataset.doc_id_field", "data.doc_id_field"], default="doc_id"),
        "contexts_field": first_value(
            config,
            ["dataset.contexts_field", "data.contexts_field", "dataset.retrieved_contexts_field"],
            default=None,
        ),
        "document_id_field": first_value(config, ["dataset.document_id_field", "corpus.id_field"], default="doc_id"),
        "document_text_field": first_value(config, ["dataset.document_text_field", "corpus.text_field"], default="text"),
        "document_title_field": first_value(config, ["dataset.document_title_field", "corpus.title_field"], default="title"),
        "max_questions": first_value(
            config,
            ["dataset.sample_size", "dataset.max_questions", "data.sample_size", "evaluation.max_questions"],
        ),
        "max_docs": first_value(config, ["dataset.max_docs", "corpus.max_docs", "retrieval.max_docs"]),
        "qa_n": first_value(config, ["dataset.qa_n"], default="all"),
        "qa_selection_method": first_value(config, ["dataset.qa_selection_method"], default="first"),
        "question_ids": _as_list(first_value(config, ["dataset.question_ids", "selection.question_ids"])),
        "task": _string_or_none(first_value(config, ["dataset.task", "task"])),
    }

    model = {
        "model_name": _string_or_none(
            first_value(config, ["model.name", "model.model_name", "generator.model_name"], default="selfrag/selfrag_llama2_7b")
        ),
        "tokenizer_name": _string_or_none(first_value(config, ["model.tokenizer_name", "generator.tokenizer_name"])),
        "download_dir": _string_or_none(first_value(config, ["model.download_dir", "runtime.download_dir"], default=".cache")),
        "dtype": first_value(config, ["model.dtype", "runtime.dtype"], default="half"),
        "tensor_parallel_size": first_value(config, ["model.tensor_parallel_size", "runtime.tensor_parallel_size"]),
        "gpu_memory_utilization": first_value(
            config,
            ["model.gpu_memory_utilization", "runtime.gpu_memory_utilization"],
            default=0.85,
        ),
        "cuda_visible_devices": _string_or_none(
            first_value(config, ["model.cuda_visible_devices", "runtime.cuda_visible_devices"])
        ),
    }

    decoding = {
        "temperature": first_value(config, ["generation.temperature", "decoding.temperature"], default=0.0),
        "top_p": first_value(config, ["generation.top_p", "decoding.top_p"], default=1.0),
        "max_new_tokens": first_value(
            config,
            ["generation.max_new_tokens", "decoding.max_new_tokens", "model.max_new_tokens"],
            default=128,
        ),
    }

    selfrag = {
        "pipeline_type": first_value(
            config,
            ["selfrag.pipeline_type", "selfrag.variant", "pipeline.type", "experiment.pipeline_type"],
            default="short_form",
        ),
        "mode": first_value(config, ["selfrag.mode", "retrieval.mode"], default="adaptive_retrieval"),
        "ndocs": first_value(config, ["selfrag.ndocs", "retrieval.n_docs", "retrieval.ndocs", "retrieval.retrieve_k"], default=5),
        "threshold": first_value(config, ["selfrag.threshold", "retrieval.threshold"], default=0.2),
        "beam_width": first_value(config, ["selfrag.beam_width"], default=2),
        "max_depth": first_value(config, ["selfrag.max_depth"], default=2),
        "use_grounding": first_value(config, ["selfrag.use_grounding"], default=True),
        "use_utility": first_value(config, ["selfrag.use_utility"], default=True),
        "use_seqscore": first_value(config, ["selfrag.use_seqscore"], default=True),
        "w_rel": first_value(config, ["selfrag.w_rel"], default=1.0),
        "w_sup": first_value(config, ["selfrag.w_sup"], default=1.0),
        "w_use": first_value(config, ["selfrag.w_use"], default=0.5),
        "scoring_strategy": first_value(config, ["selfrag.scoring_strategy"], default="marker"),
    }

    retrieval = {
        "backend": first_value(
            config,
            ["retrieval.backend", "retrieval.runtime", "selfrag.retrieval_backend"],
            default="provided_contexts",
        ),
        "model_name_or_path": _string_or_none(
            first_value(config, ["retrieval.model_name_or_path"], default="facebook/contriever-msmarco")
        ),
        "retrieve_k": first_value(config, ["retrieval.retrieve_k"], default=None),
        "passages": _string_or_none(first_value(config, ["retrieval.passages", "retrieval.passages_path"])),
        "passages_embeddings": _string_or_none(
            first_value(config, ["retrieval.passages_embeddings", "retrieval.embeddings"])
        ),
        "n_docs": first_value(config, ["retrieval.n_docs", "retrieval.ndocs"], default=selfrag["ndocs"]),
        "save_or_load_index": bool(first_value(config, ["retrieval.save_or_load_index"], default=False)),
        "no_fp16": bool(first_value(config, ["retrieval.no_fp16"], default=False)),
        "question_maxlength": first_value(config, ["retrieval.question_maxlength"], default=512),
        "per_gpu_batch_size": first_value(config, ["retrieval.per_gpu_batch_size"], default=64),
        "lowercase": bool(first_value(config, ["retrieval.lowercase"], default=False)),
        "normalize_text": bool(first_value(config, ["retrieval.normalize_text"], default=False)),
        "indexing_batch_size": first_value(config, ["retrieval.indexing_batch_size"], default=1000000),
        "n_subquantizers": first_value(config, ["retrieval.n_subquantizers"], default=0),
        "n_bits": first_value(config, ["retrieval.n_bits"], default=8),
        "projection_size": first_value(config, ["retrieval.projection_size"], default=768),
    }

    profiling = {
        "enabled": bool(first_value(config, ["profiling.enabled"], default=True)),
        "capture_resource_usage": bool(first_value(config, ["profiling.capture_resource_usage"], default=True)),
    }

    run_root = PROJECT_ROOT / "selfrag_runs" / dataset_name / run_name
    resolved = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "run_root": str(run_root),
        "dataset_loader": dataset_loader,
        "model": model,
        "decoding": decoding,
        "selfrag": selfrag,
        "retrieval": retrieval,
        "profiling": profiling,
        "ignored_default_yaml_fields": [],
    }

    if selfrag["pipeline_type"] not in {"short_form", "long_form"}:
        notes.append(
            f"Unrecognized pipeline_type '{selfrag['pipeline_type']}' from the YAML; using the configured value, "
            "but the runner currently implements short_form and a simplified long_form mode."
        )

    if retrieval["backend"] == "provided_contexts" and not dataset_loader["contexts_field"]:
        notes.append(
            "No explicit contexts_field was provided; the runner will auto-detect from ctxs/top_contexts/docs when present."
        )

    native_source_types = {"loogle"}
    if dataset_loader["source"] is None and dataset_loader["qa_path"] is None and str(dataset_loader["source_type"]).lower() not in native_source_types:
        notes.append("No dataset source path was found in the YAML; the CLI must provide a dataset source override or the YAML must be updated.")

    unmatched_top_level = sorted(set(config.keys()) - {"run_name", "name", "experiment", "dataset", "data", "qa", "corpus", "model", "generator", "generation", "decoding", "selfrag", "pipeline", "retrieval", "profiling", "evaluation", "task"})
    resolved["ignored_default_yaml_fields"] = unmatched_top_level
    if unmatched_top_level:
        notes.append(
            "Some top-level YAML fields did not map directly into the SELF-RAG runner config: "
            + ", ".join(unmatched_top_level)
        )

    return resolved, notes
