from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from selfrag_experiment.config import PROJECT_ROOT, dump_yaml_file, load_yaml_file, resolve_run_config
from selfrag_experiment.data import load_and_normalize_dataset
from selfrag_experiment.profiling import capture_resource_usage, utc_now_iso

RETRIEVAL_LM_DIR = PROJECT_ROOT / "retrieval_lm"
if str(RETRIEVAL_LM_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_LM_DIR))


REFLECTION_PATTERN = re.compile(
    r"\[(?:No Retrieval|Retrieval|Continue to Use Evidence|Relevant|Irrelevant|Fully supported|Partially supported|No support / Contradictory|Utility:[1-5])\]"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def overwrite_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_exact_yaml_copy(source_path: str, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(Path(source_path).read_text(encoding="utf-8"), encoding="utf-8")


def parse_visible_devices(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def resolve_directory(path_value: Optional[str], default_name: str) -> str:
    text = (path_value or "").strip()
    target = PROJECT_ROOT / default_name if not text else Path(text).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    return str(target.resolve())


def configure_runtime_environment(model_cfg: Dict[str, Any]) -> int:
    cli_visible = parse_visible_devices(model_cfg.get("cuda_visible_devices"))
    env_visible = parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))

    if cli_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cli_visible)
        visible_devices = cli_visible
    else:
        visible_devices = env_visible

    tensor_parallel_size = model_cfg.get("tensor_parallel_size")
    if tensor_parallel_size is None:
        tensor_parallel_size = len(visible_devices) if visible_devices else 1
    tensor_parallel_size = int(tensor_parallel_size)
    if visible_devices and tensor_parallel_size > len(visible_devices):
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} exceeds visible GPU count={len(visible_devices)}."
        )

    hf_home = resolve_directory(os.environ.get("HF_HOME"), ".hf_home")
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return tensor_parallel_size


def listify_answers(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_control_tokens(text: str) -> str:
    from utils import control_tokens

    output = text
    for token in control_tokens:
        output = output.replace(token, "")
    return output.replace("</s>", "").strip()


def extract_reflection_tokens(text: str) -> List[str]:
    return REFLECTION_PATTERN.findall(text)


def parse_utility_score(text: str) -> float:
    match = re.search(r"\[Utility:(\d)\]", text)
    if not match:
        return 0.0
    return {
        1: -1.0,
        2: -0.5,
        3: 0.0,
        4: 0.5,
        5: 1.0,
    }.get(int(match.group(1)), 0.0)


def marker_score(text: str, positive_marker: str, partial_marker: Optional[str] = None) -> float:
    if positive_marker in text:
        return 1.0
    if partial_marker and partial_marker in text:
        return 0.5
    return 0.0


class SelfRAGGenerator:
    def __init__(self, resolved_cfg: Dict[str, Any]) -> None:
        self.resolved_cfg = resolved_cfg
        self.model_cfg = resolved_cfg["model"]
        self.decoding_cfg = resolved_cfg["decoding"]
        self.selfrag_cfg = resolved_cfg["selfrag"]
        self.tensor_parallel_size = configure_runtime_environment(self.model_cfg)

        from vllm import LLM, SamplingParams
        from utils import PROMPT_DICT, TASK_INST
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.PROMPT_DICT = PROMPT_DICT
        self.TASK_INST = TASK_INST
        tokenizer_name = self.model_cfg.get("tokenizer_name") or self.model_cfg["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left")
        self.download_dir = resolve_directory(self.model_cfg.get("download_dir"), ".cache")
        self.model = LLM(
            model=self.model_cfg["model_name"],
            download_dir=self.download_dir,
            dtype=self.model_cfg["dtype"],
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=float(self.model_cfg["gpu_memory_utilization"]),
        )

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def build_prompt(self, question: str, task_name: Optional[str]) -> str:
        if task_name and task_name in self.TASK_INST:
            instruction = self.TASK_INST[task_name] + "## Input:\n\n" + question
        else:
            instruction = question
        return self.PROMPT_DICT["prompt_no_input"].format(instruction=instruction)

    def _generate(self, prompts: List[str], max_new_tokens: Optional[int] = None) -> List[Any]:
        sampling_params = self.SamplingParams(
            temperature=float(self.decoding_cfg["temperature"]),
            top_p=float(self.decoding_cfg["top_p"]),
            max_tokens=int(max_new_tokens or self.decoding_cfg["max_new_tokens"]),
            skip_special_tokens=False,
        )
        return self.model.generate(prompts, sampling_params)

    def _decide_retrieval(self, prompt: str) -> Tuple[bool, str, str]:
        output = self._generate([prompt], max_new_tokens=min(int(self.decoding_cfg["max_new_tokens"]), 32))[0].outputs[0].text
        if "[Retrieval]" in output and "[No Retrieval]" not in output:
            return True, output, "text-marker:[Retrieval]"
        if "[No Retrieval]" in output and "[Retrieval]" not in output:
            return False, output, "text-marker:[No Retrieval]"
        return "[Retrieval]" in output, output, "text-marker:mixed"

    def _score_output(self, raw_text: str) -> Dict[str, float]:
        relevance_score = marker_score(raw_text, "[Relevant]")
        ground_score = marker_score(raw_text, "[Fully supported]", "[Partially supported]")
        utility_score = parse_utility_score(raw_text)
        final_score = (
            float(self.selfrag_cfg["w_rel"]) * relevance_score
            + float(self.selfrag_cfg["w_sup"]) * ground_score
            + float(self.selfrag_cfg["w_use"]) * utility_score
        )
        return {
            "relevance_score": relevance_score,
            "ground_score": ground_score,
            "utility_score": utility_score,
            "final_score": final_score,
        }

    def run_query(
        self,
        question: str,
        task_name: Optional[str],
        passages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prompt = self.build_prompt(question, task_name)
        decision_started = time.perf_counter()

        mode = self.selfrag_cfg["mode"]
        if mode == "always_retrieve":
            should_retrieve = True
            decision_output = "[Retrieval]"
            decision_signal = "forced:always_retrieve"
        elif mode == "no_retrieval":
            should_retrieve = False
            decision_output = "[No Retrieval]"
            decision_signal = "forced:no_retrieval"
        else:
            should_retrieve, decision_output, decision_signal = self._decide_retrieval(prompt)
        decision_latency_ms = (time.perf_counter() - decision_started) * 1000.0

        generation_started = time.perf_counter()
        candidate_traces: List[Dict[str, Any]] = []
        if should_retrieve and passages:
            augmented_prompts = [
                f"{prompt}[Retrieval]<paragraph>{passage.get('title', '')}\n{passage['text']}</paragraph>"
                for passage in passages
            ]
            outputs = self._generate(augmented_prompts)
            for passage, prediction in zip(passages, outputs):
                raw_text = prediction.outputs[0].text
                candidate_traces.append(
                    {
                        "passage": passage,
                        "raw_output": raw_text,
                        "clean_output": strip_control_tokens(raw_text),
                        "reflection_tokens": extract_reflection_tokens(raw_text),
                        **self._score_output(raw_text),
                    }
                )
            candidate_traces.sort(key=lambda item: item["final_score"], reverse=True)
            selected = candidate_traces[0]
            raw_model_output = selected["raw_output"]
            final_prediction = selected["clean_output"]
            passages_used = [selected["passage"]]
        else:
            final_prompt = prompt + "[No Retrieval]"
            raw_model_output = self._generate([final_prompt])[0].outputs[0].text
            final_prediction = strip_control_tokens(raw_model_output)
            passages_used = []
            candidate_traces = []
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000.0

        context_text = "\n\n".join(
            f"{passage.get('title', '')}\n{passage.get('text', '')}".strip()
            for passage in passages_used
        ).strip()
        return {
            "prompt": prompt,
            "decision_output": decision_output,
            "decision_signal": decision_signal,
            "decision_latency_ms": decision_latency_ms,
            "raw_model_output": raw_model_output,
            "final_prediction": final_prediction,
            "candidate_traces": candidate_traces,
            "passages_used": passages_used,
            "retrieval_triggered": bool(passages_used),
            "number_of_retrieval_steps": 1 if passages_used else 0,
            "number_of_passages_used": len(passages_used),
            "context_text": context_text,
            "context_token_count": self.count_tokens(context_text),
            "answer_token_count": self.count_tokens(final_prediction),
            "generation_latency_ms": generation_latency_ms,
            "retrieval_decisions": [
                {
                    "mode": mode,
                    "triggered": bool(passages_used),
                    "signal": decision_signal,
                    "raw_output": decision_output,
                }
            ],
            "reflection_tokens": extract_reflection_tokens(raw_model_output),
            "generation_steps": [
                {
                    "stage": "decision",
                    "raw_output": decision_output,
                    "signal": decision_signal,
                    "latency_ms": decision_latency_ms,
                },
                {
                    "stage": "generation",
                    "raw_output": raw_model_output,
                    "latency_ms": generation_latency_ms,
                    "candidate_count": len(candidate_traces),
                },
            ],
        }


class ExperimentRunner:
    def __init__(self, dataset_name: str, default_yaml_path: str, resume: bool = False) -> None:
        self.dataset_name = dataset_name
        self.default_yaml_path = default_yaml_path
        self.default_yaml = load_yaml_file(default_yaml_path)
        self.resolved_cfg, self.notes = resolve_run_config(self.default_yaml, dataset_name)
        self.run_root = Path(self.resolved_cfg["run_root"])
        self.resume = resume
        self.paths = {
            "config_dir": self.run_root / "config",
            "selection_dir": self.run_root / "selection",
            "corpus_dir": self.run_root / "corpus",
            "rag_dir": self.run_root / "rag",
            "profiling_dir": self.run_root / "profiling",
            "manifest": self.run_root / "run_manifest.json",
            "default_experiment_yaml": self.run_root / "config" / "default_experiment.yaml",
            "selfrag_run_yaml": self.run_root / "config" / "selfrag_run.yaml",
            "selected_doc_ids": self.run_root / "selection" / "selected_doc_ids.json",
            "qa_entries": self.run_root / "selection" / "qa_entries.json",
            "documents_jsonl": self.run_root / "corpus" / "documents.jsonl",
            "qa_predictions": self.run_root / "rag" / "qa_predictions.jsonl",
            "rag_traces": self.run_root / "rag" / "rag_run_traces.jsonl",
            "query_times": self.run_root / "profiling" / "query_times.jsonl",
            "resource_usage": self.run_root / "profiling" / "resource_usage.jsonl",
        }

    def _ensure_structure(self) -> None:
        for key in ("config_dir", "selection_dir", "corpus_dir", "rag_dir", "profiling_dir"):
            ensure_dir(self.paths[key])

    def _save_configs(self) -> None:
        save_exact_yaml_copy(self.default_yaml_path, self.paths["default_experiment_yaml"])
        dump_yaml_file(self.paths["selfrag_run_yaml"], self.resolved_cfg)

    def _write_selection_and_corpus(
        self,
        qa_entries: List[Dict[str, Any]],
        documents: List[Dict[str, Any]],
        generator: SelfRAGGenerator,
    ) -> None:
        selected_doc_ids = [doc["doc_id"] for doc in documents]
        write_json(self.paths["selected_doc_ids"], selected_doc_ids)

        selection_entries = [
            {
                "query_id": entry["query_id"],
                "doc_id": entry["doc_id"],
                "question": entry["question"],
                "reference_answers": entry["reference_answers"],
                "provided_context_count": len(entry["provided_contexts"]),
            }
            for entry in qa_entries
        ]
        write_json(self.paths["qa_entries"], selection_entries)

        corpus_rows = [
            {
                "doc_id": doc["doc_id"],
                "text": doc["text"],
                "character_count": len(doc["text"]),
                "token_count": generator.count_tokens(doc["text"]),
            }
            for doc in documents
        ]
        overwrite_jsonl(self.paths["documents_jsonl"], corpus_rows)

    def _build_retriever(self) -> Optional[Retriever]:
        retrieval_cfg = self.resolved_cfg["retrieval"]
        if retrieval_cfg["backend"] != "contriever":
            return None
        from passage_retrieval import Retriever

        if not retrieval_cfg["passages"] or not retrieval_cfg["passages_embeddings"]:
            raise ValueError(
                "Contriever retrieval was requested, but retrieval.passages and retrieval.passages_embeddings are not both set."
            )
        args = Namespace(
            model_name_or_path=retrieval_cfg["model_name_or_path"],
            passages=retrieval_cfg["passages"],
            passages_embeddings=retrieval_cfg["passages_embeddings"],
            n_docs=int(retrieval_cfg["n_docs"]),
            save_or_load_index=bool(retrieval_cfg["save_or_load_index"]),
            no_fp16=bool(retrieval_cfg["no_fp16"]),
            question_maxlength=int(retrieval_cfg["question_maxlength"]),
            per_gpu_batch_size=int(retrieval_cfg["per_gpu_batch_size"]),
            lowercase=bool(retrieval_cfg["lowercase"]),
            normalize_text=bool(retrieval_cfg["normalize_text"]),
            indexing_batch_size=int(retrieval_cfg["indexing_batch_size"]),
            n_subquantizers=int(retrieval_cfg["n_subquantizers"]),
            n_bits=int(retrieval_cfg["n_bits"]),
            projection_size=int(retrieval_cfg["projection_size"]),
        )
        retriever = Retriever(args)
        retriever.setup_retriever()
        return retriever

    def _resolve_passages_for_query(
        self,
        qa_entry: Dict[str, Any],
        retriever: Optional[Retriever],
        docs_by_id: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], float]:
        start = time.perf_counter()
        ndocs = int(self.resolved_cfg["selfrag"]["ndocs"])
        if qa_entry["provided_contexts"]:
            passages = qa_entry["provided_contexts"][:ndocs]
            latency_ms = (time.perf_counter() - start) * 1000.0
            return passages, latency_ms
        if retriever is not None:
            passages = retriever.search_document(qa_entry["question"], top_n=ndocs)
            normalized = [
                {
                    "doc_id": str(item.get("id", item.get("doc_id", index))),
                    "title": str(item.get("title", "")),
                    "text": str(item.get("text", "")),
                }
                for index, item in enumerate(passages)
            ]
            latency_ms = (time.perf_counter() - start) * 1000.0
            return normalized, latency_ms
        if qa_entry["doc_id"] in docs_by_id:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return [docs_by_id[qa_entry["doc_id"]]], latency_ms
        latency_ms = (time.perf_counter() - start) * 1000.0
        return [], latency_ms

    def _load_completed_query_ids(self) -> set:
        path = self.paths["qa_predictions"]
        if not self.resume or not path.exists():
            return set()
        completed = set()
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "query_id" in row:
                    completed.add(str(row["query_id"]))
        return completed

    def _package_versions(self) -> Dict[str, Optional[str]]:
        packages = ["torch", "transformers", "vllm", "datasets", "PyYAML"]
        versions: Dict[str, Optional[str]] = {}
        try:
            from importlib.metadata import version
        except Exception:  # pragma: no cover
            return {name: None for name in packages}
        for package in packages:
            try:
                versions[package] = version(package)
            except Exception:
                versions[package] = None
        return versions

    def _git_commit(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return result.stdout.strip() or None

    def _hardware_summary(self) -> Dict[str, Any]:
        summary = {
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        usage = capture_resource_usage("hardware_summary", None, None)
        summary["resource_snapshot"] = usage
        return summary

    def run(self) -> Path:
        self._ensure_structure()
        self._save_configs()

        qa_entries, documents, data_notes = load_and_normalize_dataset(self.resolved_cfg)
        self.notes.extend(data_notes)

        generator = SelfRAGGenerator(self.resolved_cfg)
        docs_by_id = {doc["doc_id"]: doc for doc in documents}
        self._write_selection_and_corpus(qa_entries, documents, generator)

        retriever = self._build_retriever()
        completed_ids = self._load_completed_query_ids()

        if not self.resume:
            for path_key in ("qa_predictions", "rag_traces", "query_times", "resource_usage"):
                if self.paths[path_key].exists():
                    self.paths[path_key].unlink()

        for qa_entry in qa_entries:
            if qa_entry["query_id"] in completed_ids:
                continue

            append_jsonl(self.paths["resource_usage"], capture_resource_usage("before_query", qa_entry["query_id"], qa_entry["doc_id"]))
            passages, retrieval_latency_ms = self._resolve_passages_for_query(qa_entry, retriever, docs_by_id)
            started = time.perf_counter()
            result = generator.run_query(
                question=qa_entry["question"],
                task_name=self.resolved_cfg["dataset_loader"].get("task"),
                passages=passages,
            )
            total_latency_ms = (time.perf_counter() - started) * 1000.0 + retrieval_latency_ms
            append_jsonl(self.paths["resource_usage"], capture_resource_usage("after_query", qa_entry["query_id"], qa_entry["doc_id"]))

            prediction_row = {
                "query_id": qa_entry["query_id"],
                "doc_id": qa_entry["doc_id"],
                "question": qa_entry["question"],
                "prediction": result["final_prediction"],
                "reference_answers": listify_answers(qa_entry["reference_answers"]),
                "context": result["context_text"],
                "context_token_count": result["context_token_count"],
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": result["generation_latency_ms"],
                "total_latency_ms": total_latency_ms,
                "answer_token_count": result["answer_token_count"],
                "model_name": self.resolved_cfg["model"]["model_name"],
                "prompt_metadata": {
                    "task": self.resolved_cfg["dataset_loader"].get("task"),
                    "prompt_template": "prompt_no_input",
                },
                "mode": self.resolved_cfg["selfrag"]["mode"],
                "retrieval_triggered": result["retrieval_triggered"],
                "number_of_retrieval_steps": result["number_of_retrieval_steps"],
                "number_of_passages_used": result["number_of_passages_used"],
            }
            append_jsonl(self.paths["qa_predictions"], prediction_row)

            trace_row = {
                "query_id": qa_entry["query_id"],
                "doc_id": qa_entry["doc_id"],
                "question": qa_entry["question"],
                "mode": self.resolved_cfg["selfrag"]["mode"],
                "raw_model_output": result["raw_model_output"],
                "final_prediction": result["final_prediction"],
                "context": result["context_text"],
                "passages_used": result["passages_used"],
                "retrieval_decisions": result["retrieval_decisions"],
                "critique_reflection_tokens": result["reflection_tokens"],
                "generation_steps": result["generation_steps"],
                "candidate_traces": result["candidate_traces"],
                "context_token_count": result["context_token_count"],
                "answer_token_count": result["answer_token_count"],
                "retrieval_latency_ms": retrieval_latency_ms,
                "generation_latency_ms": result["generation_latency_ms"],
                "total_latency_ms": total_latency_ms,
            }
            append_jsonl(self.paths["rag_traces"], trace_row)

            append_jsonl(
                self.paths["query_times"],
                {
                    "query_id": qa_entry["query_id"],
                    "doc_id": qa_entry["doc_id"],
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "generation_latency_ms": result["generation_latency_ms"],
                    "total_latency_ms": total_latency_ms,
                },
            )

        manifest = {
            "created_at_utc": utc_now_iso(),
            "dataset_name": self.dataset_name,
            "run_name": self.resolved_cfg["run_name"],
            "command": " ".join(sys.argv),
            "git_commit": self._git_commit(),
            "python_version": sys.version,
            "package_versions": self._package_versions(),
            "hardware_summary": self._hardware_summary(),
            "selected_documents_count": len(documents),
            "selected_questions_count": len(qa_entries),
            "artifact_paths": {key: str(path) for key, path in self.paths.items() if key not in {"config_dir", "selection_dir", "corpus_dir", "rag_dir", "profiling_dir"}},
            "config_paths": {
                "default_experiment_yaml": str(self.paths["default_experiment_yaml"]),
                "selfrag_run_yaml": str(self.paths["selfrag_run_yaml"]),
            },
            "notes_about_assumptions_or_ignored_yaml_fields": self.notes,
        }
        write_json(self.paths["manifest"], manifest)
        return self.run_root


def run_experiment(dataset_name: str, default_yaml_path: str, resume: bool = False) -> Path:
    runner = ExperimentRunner(dataset_name=dataset_name, default_yaml_path=default_yaml_path, resume=resume)
    return runner.run()
