#!/usr/bin/env python3
"""Minimal Self-RAG demo for a single document and a single query.

This script:
1. Splits one input document into candidate chunks.
2. Lets Self-RAG decide whether retrieval is needed.
3. If retrieval is needed, scores each chunk as evidence.
4. Prints the final answer and the chunk Self-RAG selected.

Example:
    python test.py \
        --query "What is the difference between llamas and alpacas?" \
        --document-file sample_doc.txt
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
RETRIEVAL_LM_DIR = PROJECT_ROOT / "retrieval_lm"
if str(RETRIEVAL_LM_DIR) not in sys.path:
    sys.path.insert(0, str(RETRIEVAL_LM_DIR))

from utils import PROMPT_DICT, control_tokens, load_special_tokens, postprocess  # noqa: E402

LLM = None
SamplingParams = None


DEFAULT_DOCUMENT = """Llamas and alpacas are both domesticated South American camelids.

Llamas are larger and stronger than alpacas. They were traditionally used as pack animals to carry loads over long distances in the Andes.

Alpacas are smaller than llamas and were bred mainly for their soft fiber. Their fleece is widely used to make textiles and clothing.

Llamas usually have longer faces and banana-shaped ears, while alpacas tend to have shorter faces and straight, spear-shaped ears."""

DEFAULT_QUERY = "What is the difference between llamas and alpacas?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-document Self-RAG demo.")
    parser.add_argument(
        "--model-name",
        default="selfrag/selfrag_llama2_7b",
        help="Hugging Face model name or local model path.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Question or instruction for Self-RAG.",
    )
    parser.add_argument(
        "--document",
        default=None,
        help="Raw input document text. Overrides --document-file when provided.",
    )
    parser.add_argument(
        "--document-file",
        default=None,
        help="Path to a text file containing the source document.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="Approximate maximum number of words per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=25,
        help="Number of overlapping words between chunks when paragraph splitting is needed.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Retrieval probability threshold for adaptive retrieval.",
    )
    parser.add_argument(
        "--download-dir",
        default=".cache",
        help="Where vLLM should store downloaded model weights.",
    )
    parser.add_argument(
        "--dtype",
        default="half",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Comma-separated GPU ids to expose to vLLM, for example '4,5,6,7'.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Number of visible GPUs to shard the model across. Defaults to the number of ids in --cuda-visible-devices.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.75,
        help="Fraction of GPU memory vLLM is allowed to reserve.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many top chunks to print.",
    )
    return parser.parse_args()


def load_document_text(args: argparse.Namespace) -> str:
    if args.document:
        return args.document.strip()
    if args.document_file:
        return Path(args.document_file).read_text(encoding="utf-8").strip()
    return DEFAULT_DOCUMENT


def format_prompt(query: str) -> str:
    return PROMPT_DICT["prompt_no_input"].format(instruction=query)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(text: str) -> List[str]:
    pieces = re.split(r"(?<=[.!?])\s+", normalize_whitespace(text))
    return [piece.strip() for piece in pieces if piece.strip()]


def sliding_word_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = normalize_whitespace(text).split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - chunk_overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            continue
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_document(document: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n", document) if part.strip()]
    chunks: List[str] = []

    for paragraph in raw_paragraphs:
        words = normalize_whitespace(paragraph).split()
        if len(words) <= chunk_size:
            chunks.append(" ".join(words))
            continue

        sentences = sentence_split(paragraph)
        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current and current_len + sentence_len > chunk_size:
                chunks.append(" ".join(current))
                overlap_words = " ".join(current).split()[-chunk_overlap:] if chunk_overlap > 0 else []
                current = [" ".join(overlap_words)] if overlap_words else []
                current_len = len(overlap_words)
            current.append(sentence)
            current_len += sentence_len
        if current:
            chunks.append(" ".join(part for part in current if part))

    deduped = []
    seen = set()
    for chunk in chunks or sliding_word_chunks(document, chunk_size, chunk_overlap):
        cleaned = normalize_whitespace(chunk)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            deduped.append(cleaned)
    return deduped


def strip_control_tokens(text: str) -> str:
    for token in control_tokens:
        text = text.replace(token, "")
    return text.replace("</s>", "").strip()


def safe_exp(logprob: float) -> float:
    if logprob <= -100:
        return 0.0
    return math.exp(logprob)


def get_token_prob(logprob_dict: Dict[int, float], token_id: int) -> float:
    if token_id not in logprob_dict:
        return 0.0
    return safe_exp(float(logprob_dict[token_id]))


def decide_retrieval(
    model,
    prompt: str,
    ret_tokens: Dict[str, int],
    threshold: float,
) -> Tuple[bool, str, float]:
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=25,
        logprobs=32016,
        skip_special_tokens=False,
    )
    pred = model.generate([prompt], sampling_params)[0]
    output = pred.outputs[0]
    first_step = output.logprobs[0]
    retrieval_prob = get_token_prob(first_step, ret_tokens["[Retrieval]"])
    no_retrieval_prob = get_token_prob(first_step, ret_tokens["[No Retrieval]"])
    denom = retrieval_prob + no_retrieval_prob
    ratio = retrieval_prob / denom if denom else 0.0
    return ratio >= threshold, output.text, ratio


def find_first_token_position(token_ids: Sequence[int], candidate_ids: Sequence[int]) -> Optional[int]:
    candidate_set = set(candidate_ids)
    for index, token_id in enumerate(token_ids):
        if token_id in candidate_set:
            return index
    return None


def score_candidate(
    model,
    prompt: str,
    evidence: Dict[str, str],
    max_new_tokens: int,
    rel_tokens: Dict[str, int],
    grd_tokens: Dict[str, int],
    ut_tokens: Dict[str, int],
) -> Dict[str, object]:
    evidence_prompt = (
        f"{prompt}[Retrieval]<paragraph>{evidence['title']}\n{evidence['text']}</paragraph>"
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        logprobs=5000,
        skip_special_tokens=False,
    )
    pred = model.generate([evidence_prompt], sampling_params)[0]
    output = pred.outputs[0]
    token_ids = output.token_ids
    token_logprobs = output.logprobs

    relevance_raw = {
        token: get_token_prob(token_logprobs[0], token_id)
        for token, token_id in rel_tokens.items()
    }
    relevance_sum = sum(relevance_raw.values())
    relevance_score = (
        relevance_raw.get("[Relevant]", 0.0) / relevance_sum if relevance_sum else 0.0
    )

    support_index = find_first_token_position(token_ids, grd_tokens.values())
    support_raw = {}
    if support_index is not None:
        support_raw = {
            token: get_token_prob(token_logprobs[support_index], token_id)
            for token, token_id in grd_tokens.items()
        }
    support_sum = sum(support_raw.values())
    if support_sum:
        ground_score = (
            support_raw.get("[Fully supported]", 0.0) / support_sum
            + 0.5 * support_raw.get("[Partially supported]", 0.0) / support_sum
        )
    else:
        ground_score = 0.0

    utility_index = find_first_token_position(token_ids, ut_tokens.values())
    utility_raw = {}
    if utility_index is not None:
        utility_raw = {
            token: get_token_prob(token_logprobs[utility_index], token_id)
            for token, token_id in ut_tokens.items()
        }
    utility_sum = sum(utility_raw.values())
    if utility_sum:
        utility_weights = {
            "[Utility:1]": -1.0,
            "[Utility:2]": -0.5,
            "[Utility:3]": 0.0,
            "[Utility:4]": 0.5,
            "[Utility:5]": 1.0,
        }
        utility_score = sum(
            utility_weights[token] * (value / utility_sum)
            for token, value in utility_raw.items()
        )
    else:
        utility_score = 0.0

    final_score = relevance_score + ground_score + 0.5 * utility_score
    clean_answer = postprocess(output.text)

    return {
        "answer": clean_answer,
        "raw_answer": output.text,
        "evidence": evidence,
        "score": final_score,
        "relevance_score": relevance_score,
        "ground_score": ground_score,
        "utility_score": utility_score,
    }


def build_evidences(chunks: Sequence[str]) -> List[Dict[str, str]]:
    return [
        {"title": f"Chunk {index + 1}", "text": chunk}
        for index, chunk in enumerate(chunks)
    ]


def parse_visible_devices(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    return [device.strip() for device in raw_value.split(",") if device.strip()]


def resolve_directory(path_value: Optional[str], default_name: str) -> str:
    cleaned = (path_value or "").strip()
    resolved = PROJECT_ROOT / default_name if not cleaned else Path(cleaned).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)
    return str(resolved.resolve())


def configure_gpu_environment(args: argparse.Namespace) -> int:
    cli_visible_devices = parse_visible_devices(args.cuda_visible_devices)
    env_visible_devices = parse_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))

    if cli_visible_devices:
        visible_devices = cli_visible_devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    else:
        visible_devices = env_visible_devices

    if args.tensor_parallel_size is None:
        tensor_parallel_size = len(visible_devices) if visible_devices else 1
    else:
        tensor_parallel_size = args.tensor_parallel_size

    if visible_devices and tensor_parallel_size > len(visible_devices):
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} is larger than the number of visible GPUs "
            f"({len(visible_devices)} from CUDA_VISIBLE_DEVICES={','.join(visible_devices)})."
        )
    if tensor_parallel_size < 1:
        raise ValueError("--tensor-parallel-size must be at least 1.")
    return tensor_parallel_size


def main() -> None:
    global LLM, SamplingParams

    args = parse_args()
    document = load_document_text(args)
    chunks = chunk_document(document, args.chunk_size, args.chunk_overlap)
    evidences = build_evidences(chunks)

    print(f"Loaded {len(evidences)} chunk(s) from the document.\n")
    for evidence in evidences:
        print(f"{evidence['title']}: {evidence['text']}\n")

    tensor_parallel_size = configure_gpu_environment(args)
    download_dir = resolve_directory(args.download_dir, ".cache")
    hf_home = resolve_directory(os.environ.get("HF_HOME"), ".hf_home")
    os.environ["HF_HOME"] = hf_home
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(hf_home) / "transformers"))

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<all visible>")
    print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
    print(f"tensor_parallel_size={tensor_parallel_size}")
    print(f"download_dir={download_dir}")
    print(f"HF_HOME={hf_home}\n")

    from vllm import LLM as VLLMEngine, SamplingParams as VLLMSamplingParams

    LLM = VLLMEngine
    SamplingParams = VLLMSamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    try:
        model = LLM(
            model=args.model_name,
            download_dir=download_dir,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )
    except ValueError as exc:
        message = str(exc)
        if "Free memory on device" in message and "gpu memory utilization" in message.lower():
            raise RuntimeError(
                "vLLM could not start because there is not enough free GPU memory. "
                "Try a lower value such as '--gpu-memory-utilization 0.7' or 0.6, "
                "or free some VRAM from other processes. "
                "If you have multiple GPUs, pass '--cuda-visible-devices 4,5,6,7 --tensor-parallel-size 4'."
            ) from exc
        raise
    except FileNotFoundError as exc:
        raise RuntimeError(
            "vLLM failed during startup because one of its cache or download paths resolved to an empty or missing directory. "
            "The script now creates local cache directories automatically, but if you still see this, run through the provided shell launcher "
            "so CUDA_VISIBLE_DEVICES and cache paths are exported before Python starts."
        ) from exc

    ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_special_tokens(
        tokenizer,
        use_grounding=True,
        use_utility=True,
    )
    prompt = format_prompt(args.query)

    should_retrieve, draft_output, retrieval_ratio = decide_retrieval(
        model=model,
        prompt=prompt,
        ret_tokens=ret_tokens,
        threshold=args.threshold,
    )

    print(f"Query: {args.query}")
    print(f"Retrieval probability: {retrieval_ratio:.4f}")
    print(f"Initial draft: {strip_control_tokens(draft_output)}\n")

    if not should_retrieve:
        final_prompt = prompt + "[No Retrieval]"
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=args.max_new_tokens,
            skip_special_tokens=False,
        )
        pred = model.generate([final_prompt], sampling_params)[0].outputs[0].text
        print("Self-RAG decided not to use retrieval.")
        print(f"Final answer: {postprocess(pred)}")
        return

    ranked_results = [
        score_candidate(
            model=model,
            prompt=prompt,
            evidence=evidence,
            max_new_tokens=args.max_new_tokens,
            rel_tokens=rel_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
        )
        for evidence in evidences
    ]
    ranked_results.sort(key=lambda item: item["score"], reverse=True)

    best = ranked_results[0]
    print("Self-RAG decided to retrieve evidence.\n")
    print("Top retrieved chunks:")
    for index, item in enumerate(ranked_results[: args.top_k], start=1):
        evidence = item["evidence"]
        print(
            f"{index}. {evidence['title']} | score={item['score']:.4f} | "
            f"relevance={item['relevance_score']:.4f} | support={item['ground_score']:.4f} | "
            f"utility={item['utility_score']:.4f}"
        )
        print(f"   {evidence['text']}\n")

    print("Selected chunk:")
    print(best["evidence"]["text"])
    print("\nGenerated answer:")
    print(best["answer"])


if __name__ == "__main__":
    main()
