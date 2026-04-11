from __future__ import annotations

import json
import os
import re
import unicodedata
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Tuple


DATASET_ID = "abhaygupta1266/novelhopqa"
VALID_SPLITS = ("hop_1", "hop_2", "hop_3", "hop_4")
CONFIG_ALIASES = {
    "default": "all",
    "full": "all",
    "all_hops": "all",
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
        raise RuntimeError("datasets package is required for the in-repo NovelHopQA loader.") from exc
    return load_dataset


def _datasets_version_major() -> int | None:
    try:
        raw = str(pkg_version("datasets")).strip()
        return int(raw.split(".", 1)[0]) if raw else None
    except Exception:
        return None


def _normalize_config(config_name: str | None) -> str:
    raw = str(config_name or "all").strip().lower()
    normalized = CONFIG_ALIASES.get(raw, raw)
    if normalized == "all" or normalized in VALID_SPLITS:
        return normalized
    raise ValueError("Unsupported NovelHopQA config_name. Use one of: all/default, hop_1, hop_2, hop_3, hop_4.")


def _selected_splits(mode: str) -> List[str]:
    return list(VALID_SPLITS) if mode == "all" else [mode]


def _load_novelhopqa_split(split_name: str):
    load_dataset = _datasets()
    major = _datasets_version_major()
    kwargs: Dict[str, Any] = {}
    if major is not None and major >= 4:
        kwargs["revision"] = "refs/convert/parquet"
    try:
        return load_dataset(DATASET_ID, "default", split=split_name, **kwargs)
    except TypeError:
        try:
            return load_dataset(DATASET_ID, split=split_name, **kwargs)
        except TypeError:
            return load_dataset(DATASET_ID, split=split_name)
    except Exception:
        return load_dataset(DATASET_ID, "default", split=split_name)


def _safe_component(value: str, default: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return out or default


def _normalize_book_key(value: str | None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\xa0", " ")
        .replace("\ufeff", " ")
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\(\[\{].*?[\)\]\}]", " ", text)
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _query_id(row: Dict[str, Any], split_name: str, index: int) -> str:
    base = row.get("qid") or row.get("question_id") or row.get("id") or index
    return f"{split_name}:{_safe_component(str(base), default=str(index))}"


def _find_top_file_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_file() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _find_child_dir_ci(root: Path, name: str) -> Path | None:
    try:
        for child in root.iterdir():
            if child.is_dir() and child.name.lower() == name.lower():
                return child
    except Exception:
        return None
    return None


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _looks_like_books_root(root: Path) -> bool:
    if not root.exists():
        return False
    if root.is_file():
        return root.suffix.lower() == ".txt"
    if _find_top_file_ci(root, "bookmeta.json") is not None:
        return True
    if _find_child_dir_ci(root, "Books") is not None:
        return True
    try:
        return any(child.is_file() and child.suffix.lower() == ".txt" for child in root.iterdir())
    except Exception:
        return False


def _coerce_books_root(raw_root: Path) -> Path:
    root = raw_root.expanduser().resolve()
    if _looks_like_books_root(root):
        return root
    try:
        for child in root.iterdir():
            if child.is_dir() and _looks_like_books_root(child):
                return child.resolve()
    except Exception:
        pass
    return root


def _resolve_books_root(raw_root: str | os.PathLike[str] | None) -> Path:
    configured_raw = str(raw_root).strip() if raw_root is not None else ""
    env_books_root = str(os.environ.get("NOVELHOPQA_BOOKS_ROOT") or "").strip()
    env_novelqa_root = str(os.environ.get("NOVELQA_DATASET_DIR") or "").strip()
    candidates = [
        ("NOVELHOPQA_BOOKS_ROOT", env_books_root),
        ("NOVELQA_DATASET_DIR", env_novelqa_root),
        ("dataset.books_root", configured_raw),
    ]
    attempts: List[str] = []
    saw_candidate = False
    for label, configured in candidates:
        if not configured:
            continue
        saw_candidate = True
        root = _coerce_books_root(Path(configured))
        if _looks_like_books_root(root):
            return root
        attempts.append(f"{label}={root}")
    if not saw_candidate:
        raise RuntimeError(
            "NovelHopQA whole-book loading requires a books_root. Set dataset.books_root or NOVELHOPQA_BOOKS_ROOT."
        )
    raise RuntimeError(
        "NovelHopQA whole-book loading could not find a usable corpus root at "
        f"{'; '.join(attempts)}."
    )


def _iter_bookmeta_entries(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        books = payload.get("books")
        if isinstance(books, list):
            return [item for item in books if isinstance(item, dict)]
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            row = dict(value)
            row.setdefault("BID", key)
            out.append(row)
    return out


def _candidate_text_paths(root: Path, file_name: str) -> List[Path]:
    p = Path(str(file_name))
    if p.is_absolute():
        return [p]
    if len(p.parts) > 1:
        return [(root / p).resolve()]
    return [
        (root / "Books" / "PublicDomain" / p).resolve(),
        (root / "Books" / "publicdomain" / p).resolve(),
        (root / "Books" / "CopyrightProtected" / p).resolve(),
        (root / "Books" / "copyrightprotected" / p).resolve(),
        (root / "Books" / p).resolve(),
        (root / p).resolve(),
    ]


def _book_doc_id(title: str, fallback: str) -> str:
    return f"book:{_safe_component(title, default=fallback)}"


def _book_title(row: Dict[str, Any]) -> str | None:
    return coerce_to_text(row.get("book") or row.get("title") or row.get("book_title"))


def _iter_title_variants(value: str | None) -> List[str]:
    raw = str(value or "").strip()
    if not raw:
        return []
    variants = [raw]
    cleanup_suffixes = (
        " complete",
        " unabridged",
        " illustrated",
        " with illustrations",
    )
    for sep in (":", ";", ",", " - ", " — ", " – "):
        if sep in raw:
            head = raw.split(sep, 1)[0].strip()
            if head:
                variants.append(head)
    for candidate in list(variants):
        lowered = candidate.casefold()
        for prefix in ("the ", "a ", "an "):
            if lowered.startswith(prefix):
                trimmed = candidate[len(prefix):].strip()
                if trimmed:
                    variants.append(trimmed)
        for suffix in cleanup_suffixes:
            if lowered.endswith(suffix):
                trimmed = candidate[: -len(suffix)].strip(" ,;-:")
                if trimmed:
                    variants.append(trimmed)
    out: List[str] = []
    seen = set()
    for candidate in variants:
        key = _normalize_book_key(candidate)
        if key and key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def _title_like_lines(text: str, limit: int = 5) -> List[str]:
    raw_candidates: List[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.replace("\ufeff", "").replace("\xa0", " ").strip()
        if not line:
            continue
        lowered = line.casefold()
        if "project gutenberg ebook of" in lowered:
            match = re.search(r"project gutenberg ebook of\s+(.+?)(?:,\s+by\b|$)", line, flags=re.IGNORECASE)
            if match:
                title = match.group(1).strip(" .,:;!-")
                if title:
                    raw_candidates.append(title)
                    if len(raw_candidates) >= limit:
                        break
            continue
        if (
            "project gutenberg" in lowered
            or "www.gutenberg.org" in lowered
            or lowered.startswith("***")
            or lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
            or lowered.startswith("release date")
            or lowered.startswith("language:")
            or lowered.startswith("contents")
            or lowered.startswith("chapter ")
            or lowered.startswith("book ")
        ):
            continue
        if len(line) > 160:
            continue
        if sum(ch.isalpha() for ch in line) < 3:
            continue
        raw_candidates.append(line)
        if len(raw_candidates) >= limit:
            break
    out: List[str] = []
    seen = set()
    for candidate in raw_candidates:
        normalized = candidate.strip(" .,:;!-")
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            out.append(normalized)
    stitched_parts: List[str] = []
    total_words = 0
    for candidate in raw_candidates[:3]:
        lowered = candidate.casefold()
        if stitched_parts and (
            lowered.startswith("by ")
            or lowered.startswith("translated by ")
            or lowered.startswith("produced by ")
        ):
            break
        if len(candidate) > 80:
            break
        stitched_parts.append(candidate.strip(" .,:;!-"))
        total_words += len(candidate.split())
        if len(stitched_parts) < 2 or total_words > 12:
            continue
        stitched = " ".join(part for part in stitched_parts if part).strip(" .,:;!-")
        key = stitched.casefold()
        if stitched and key not in seen:
            seen.add(key)
            out.append(stitched)
    return out


def _register_book_aliases(books: Dict[str, Tuple[str, str]], doc_id: str, text: str, candidates: List[str]) -> None:
    for raw in candidates:
        for variant in _iter_title_variants(raw):
            key = _normalize_book_key(variant)
            if key:
                books.setdefault(key, (doc_id, text))


def _load_books_from_root(root: Path) -> Dict[str, Tuple[str, str]]:
    books: Dict[str, Tuple[str, str]] = {}
    bookmeta_path = _find_top_file_ci(root, "bookmeta.json")
    if bookmeta_path is not None and bookmeta_path.exists():
        payload = _read_json(bookmeta_path)
        for i, row in enumerate(_iter_bookmeta_entries(payload)):
            title = coerce_to_text(row.get("title") or row.get("book") or row.get("name"))
            doc_id = str(row.get("BID") or row.get("bid") or _book_doc_id(title or "", fallback=f"book_{i}")).strip()
            txt_name = row.get("txtfile") or row.get("txt_file") or row.get("book_file") or row.get("text_file")
            candidate_names: List[str] = []
            if isinstance(txt_name, str) and txt_name.strip():
                candidate_names.append(txt_name.strip())
            if doc_id:
                candidate_names.extend([f"{doc_id}.txt", f"{doc_id.upper()}.txt", f"{doc_id.lower()}.txt"])
            if not title or not candidate_names:
                continue
            seen_candidates = set()
            for raw_name in candidate_names:
                for candidate in _candidate_text_paths(root, raw_name):
                    if candidate in seen_candidates:
                        continue
                    seen_candidates.add(candidate)
                    if not candidate.exists():
                        continue
                    text = _load_text(candidate)
                    if not text:
                        continue
                    aliases = [title, doc_id, Path(raw_name).stem, candidate.stem, *_title_like_lines(text)]
                    _register_book_aliases(books, doc_id, text, aliases)
                    break
                else:
                    continue
                break
        if books:
            return books

    try:
        txt_files = sorted(p for p in root.rglob("*.txt") if p.is_file())
    except Exception:
        txt_files = []
    for path in txt_files:
        text = _load_text(path)
        if not text:
            continue
        stem = path.stem
        doc_id = _book_doc_id(stem, fallback=stem)
        _register_book_aliases(books, doc_id, text, [stem, *_title_like_lines(text)])
    return books


def _load_novelhopqa_all(mode: str, split: str, books_root: str | os.PathLike[str] | None) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    _ = split
    root = _resolve_books_root(books_root)
    books_by_key = _load_books_from_root(root)
    if not books_by_key:
        raise RuntimeError(f"NovelHopQA whole-book loading found zero books under {root}.")

    docs: Dict[str, str] = {}
    qa_entries: List[Dict[str, Any]] = []
    missing_books = set()

    for split_name in _selected_splits(mode):
        rows = _load_novelhopqa_split(split_name)
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            book_title = _book_title(row)
            if not book_title:
                continue
            book_record = books_by_key.get(_normalize_book_key(book_title))
            if book_record is None:
                missing_books.add(book_title)
                continue
            doc_id, document_text = book_record
            if document_text and doc_id not in docs:
                docs[doc_id] = document_text

            context = coerce_to_text(row.get("context") or row.get("passage") or row.get("document"))
            question = coerce_to_text(row.get("question") or row.get("query"))
            answer = coerce_to_text(row.get("answer") or row.get("gold_answer"))
            if not context or not question or not answer:
                continue
            qa_entries.append(
                {
                    "query_id": _query_id(row, split_name, index),
                    "doc_id": doc_id,
                    "question": question,
                    "book_title": book_title,
                    "gold_context_window": context,
                    "retrieval_span_mode": "window",
                    "reference_answers": [answer],
                    "retrieval_spans": [context],
                    "provided_contexts": [
                        {
                            "doc_id": doc_id,
                            "title": book_title,
                            "text": document_text,
                        }
                    ],
                    "raw_entry": row,
                }
            )

    if missing_books and not os.environ.get("NOVELHOPQA_SUBSET_MODE"):
        sample = ", ".join(sorted(missing_books)[:5])
        raise RuntimeError(
            f"NovelHopQA could not resolve {len(missing_books)} book title(s) under {root}. Example: {sample}"
        )
    return docs, qa_entries


def load_novelhopqa_records(
    *,
    split: str = "test",
    config_name: str | None = "all",
    qa_n: int | str = "all",
    qa_selection_method: str = "first",
    max_documents: int | None = None,
    books_root: str | os.PathLike[str] | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    import random

    mode = _normalize_config(config_name)
    docs_map, qa_entries = _load_novelhopqa_all(mode, split, books_root)
    documents = [{"doc_id": doc_id, "title": doc_id, "text": text} for doc_id, text in docs_map.items()]
    notes: List[str] = []

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
