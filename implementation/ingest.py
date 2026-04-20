import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

# ---- Config ----
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_NAME = str(PROJECT_ROOT / "vector_db")

DATA_DIR = PROJECT_ROOT / "data"
CORPUS_PATH = DATA_DIR / "hotpotqa_unique_pool1200_corpus_titles.jsonl"

DEFAULT_HF_EMBED_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model_name() -> str:
    # Allow overriding via env var (shared with answer.py)
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_HF_EMBED_MODEL)


def get_embeddings(model_name: str | None = None):
    model_name = model_name or get_embedding_model_name()
    if model_name.startswith("text-embedding-"):
        return OpenAIEmbeddings(
            model=model_name,
            chunk_size=16,       # keep requests small
        )
    return HuggingFaceEmbeddings(model_name=model_name)


# Sentence chunking params (start simple; tune later)
SENTS_PER_CHUNK = 6
SENTS_OVERLAP = 1


def load_hotpotqa_corpus_jsonl() -> List[Dict]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus file not found: {CORPUS_PATH}")

    rows: List[Dict] = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            doc_id = obj.get("id")
            if not doc_id:
                raise ValueError(f"Missing 'id' at {CORPUS_PATH}:{line_num}")

            sentences = obj.get("sentences")
            if not isinstance(sentences, list) or not sentences:
                raise ValueError(
                    f"Missing/invalid 'sentences' at {CORPUS_PATH}:{line_num} (id={doc_id!r}). "
                    f"Sentence-based chunking requires a non-empty list."
                )

            rows.append(
                {
                    "id": doc_id,
                    "title": obj.get("title", doc_id),
                    "sentences": sentences,
                }
            )

    print(f"Loaded {len(rows):,} corpus title-docs from {CORPUS_PATH.name}")
    return rows


def _iter_sentence_windows(n_sents: int, window: int, overlap: int) -> List[Tuple[int, int]]:
    if window <= 0:
        raise ValueError("SENTS_PER_CHUNK must be > 0")
    if overlap < 0:
        raise ValueError("SENTS_OVERLAP must be >= 0")
    if overlap >= window:
        raise ValueError("SENTS_OVERLAP must be < SENTS_PER_CHUNK")

    windows: List[Tuple[int, int]] = []
    step = window - overlap

    start = 0
    while start < n_sents:
        end = min(start + window, n_sents)
        windows.append((start, end))
        if end == n_sents:
            break
        start += step

    return windows


def create_sentence_chunks(
    rows: List[Dict],
    sents_per_chunk: int = SENTS_PER_CHUNK,
    sents_overlap: int = SENTS_OVERLAP,
) -> List[Document]:
    chunks: List[Document] = []
    for row in rows:
        doc_id = row["id"]
        title = row["title"]
        sentences = row["sentences"]

        windows = _iter_sentence_windows(len(sentences), sents_per_chunk, sents_overlap)
        for chunk_i, (start, end) in enumerate(windows):
            text = " ".join(sentences[start:end]).strip()
            if not text:
                continue

            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "id": doc_id,
                        "title": title,
                        "chunk_index": chunk_i,
                        "sent_start": start,
                        "sent_end": end,  # exclusive
                    },
                )
            )

    print(
        f"Created {len(chunks):,} sentence-chunks "
        f"(SENTS_PER_CHUNK={sents_per_chunk}, SENTS_OVERLAP={sents_overlap})"
    )
    return chunks


def create_embeddings(chunks: List[Document], persist_directory: str = DB_NAME):
    embeddings = get_embeddings()

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # SAFELY get vector count
    try:
        count = vectorstore._collection.count()
    except Exception:
        count = None

    if count and count > 0:
        sample = vectorstore.get(limit=1, include=["embeddings"])
        dimensions = len(sample["embeddings"][0])
        print(f"Success: {count:,} vectors created with {dimensions:,} dimensions.")
    else:
        print("Warning: Vector store created but is empty.")

    print(f"DB_NAME = {persist_directory}")
    print(f"EMBEDDING_MODEL_NAME = {get_embedding_model_name()}")
    return vectorstore


def main():
    print(f"Corpus path: {CORPUS_PATH}")
    print(f"EMBEDDING_MODEL_NAME = {get_embedding_model_name()}")

    rows = load_hotpotqa_corpus_jsonl()

    chunks = create_sentence_chunks(rows)
    create_embeddings(chunks)

    print("Ingestion complete")


if __name__ == "__main__":
    main()