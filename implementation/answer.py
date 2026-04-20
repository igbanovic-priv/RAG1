from pathlib import Path
import os
import json
from typing import Any

from litellm import completion
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from dotenv import load_dotenv

load_dotenv(override=True)

# --- Model configuration (env + CLI override via set_model_overrides()) ---
ANSWER_LLM_MODEL_NAME = os.getenv("ANSWER_LLM_MODEL_NAME", "gpt-4.1-nano")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v2")
RERANK = os.getenv("RERANK", "false").lower() == "true"
FINAL_K = int(os.getenv("FINAL_K", "10"))


def set_model_overrides(
    *, 
    embedding_model: str | None = None, 
    answer_model: str | None = None,
    prompt_version: str | None = None,
    rerank: bool | None = None,
    final_k: int | None = None
) -> None:
    """
    Allow evaluator.py (Gradio) or CLI code to override model names at runtime.

    - Also syncs to os.environ so other modules that read env can see it.
    """
    global EMBEDDING_MODEL_NAME, ANSWER_LLM_MODEL_NAME, PROMPT_VERSION, RERANK, FINAL_K

    if embedding_model:
        EMBEDDING_MODEL_NAME = embedding_model
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model

    if answer_model:
        ANSWER_LLM_MODEL_NAME = answer_model
        os.environ["ANSWER_LLM_MODEL_NAME"] = answer_model

    if prompt_version:
        PROMPT_VERSION = prompt_version
        os.environ["PROMPT_VERSION"] = prompt_version

    if rerank is not None:
        RERANK = rerank
        os.environ["RERANK"] = "true" if rerank else "false"

    if final_k is not None:
        FINAL_K = final_k
        os.environ["FINAL_K"] = str(final_k)


DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# Choose embeddings based on EMBEDDING_MODEL_NAME.
if EMBEDDING_MODEL_NAME.startswith("text-embedding-"):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
else:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

RETRIEVAL_K = 10

# Define prompt versions
SYSTEM_PROMPTS = {
    "v1": """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
""",
    "v2": """
You are a helpful assistant answering questions based on Wikipedia articles.
Answer the question using ONLY the facts stated in the context below.
- If the context doesn't contain enough information, say "I don't have enough information in the context to answer that.
- Do not speculate or make inferences beyond what's explicitly given.
- Be direct and concise, but ensure your answer is complete.

Context:
{context}
""",
}


def get_system_prompt(prompt_version: str | None = None) -> str:
    """Get the system prompt template for the specified version."""
    #print(f"IGOR: prompt_version: {prompt_version}")
    version = prompt_version or PROMPT_VERSION
    return SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS["v2"])


def retrieve_documents(
    question: str,
    *,
    k: int = RETRIEVAL_K,
    rerank: bool | None = None,
    final_k: int | None = None,
) -> list[Document]:
    """
    Retrieve and optionally rerank documents for a question.

    This is the SINGLE SOURCE OF TRUTH for retrieval logic.
    Used by both answer_question() and retrieval-only evaluation.

    Args:
        question: The question to retrieve documents for
        k: Number of documents to retrieve initially
        rerank: Whether to apply LLM reranking
        final_k: Number of documents after reranking

    Returns:
        List of retrieved (and optionally reranked) documents
    """
    # Use global defaults if not specified
    use_rerank = rerank if rerank is not None else RERANK
    use_final_k = final_k if final_k is not None else FINAL_K

    # Retrieve documents
    docs = fetch_context(question, k=k)

    # Apply re-ranking if enabled
    if use_rerank:
        docs = rerank_documents(question, docs, use_final_k)

    return docs


def fetch_context(question: str, *, k: int = RETRIEVAL_K) -> list[Document]:
    """
    Retrieve relevant context documents for a question.

    Args:
      k: number of chunks to retrieve.

    Returns:
      list[Document] of length up to k.
    """
    vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings, collection_name="langchain")
    retriever = vectorstore.as_retriever()
    return retriever.invoke(question, k=k)


def rerank_documents(question: str, docs: list[Document], final_k: int) -> list[Document]:
    """
    Re-rank documents using LLM to select most relevant chunks.

    Args:
        question: The question to answer
        docs: List of retrieved documents
        final_k: Number of documents to return after re-ranking

    Returns:
        Re-ranked list of top final_k documents
    """
    if not docs:
        return []

    if final_k > len(docs):
        final_k = len(docs)

    # Build prompt with all documents
    docs_text = ""
    for i, doc in enumerate(docs):
        title = doc.metadata.get("title", "Unknown")
        content = doc.page_content
        docs_text += f"\n\n[{i}] Title: {title}\nContent: {content}"

    prompt = f"""Rank these documents by relevance to answering the question.

**CRITICAL INSTRUCTIONS:**

1. **TITLE MATCHING (HIGHEST PRIORITY):**
   - If the question mentions a specific name/title (person, movie, band, etc.),
     documents with that EXACT title should be ranked in top 3
   - Example: Question mentions "High School (2010 film)" → 
     The document titled "High School (2010 film)" MUST be rank #1 or #2
   
2. **MULTI-HOP QUESTIONS:**
   - If the question requires chaining information (e.g., "The star of X won Y for Z"),
     you need MULTIPLE documents in top positions
   - Example: "The star of X won Y" needs:
     - Document about X (to identify the star)
     - Document about the star (to find what they won)
   - Both should be in top 3

3. **AVOID KEYWORD STUFFING:**
   - Don't over-rank documents just because they mention keywords many times
   - A document that mentions "Academy Award" 10 times isn't necessarily better
     than one that mentions it once but has the RIGHT answer
   
4. **DIRECT > VERBOSE:**
   - Brief, direct answers are often better than long, generic documents
   - A specific biography is better than a "List of all actors" document

Question: {question}

Documents:{docs_text}

**Think step-by-step:**
1. What specific entities/titles/names are mentioned in the question?
2. Which documents have those EXACT titles? → Mark as PRIORITY
3. Is this a multi-hop question? What documents are needed for each hop?
4. Rank: Exact title matches first → Then semantic relevance

Return format: {{"ranked_indices": [index1, index2, ...]}}
Return ONLY the JSON, no other text."""

    print(f"IGOR: rerank prompt: {prompt}")
    try:
        response = completion(
            model=ANSWER_LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a document relevance ranking assistant. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        result_text = response.choices[0].message.content.strip()
        # Extract JSON if wrapped in markdown
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        ranked_indices = result.get("ranked_indices", [])

        # Validate indices and return re-ranked docs
        reranked = []
        for idx in ranked_indices[:final_k]:
            if 0 <= idx < len(docs):
                reranked.append(docs[idx])

        # If we didn't get enough valid indices, fill with remaining docs
        if len(reranked) < final_k:
            used_indices = set(ranked_indices[:final_k])
            for i, doc in enumerate(docs):
                if i not in used_indices and len(reranked) < final_k:
                    reranked.append(doc)
        
        return reranked[:final_k]

        
    except Exception as e:
        # On error, fall back to original order
        print(f"Re-ranking failed: {e}. Using original order.")
        return docs[:final_k]


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.

    Note: This assumes history is list[dict] with at least {"role": "...", "content": "..."}.
    """
    prior = "\n".join(m["content"] for m in history if m.get("role") == "user")
    return prior + "\n" + question


def _history_dicts_to_messages(history: list[dict]) -> list[dict]:
    """
    Convert history in the format:
      [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
    into OpenAI/LiteLLM-compatible messages.

    We intentionally only support dict history here (Option A).
    """
    messages: list[dict] = []
    for i, m in enumerate(history):
        if not isinstance(m, dict):
            raise TypeError(
                f"history[{i}] must be a dict like {{'role': 'user', 'content': '...'}}, got {type(m)}"
            )
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            raise ValueError(f"history[{i}]['role'] must be one of user/assistant/system, got {role!r}")
        if content is None:
            content = ""
        messages.append({"role": role, "content": str(content)})
    return messages


def _get_usage_dict(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        usage = getattr(resp, "model_usage", None)
    if usage is None:
        return {}

    try:
        if isinstance(usage, dict):
            return {k: int(v) for k, v in usage.items() if v is not None}
        d = dict(usage)  # type: ignore[arg-type]
        return {k: int(v) for k, v in d.items() if v is not None}
    except Exception:
        return {}


def _get_cost_usd(resp: Any) -> float | None:
    for key in ("cost", "response_cost", "cost_usd"):
        v = getattr(resp, key, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def answer_question(
    question: str,
    history: list[dict] = [],
    *,
    k: int = RETRIEVAL_K,
    structured_context: bool = False,
    prompt_version: str | None = None,
    rerank: bool | None = None,
    final_k: int | None = None,
    docs: list[Document] | None = None,
) -> tuple[str, list[Document], dict]:
    """
    Answer the given question with RAG; return the answer, the context documents, and pricing.

    - Retrieval stays LangChain/Chroma.
    - Answer generation uses LiteLLM (completion).

    Args:
      k: number of chunks to retrieve initially (ignored if docs is provided).
      structured_context: whether to add annotation to context.
      prompt_version: which prompt version to use (defaults to PROMPT_VERSION global).
      rerank: whether to re-rank documents (defaults to RERANK global, ignored if docs is provided).
      final_k: number of chunks to use after re-ranking (defaults to FINAL_K global, ignored if docs is provided).
      docs: Pre-retrieved and optionally re-ranked documents. If provided, skips retrieval and re-ranking.

    Returns:
      (answer_text, docs, pricing)
    where pricing is best-effort info: {"model": ..., "usage": {...}, "cost_usd": ...}
    """
    # Use global defaults if not specified (needed for logging even when docs is provided)
    use_rerank = rerank if rerank is not None else RERANK
    use_final_k = final_k if final_k is not None else FINAL_K

    # If documents are pre-provided, use them directly (skip retrieval/reranking)
    if docs is None:
        combined = combined_question(question, history)
        # Use the dedicated retrieval function
        docs = retrieve_documents(combined, k=k, rerank=use_rerank, final_k=use_final_k)

    if structured_context:
        context = "\n\n".join(
            f"#{i+1} [Title={getattr(doc, 'metadata', {}).get('title', 'N/A')}] {doc.page_content.strip()}"
            for i, doc in enumerate(docs)
        )
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt_template = get_system_prompt(prompt_version)
    system_prompt = system_prompt_template.format(context=context)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_dicts_to_messages(history))
    messages.append({"role": "user", "content": question})

    response = completion(model=ANSWER_LLM_MODEL_NAME, messages=messages, temperature=0)
    answer_text = response.choices[0].message.content

    pricing = {
        "model": ANSWER_LLM_MODEL_NAME,
        "usage": _get_usage_dict(response),
        "cost_usd": _get_cost_usd(response),
    }

    print(f"### answer_question():\n question: {question},\n history: {history},\n k: {k},\n structured_context: {structured_context},\n prompt_version: {prompt_version},\n rerank: {use_rerank},\n final_k: {use_final_k},\n system_prompt: {system_prompt},\n answer_text: {answer_text},\n pricing: {pricing}\n")


    return answer_text, docs, pricing