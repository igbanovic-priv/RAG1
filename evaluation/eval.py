import sys
import json
import math
import os
import argparse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from implementation.answer import answer_question, fetch_context

load_dotenv(override=True)

# --- Model configuration (env + CLI override via set_model_overrides()) ---
ANSWER_LLM_MODEL_NAME = os.getenv("ANSWER_LLM_MODEL_NAME", "gpt-4.1-nano")
JUDGE_LLM_MODEL_NAME = os.getenv("JUDGE_LLM_MODEL_NAME", "gpt-4.1-nano")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")


def set_model_overrides(
    *,
    embedding_model: str | None = None,
    answer_model: str | None = None,
    judge_model: str | None = None,
) -> None:
    """
    Allow evaluator.py (Gradio) or CLI code to override model names at runtime.

    - Also syncs to os.environ so downstream modules that read env can see it.
    - Note: ingest.py is separate; these overrides are mainly for evaluation runs.
    """
    global EMBEDDING_MODEL_NAME, ANSWER_LLM_MODEL_NAME, JUDGE_LLM_MODEL_NAME

    if embedding_model:
        EMBEDDING_MODEL_NAME = embedding_model
        os.environ["EMBEDDING_MODEL_NAME"] = embedding_model

    if answer_model:
        ANSWER_LLM_MODEL_NAME = answer_model
        os.environ["ANSWER_LLM_MODEL_NAME"] = answer_model

    if judge_model:
        JUDGE_LLM_MODEL_NAME = judge_model
        os.environ["JUDGE_LLM_MODEL_NAME"] = judge_model


DEFAULT_DEV_FILE = Path(__file__).resolve().parent.parent / "data" / "hotpotqa_unique_pool1200_dev500.jsonl"


class HotpotExample(BaseModel):
    id: str
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: dict
    context: dict


class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance (HotpotQA supporting facts)."""

    recall: float = Field(description="Supporting-facts recall@k (0..1)")
    mrr: float = Field(description="Mean Reciprocal Rank over supporting facts (0..1)")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary supporting-fact relevance) (0..1)")
    facts_found: int = Field(description="How many supporting facts were found in top-k")
    total_facts: int = Field(description="Total supporting facts for the example")


class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


def _resolve_dataset_path(dataset_path: str | Path | None) -> Path:
    if dataset_path is None:
        return DEFAULT_DEV_FILE
    p = Path(dataset_path).expanduser().resolve()
    return p


def load_dev(dataset_path: str | Path | None = None) -> list[HotpotExample]:
    """Load HotpotQA examples from a JSONL."""
    dev_file = _resolve_dataset_path(dataset_path)

    if not dev_file.exists():
        raise FileNotFoundError(f"Dev file not found: {dev_file}")

    rows: list[HotpotExample] = []
    with dev_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            try:
                rows.append(HotpotExample(**obj))
            except Exception as e:
                raise ValueError(f"Failed parsing {dev_file}:{line_num}: {e}") from e
    return rows


def _supporting_fact_pairs(ex: HotpotExample) -> list[tuple[str, int]]:
    """
    Returns list of (title, sent_id) supporting facts.
    The dataset uses the columnar format:
      supporting_facts: { "title": [...], "sent_id": [...] }
    """
    sf = ex.supporting_facts or {}
    titles = sf.get("title", [])
    sent_ids = sf.get("sent_id", [])

    if not isinstance(titles, list) or not isinstance(sent_ids, list) or len(titles) != len(sent_ids):
        raise ValueError(f"Invalid supporting_facts format for id={ex.id!r}")

    pairs: list[tuple[str, int]] = []
    for t, s in zip(titles, sent_ids):
        if t is None or s is None:
            continue
        pairs.append((str(t), int(s)))
    return pairs


def _chunk_covers_fact(doc, title: str, sent_id: int) -> bool:
    """
    A retrieved chunk covers a supporting fact if:
      - chunk title matches the fact title
      - sent_id is inside [sent_start, sent_end) (sent_end is exclusive)
    """
    md = getattr(doc, "metadata", {}) or {}
    if md.get("title") != title:
        return False

    sent_start = md.get("sent_start")
    sent_end = md.get("sent_end")
    if sent_start is None or sent_end is None:
        return False

    return int(sent_start) <= int(sent_id) < int(sent_end)


def _reciprocal_rank_for_fact(retrieved_docs, title: str, sent_id: int, k: int) -> float:
    """Reciprocal rank for a single supporting fact within top-k."""
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        if _chunk_covers_fact(doc, title, sent_id):
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[int]) -> float:
    """Discounted Cumulative Gain (binary relevance)."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += rel / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def _ndcg_binary(retrieved_docs, facts: list[tuple[str, int]], k: int) -> float:
    """
    nDCG@k with binary relevance:
      relevance(rank i) = 1 if retrieved chunk i covers ANY supporting fact else 0

    Note: Multiple facts covered by the same chunk still count as 1 for that rank.
    """
    relevances: list[int] = []
    for doc in retrieved_docs[:k]:
        is_rel = 0
        for title, sent_id in facts:
            if _chunk_covers_fact(doc, title, sent_id):
                is_rel = 1
                break
        relevances.append(is_rel)

    dcg = _dcg(relevances)

    ideal_relevances = sorted(relevances, reverse=True)
    idcg = _dcg(ideal_relevances)

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(ex: HotpotExample, k: int = 10) -> RetrievalEval:
    """
    Supporting-facts recall@k + MRR@k + nDCG@k:
    - recall@k: fraction of supporting facts covered by top-k retrieved chunks
    - mrr@k: mean reciprocal rank over supporting facts (average of 1/rank for each fact)
    - ndcg@k: binary relevance per rank if the chunk covers at least one supporting fact

    IMPORTANT:
      We retrieve with k so evaluation@k is meaningful for k != default.
    """
    retrieved_docs = fetch_context(ex.question, k=k)

    facts = _supporting_fact_pairs(ex)
    if not facts:
        return RetrievalEval(recall=0.0, mrr=0.0, ndcg=0.0, facts_found=0, total_facts=0)

    found = 0
    rr_scores: list[float] = []
    for title, sent_id in facts:
        rr = _reciprocal_rank_for_fact(retrieved_docs, title, sent_id, k=k)
        rr_scores.append(rr)
        if rr > 0:
            found += 1

    total = len(facts)
    recall = found / total if total else 0.0
    mrr = sum(rr_scores) / total if total else 0.0
    ndcg = _ndcg_binary(retrieved_docs, facts, k=k)

    return RetrievalEval(recall=recall, mrr=mrr, ndcg=ndcg, facts_found=found, total_facts=total)


def retrieval_details(
    ex: HotpotExample, 
    k: int = 10, 
    rerank: bool = False, 
    final_k: int | None = None,
    docs: list | None = None,
) -> tuple[RetrievalEval, list, list[tuple[str, int]]]:
    """
    Returns (retrieval_eval, retrieved_docs_top_k, supporting_facts_pairs).

    Args:
        ex: HotpotQA example
        k: Number of documents to retrieve (ignored if docs is provided)
        rerank: Whether to re-rank (ignored if docs is provided)
        final_k: Number of documents after re-ranking (ignored if docs is provided)
        docs: Pre-retrieved and optionally re-ranked documents. If provided, skips retrieval and re-ranking.

    Returns:
        tuple of (metrics, documents, supporting_facts)
    """
    from implementation.answer import rerank_documents, fetch_context

    # If documents are pre-provided, use them directly
    if docs is None:
        retrieved_docs = fetch_context(ex.question, k=k)

        # Apply re-ranking if enabled
        if rerank and final_k is not None:
            retrieved_docs = rerank_documents(ex.question, retrieved_docs, final_k)
    else:
        retrieved_docs = docs

    top_k = list(retrieved_docs)
    facts = _supporting_fact_pairs(ex)

    # Calculate metrics on (possibly re-ranked) docs
    metrics = evaluate_retrieval_on_docs(ex, retrieved_docs, facts)

    return metrics, top_k, facts


def evaluate_retrieval_on_docs(ex: HotpotExample, retrieved_docs: list, facts: list[tuple[str, int]]) -> RetrievalEval:
    """
    Calculate retrieval metrics on provided documents (supports re-ranked docs).
    """
    k = len(retrieved_docs)

    if not facts:
        return RetrievalEval(recall=0.0, mrr=0.0, ndcg=0.0, facts_found=0, total_facts=0)

    found = 0
    rr_scores: list[float] = []
    for title, sent_id in facts:
        rr = _reciprocal_rank_for_fact(retrieved_docs, title, sent_id, k=k)
        rr_scores.append(rr)
        if rr > 0:
            found += 1

    total = len(facts)
    recall = found / total if total else 0.0
    mrr = sum(rr_scores) / total if total else 0.0
    ndcg = _ndcg_binary(retrieved_docs, facts, k=k)

    return RetrievalEval(recall=recall, mrr=mrr, ndcg=ndcg, facts_found=found, total_facts=total)


def _get_usage_dict(resp: Any) -> dict[str, int]:
    """
    Best-effort extraction of token usage from a LiteLLM response.
    Returns dict with keys: prompt_tokens, completion_tokens, total_tokens (when available).
    """
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
    """
    Best-effort extraction of cost from a LiteLLM response.
    Not all providers/versions include it.
    """
    for key in ("cost", "response_cost", "cost_usd"):
        v = getattr(resp, key, None)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            pass
    return None


def evaluate_answer(
    ex: HotpotExample,
    *,
    k: int = 10,
    structured_context: bool = False,
    prompt_version: str | None = None,
    rerank: bool = False,
    final_k: int | None = None,
) -> tuple[AnswerEval, str, list, dict[str, Any], dict[str, Any]]:
    """
    Evaluate answer quality using LLM-as-a-judge.

    Args:
      k: number of chunks to retrieve initially.
      structured_context: if True, annotate context in answer prompt.
      prompt_version: which prompt version to use.
      rerank: if True, re-rank documents after retrieval.
      final_k: number of documents to use after re-ranking.

    Returns:
      (answer_eval, generated_answer, retrieved_docs, answer_pricing, judge_pricing)
    """
    # Let answer_question handle retrieval and reranking
    generated_answer, retrieved_docs, answer_pricing = answer_question(
        ex.question, 
        k=k, 
        structured_context=structured_context, 
        prompt_version=prompt_version,
        rerank=rerank,
        final_k=final_k,
        # No docs parameter - let answer_question do retrieval
    )

    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{ex.question}

Generated Answer:
{generated_answer}

Reference Answer:
{ex.answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]
    judge_response = completion(model=JUDGE_LLM_MODEL_NAME, messages=judge_messages, temperature=0, response_format=AnswerEval)
    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    #print(f"### Judge prompt messages:\n{json.dumps(judge_messages, indent=2)}\n")
    #print(f"### Judge response:\n{judge_response.choices[0].message.content}\n")

    judge_pricing = {
        "model": JUDGE_LLM_MODEL_NAME,
        "usage": _get_usage_dict(judge_response),
        "cost_usd": _get_cost_usd(judge_response),
    }

    return answer_eval, generated_answer, retrieved_docs, answer_pricing, judge_pricing


def evaluate_retrievals(
    *,
    k: int = 10,
    n_first: int | None = None,
    dataset_path: str | Path | None = None,
):
    """Evaluate retrieval over HotpotQA examples from dataset_path."""
    tests = load_dev(dataset_path=dataset_path)

    limit = len(tests) if n_first is None else max(0, min(n_first, len(tests)))
    tests = tests[:limit]

    total_tests = len(tests)
    if total_tests == 0:
        return
        yield  # pragma: no cover

    for index, ex in enumerate(tests):
        result = evaluate_retrieval(ex, k=k)
        progress = (index + 1) / total_tests
        yield ex, result, progress


def evaluate_answers(
    *,
    n_first: int | None = None,
    dataset_path: str | Path | None = None,
    k: int = 10,
):
    """Evaluate answers over HotpotQA examples from dataset_path."""
    tests = load_dev(dataset_path=dataset_path)

    limit = len(tests) if n_first is None else max(0, min(n_first, len(tests)))
    tests = tests[:limit]

    total_tests = len(tests)
    if total_tests == 0:
        return
        yield  # pragma: no cover

    for index, ex in enumerate(tests):
        result = evaluate_answer(ex, k=k)[0]
        progress = (index + 1) / total_tests
        yield ex, result, progress


def run_cli_evaluation(
    row_number: int, 
    k: int = 10, 
    dataset_path: str | Path | None = None, 
    structured_context: bool = False,
    prompt_version: str | None = None,
    rerank: bool = False,
    final_k: int | None = None,
):
    """
    Run evaluation on a single example from the dataset.

    Args:
        row_number: Index of the example to evaluate
        k: Number of documents to retrieve
        dataset_path: Path to dataset file
        structured_context: Whether to use structured context in answer prompt
        prompt_version: Which prompt version to use
        rerank: Whether to apply LLM reranking
        final_k: Number of documents after reranking (defaults to k)
    """
    if final_k is None:
        final_k = k

    tests = load_dev(dataset_path=dataset_path)

    if row_number < 0 or row_number >= len(tests):
        print(f"Error: row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    ex = tests[row_number]

    print(f"\n{'='*80}")
    print(f"Example #{row_number} (id={ex.id})")
    print(f"{'='*80}")
    print(f"Question: {ex.question}")
    print(f"Type/Level: {ex.type} / {ex.level}")
    print(f"Reference Answer: {ex.answer}")
    print()

    # Supporting facts
    facts = _supporting_fact_pairs(ex)
    print(f"Supporting facts ({len(facts)}):")
    for title, sent_id in facts:
        print(f"  - {title} (sentence {sent_id})")
    print()

    # Answer evaluation - let answer_question handle retrieval
    print(f"Generating answer (k={k}, rerank={rerank}, final_k={final_k})...")
    a_eval, generated_answer, retrieved_docs, answer_pricing, judge_pricing = evaluate_answer(
        ex, 
        k=k, 
        structured_context=structured_context, 
        prompt_version=prompt_version,
        rerank=rerank,
        final_k=final_k
    )

    # Calculate retrieval metrics using the SAME docs that were used for answering
    print(f"Retrieval Metrics:")
    r_metrics, docs, _facts = retrieval_details(ex, k=k, rerank=False, final_k=final_k, docs=retrieved_docs)
    print(f"  Recall@{len(retrieved_docs)}: {r_metrics.recall:.4f} ({r_metrics.facts_found}/{r_metrics.total_facts})")
    print(f"  MRR@{len(retrieved_docs)}: {r_metrics.mrr:.4f}")
    print(f"  nDCG@{len(retrieved_docs)}: {r_metrics.ndcg:.4f}")
    print()

    print("Generated Answer:")
    print(f"  {generated_answer}")
    print()

    print("Answer Evaluation:")
    print(f"  Accuracy:     {a_eval.accuracy:.2f}/5")
    print(f"  Completeness: {a_eval.completeness:.2f}/5")
    print(f"  Relevance:    {a_eval.relevance:.2f}/5")
    print()
    print("Judge Feedback:")
    print(f"  {a_eval.feedback}")
    print()

    # Pricing information
    if answer_pricing.get("usage") or answer_pricing.get("cost_usd") is not None:
        print("Answer LLM Usage:")
        if answer_pricing.get("usage"):
            usage = answer_pricing["usage"]
            print(f"  Tokens: {usage.get('prompt_tokens', 0)} prompt + "
                  f"{usage.get('completion_tokens', 0)} completion = "
                  f"{usage.get('total_tokens', 0)} total")
        if answer_pricing.get("cost_usd") is not None:
            print(f"  Cost: ${answer_pricing['cost_usd']:.6f}")
        print()

    if judge_pricing.get("usage") or judge_pricing.get("cost_usd") is not None:
        print("Judge LLM Usage:")
        if judge_pricing.get("usage"):
            usage = judge_pricing["usage"]
            print(f"  Tokens: {usage.get('prompt_tokens', 0)} prompt + "
                  f"{usage.get('completion_tokens', 0)} completion = "
                  f"{usage.get('total_tokens', 0)} total")
        if judge_pricing.get("cost_usd") is not None:
            print(f"  Cost: ${judge_pricing['cost_usd']:.6f}")
        print()

    print(f"{'='*80}\n")


def main():
    """CLI entry point for running evaluation on a single example."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single HotpotQA example",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "row_number", 
        type=int, 
        help="Index of the example to evaluate (0-based)"
    )
    parser.add_argument(
        "-k", "--k", 
        type=int, 
        default=10, 
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default=None, 
        help=f"Path to dataset file (default: {DEFAULT_DEV_FILE})"
    )
    parser.add_argument(
        "--structured-context", 
        action="store_true", 
        help="Use structured context annotations in answer prompt"
    )
    parser.add_argument(
        "--prompt-version", 
        type=str, 
        default=None, 
        help="Prompt version to use (e.g., 'v1', 'v2')"
    )
    parser.add_argument(
        "--rerank", 
        action="store_true", 
        help="Apply LLM reranking to retrieved documents"
    )
    parser.add_argument(
        "--final-k", 
        type=int, 
        default=None, 
        help="Number of documents after reranking (default: same as k)"
    )

    args = parser.parse_args()

    run_cli_evaluation(
        row_number=args.row_number,
        k=args.k,
        dataset_path=args.dataset,
        structured_context=args.structured_context,
        prompt_version=args.prompt_version,
        rerank=args.rerank,
        final_k=args.final_k,
    )


if __name__ == "__main__":
    main()