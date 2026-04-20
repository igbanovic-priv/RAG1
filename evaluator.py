import argparse
import hashlib
import html
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from dotenv import load_dotenv

from evaluation.eval import (
    DEFAULT_DEV_FILE,
    load_dev,
    retrieval_details,
    evaluate_answer,
    set_model_overrides as eval_set_model_overrides,
    EMBEDDING_MODEL_NAME as DEFAULT_EMBEDDING_MODEL_NAME,
    ANSWER_LLM_MODEL_NAME as DEFAULT_ANSWER_LLM_MODEL_NAME,
    JUDGE_LLM_MODEL_NAME as DEFAULT_JUDGE_LLM_MODEL_NAME,
)
from implementation.answer import set_model_overrides as answer_set_model_overrides

load_dotenv(override=True)

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
RECALL_GREEN = 0.9
RECALL_AMBER = 0.75

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


@dataclass
class RetrievedDocLog:
    rank: int
    metadata: dict[str, Any]
    page_content: str


@dataclass
class RetrievalLog:
    k: int
    metrics: dict[str, Any]
    retrieved: list[RetrievedDocLog]


@dataclass
class AnswerLog:
    generated_answer: str
    judge: dict[str, Any]


@dataclass
class ExampleLog:
    example_index: int
    example_id: str
    retrieval: RetrievalLog | None = None
    answer: AnswerLog | None = None


@dataclass
class PricingTotals:
    currency: str = "USD"

    answer_prompt_tokens: int = 0
    answer_completion_tokens: int = 0
    answer_total_tokens: int = 0
    answer_cost_usd: float = 0.0
    answer_cost_known_calls: int = 0
    answer_cost_unknown_calls: int = 0

    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0
    judge_total_tokens: int = 0
    judge_cost_usd: float = 0.0
    judge_cost_known_calls: int = 0
    judge_cost_unknown_calls: int = 0

    def _add_call(self, *, kind: str, pricing: dict[str, Any] | None):
        if not pricing:
            if kind == "answer":
                self.answer_cost_unknown_calls += 1
            else:
                self.judge_cost_unknown_calls += 1
            return

        usage = pricing.get("usage", {}) or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or 0)
        cost = pricing.get("cost_usd", None)

        if kind == "answer":
            self.answer_prompt_tokens += prompt
            self.answer_completion_tokens += completion
            self.answer_total_tokens += total
            if cost is None:
                self.answer_cost_unknown_calls += 1
            else:
                self.answer_cost_usd += float(cost)
                self.answer_cost_known_calls += 1
        else:
            self.judge_prompt_tokens += prompt
            self.judge_completion_tokens += completion
            self.judge_total_tokens += total
            if cost is None:
                self.judge_cost_unknown_calls += 1
            else:
                self.judge_cost_usd += float(cost)
                self.judge_cost_known_calls += 1

    def add_answer_call(self, pricing: dict[str, Any] | None):
        self._add_call(kind="answer", pricing=pricing)

    def add_judge_call(self, pricing: dict[str, Any] | None):
        self._add_call(kind="judge", pricing=pricing)

    def total_cost_usd(self) -> float:
        return float(self.answer_cost_usd + self.judge_cost_usd)

    def to_meta(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "answer": {
                "model": os.getenv("ANSWER_LLM_MODEL_NAME", DEFAULT_ANSWER_LLM_MODEL_NAME),
                "prompt_tokens": self.answer_prompt_tokens,
                "completion_tokens": self.answer_completion_tokens,
                "total_tokens": self.answer_total_tokens,
                "cost_usd": round(self.answer_cost_usd, 6),
                "known_cost_calls": self.answer_cost_known_calls,
                "unknown_cost_calls": self.answer_cost_unknown_calls,
            },
            "judge": {
                "model": os.getenv("JUDGE_LLM_MODEL_NAME", DEFAULT_JUDGE_LLM_MODEL_NAME),
                "prompt_tokens": self.judge_prompt_tokens,
                "completion_tokens": self.judge_completion_tokens,
                "total_tokens": self.judge_total_tokens,
                "cost_usd": round(self.judge_cost_usd, 6),
                "known_cost_calls": self.judge_cost_known_calls,
                "unknown_cost_calls": self.judge_cost_unknown_calls,
            },
            "total_cost_usd": round(self.total_cost_usd(), 6),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _escape(s: Any) -> str:
    return html.escape("" if s is None else str(s))


def _format_duration(seconds: float) -> str:
    """
    Returns duration formatted as Xm Ys (minutes/seconds), with hours when needed.
    """
    if seconds < 0:
        seconds = 0
    total = int(round(seconds))
    s = total % 60
    m = (total // 60) % 60
    h = total // 3600
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def get_color(value: float, metric_type: str) -> str:
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "recall":
        if value >= RECALL_GREEN:
            return "green"
        elif value >= RECALL_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


def format_metric_html(label: str, value: float, metric_type: str, score_format: bool = False) -> str:
    color = get_color(value, metric_type)
    value_str = f"{value:.2f}/5" if score_format else f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{_escape(label)}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{_escape(value_str)}</div>
    </div>
    """


def _safe_md(d: Any) -> dict[str, Any]:
    try:
        return dict(d or {})
    except Exception:
        return {"_unserializable_metadata": str(d)}


def _doc_to_log(rank: int, doc) -> RetrievedDocLog:
    md = _safe_md(getattr(doc, "metadata", {}) or {})
    text = getattr(doc, "page_content", "") or ""
    return RetrievedDocLog(rank=rank, metadata=md, page_content=text)


def _render_supporting_facts(ex) -> str:
    facts = ex.supporting_facts or {}
    titles = facts.get("title", []) or []
    sent_ids = facts.get("sent_id", []) or []

    rows = []
    for t, s in zip(titles, sent_ids):
        rows.append(f"<li><code>{_escape(t)}</code> — sent_id=<b>{_escape(s)}</b></li>")
    if not rows:
        rows = ["<li><i>No supporting facts</i></li>"]

    return "<ul style='margin-top:8px;'>" + "\n".join(rows) + "</ul>"


def _render_retrieved_docs(ex, docs: list[RetrievedDocLog]) -> str:
    facts = ex.supporting_facts or {}
    fact_titles = set((facts.get("title", []) or []))

    blocks = []
    for d in docs:
        md = d.metadata or {}
        title = md.get("title", "")
        sent_start = md.get("sent_start", None)
        sent_end = md.get("sent_end", None)

        is_title_match = title in fact_titles if title else False
        border = "5px solid #28a745" if is_title_match else "5px solid #bbb"
        bg = "#f0fff4" if is_title_match else "#fafafa"

        span = ""
        if sent_start is not None and sent_end is not None:
            span = f"<span style='color:#555;'>sent [{_escape(sent_start)}, {_escape(sent_end)})</span>"

        blocks.append(
            f"""
            <div style="margin: 10px 0; padding: 12px; background:{bg}; border-radius: 8px; border-left: {border};">
              <div style="display:flex; justify-content:space-between; gap:12px;">
                <div style="font-weight:700;">#{_escape(d.rank)} — {_escape(title) if title else "<i>(no title)</i>"}</div>
                <div style="font-size: 12px;">{span}</div>
              </div>
              <pre style="white-space: pre-wrap; margin-top: 8px; font-size: 12px; line-height: 1.35; background: #fff; padding: 10px; border-radius: 6px; border: 1px solid #eee;">{_escape(d.page_content)}</pre>
            </div>
            """
        )

    if not blocks:
        return "<div><i>No retrieved docs</i></div>"

    return "\n".join(blocks)


def _render_example_view(ex, entry: ExampleLog | None) -> str:
    if ex is None:
        return "<div style='color:#999; padding:20px;'>No example loaded.</div>"

    header = f"""
    <div style="padding: 10px 0;">
      <div style="font-size:12px; color:#666;">id: <code>{_escape(ex.id)}</code> | type: <code>{_escape(ex.type)}</code> | level: <code>{_escape(ex.level)}</code></div>
      <div style="font-size:16px; margin-top:8px;"><b>Question</b></div>
      <div style="padding:10px; background:#fff; border:1px solid #eee; border-radius:6px;">{_escape(ex.question)}</div>
      <div style="font-size:16px; margin-top:12px;"><b>Gold answer</b></div>
      <div style="padding:10px; background:#fff; border:1px solid #eee; border-radius:6px;">{_escape(ex.answer)}</div>
    </div>
    """

    sf = f"""
    <div style="margin-top: 10px;">
      <div style="font-size:16px;"><b>Supporting facts</b></div>
      {_render_supporting_facts(ex)}
    </div>
    """

    retrieval_html = "<div style='margin-top:10px;'><b>Retrieval</b><div><i>Not evaluated</i></div></div>"
    answer_html = "<div style='margin-top:10px;'><b>Answer</b><div><i>Not evaluated</i></div></div>"

    if entry and entry.retrieval:
        m = entry.retrieval.metrics
        retrieval_html = f"""
        <div style="margin-top: 14px;">
          <div style="font-size:16px;"><b>Retrieval</b> (k={_escape(entry.retrieval.k)})</div>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px;">
            {format_metric_html("Recall@10", float(m.get("recall", 0.0)), "recall")}
            {format_metric_html("MRR@10", float(m.get("mrr", 0.0)), "mrr")}
            {format_metric_html("nDCG@10", float(m.get("ndcg", 0.0)), "ndcg")}
          </div>
          <div style="margin-top: 10px;">
            <div style="font-size:16px;"><b>Retrieved chunks</b></div>
            {_render_retrieved_docs(ex, entry.retrieval.retrieved)}
          </div>
        </div>
        """

    if entry and entry.answer:
        j = entry.answer.judge
        answer_html = f"""
        <div style="margin-top: 14px;">
          <div style="font-size:16px;"><b>Answer</b></div>
          <div style="font-size:14px; margin-top: 8px;"><b>Generated answer</b></div>
          <div style="padding:10px; background:#fff; border:1px solid #eee; border-radius:6px;">{_escape(entry.answer.generated_answer)}</div>

          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 10px;">
            {format_metric_html("Accuracy", float(j.get("accuracy", 0.0)), "accuracy", score_format=True)}
            {format_metric_html("Completeness", float(j.get("completeness", 0.0)), "completeness", score_format=True)}
            {format_metric_html("Relevance", float(j.get("relevance", 0.0)), "relevance", score_format=True)}
          </div>

          <div style="font-size:14px; margin-top: 8px;"><b>Judge feedback</b></div>
          <div style="padding:10px; background:#fff; border:1px solid #eee; border-radius:6px; white-space: pre-wrap;">{_escape(j.get("feedback", ""))}</div>
        </div>
        """

    return header + sf + retrieval_html + answer_html


def _make_run_id(run_name: str | None) -> str:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if run_name:
        safe = "".join(c for c in run_name if c.isalnum() or c in ("-", "_")).strip("_-")
        if safe:
            return f"{safe}_{ts}"
    return ts


def _write_json(path: Path, obj: Any):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, obj: Any):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _entry_from_dict(d: dict[str, Any]) -> ExampleLog:
    retrieval = None
    if d.get("retrieval") is not None:
        r = d["retrieval"]
        retrieved = [
            RetrievedDocLog(
                rank=int(rd["rank"]),
                metadata=rd.get("metadata", {}) or {},
                page_content=rd.get("page_content", "") or "",
            )
            for rd in (r.get("retrieved") or [])
        ]
        retrieval = RetrievalLog(k=int(r.get("k", 10)), metrics=r.get("metrics", {}) or {}, retrieved=retrieved)

    answer = None
    if d.get("answer") is not None:
        a = d["answer"]
        answer = AnswerLog(generated_answer=a.get("generated_answer", "") or "", judge=a.get("judge", {}) or {})

    return ExampleLog(
        example_index=int(d["example_index"]),
        example_id=str(d.get("example_id", "")),
        retrieval=retrieval,
        answer=answer,
    )


def _summarize_overall_retrieval(entries: list[ExampleLog]) -> tuple[float, float, float]:
    rs = [e.retrieval for e in entries if e.retrieval is not None]
    if not rs:
        return 0.0, 0.0, 0.0
    n = len(rs)
    recalls = [float(r.metrics.get("recall", 0.0)) for r in rs]
    mrrs = [float(r.metrics.get("mrr", 0.0)) for r in rs]
    ndcgs = [float(r.metrics.get("ndcg", 0.0)) for r in rs]
    return sum(recalls) / n, sum(mrrs) / n, sum(ndcgs) / n


def _summarize_overall_answer(entries: list[ExampleLog]) -> tuple[float, float, float, int]:
    answered = [e for e in entries if e.answer is not None and isinstance(e.answer.judge, dict)]
    if not answered:
        return 0.0, 0.0, 0.0, 0

    accs: list[float] = []
    comps: list[float] = []
    rels: list[float] = []

    for e in answered:
        j = e.answer.judge or {}
        try:
            accs.append(float(j.get("accuracy", 0.0)))
            comps.append(float(j.get("completeness", 0.0)))
            rels.append(float(j.get("relevance", 0.0)))
        except Exception:
            continue

    n = len(accs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0

    return sum(accs) / n, sum(comps) / n, sum(rels) / n, n


def render_current_example(entries: list[ExampleLog], dataset_path: Path, idx: int) -> tuple[str, str]:
    if not entries:
        return "<div style='padding:20px; color:#999;'>No results loaded.</div>", "0 / 0"

    idx = max(0, min(idx, len(entries) - 1))
    entry = entries[idx]

    tests = load_dev(dataset_path=dataset_path)
    if entry.example_index < 0 or entry.example_index >= len(tests):
        return (
            "<div style='padding:20px; color:#b91c1c;'>Example not found in dataset for this index.</div>",
            f"{idx+1} / {len(entries)}",
        )

    ex = tests[entry.example_index]
    html_view = _render_example_view(ex, entry)
    counter = f"{idx+1} / {len(entries)}"
    return html_view, counter


def prev_index(entries: list[ExampleLog], idx: int) -> int:
    if not entries:
        return 0
    return max(0, idx - 1)


def next_index(entries: list[ExampleLog], idx: int) -> int:
    if not entries:
        return 0
    return min(len(entries) - 1, idx + 1)


def _resolve_uploaded_file_to_path(file_obj) -> Path | None:
    if file_obj is None:
        return None

    if isinstance(file_obj, str):
        return Path(file_obj).expanduser().resolve()

    p = getattr(file_obj, "path", None)
    if p:
        return Path(p).expanduser().resolve()

    name = getattr(file_obj, "name", None)
    if name:
        return Path(name).expanduser().resolve()

    if isinstance(file_obj, dict):
        if file_obj.get("path"):
            return Path(file_obj["path"]).expanduser().resolve()
        if file_obj.get("name"):
            return Path(file_obj["name"]).expanduser().resolve()

    raise ValueError(f"Unrecognized file object from gr.File: {type(file_obj)}")


def _list_run_dirs(runs_dir: Path) -> list[str]:
    if not runs_dir.exists():
        return []
    out: list[str] = []
    for p in sorted(runs_dir.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if (p / "meta.json").exists() and (p / "results.jsonl").exists():
            out.append(str(p))
    return out


def _current_models_meta() -> dict[str, str]:
    return {
        "embedding_model": os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME),
        "answer_model": os.getenv("ANSWER_LLM_MODEL_NAME", DEFAULT_ANSWER_LLM_MODEL_NAME),
        "judge_model": os.getenv("JUDGE_LLM_MODEL_NAME", DEFAULT_JUDGE_LLM_MODEL_NAME),
    }


def _run_mode_meta(retrieval_only: bool, rerank: bool = False, final_k: int | None = None) -> dict[str, Any]:
    meta = {"mode": "retrieval_only" if retrieval_only else "retrieval_and_answer"}
    if rerank:
        meta["rerank"] = True
        meta["final_k"] = final_k
    return meta


def run_evaluation(
    *,
    dataset_path: Path,
    n_first: int | None,
    k: int,
    runs_dir: Path,
    run_name: str | None,
    retrieval_only: bool,
    structured_context: bool = False,
    prompt_version: str = "v2",
    rerank: bool = False,
    final_k: int | None = None,
    progress=gr.Progress(),
):
    run_started_at = datetime.now()

    if final_k is None:
        final_k = k

    tests = load_dev(dataset_path=dataset_path)
    limit = len(tests) if n_first is None else max(0, min(n_first, len(tests)))
    tests = tests[:limit]

    run_id = _make_run_id(run_name)
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    meta_path = run_dir / "meta.json"
    results_path = run_dir / "results.jsonl"

    dataset_hash = sha256_file(dataset_path)

    pricing_totals = PricingTotals()

    meta = {
        "run_id": run_id,
        "created_at": run_started_at.isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "k": k,
        "n_first": 0 if n_first is None else n_first,
        **_current_models_meta(),
        **_run_mode_meta(retrieval_only, rerank=rerank, final_k=final_k),
        "structured_context": structured_context,
        "prompt_version": prompt_version,
    }
    _write_json(meta_path, meta)

    entries: list[ExampleLog] = []

    total = len(tests)
    if total == 0:
        run_finished_at = datetime.now()
        duration_s = (run_finished_at - run_started_at).total_seconds()

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["summary"] = {"examples": 0}
        meta["duration"] = {
            "started_at": run_started_at.isoformat(timespec="seconds"),
            "finished_at": run_finished_at.isoformat(timespec="seconds"),
            "seconds": duration_s,
            "human": _format_duration(duration_s),
        }
        meta["pricing"] = pricing_totals.to_meta()
        _write_json(meta_path, meta)

        summary_html = "<div style='padding:20px; color:#999;'>No examples selected.</div>"
        df_retrieval = pd.DataFrame(
            [
                {"Metric": "Recall@10", "Value": 0.0},
                {"Metric": "MRR@10", "Value": 0.0},
                {"Metric": "nDCG@10", "Value": 0.0},
            ]
        )
        df_answer = pd.DataFrame(
            [
                {"Metric": "Accuracy (avg)", "Value": 0.0},
                {"Metric": "Completeness (avg)", "Value": 0.0},
                {"Metric": "Relevance (avg)", "Value": 0.0},
            ]
        )
        return (
            summary_html,
            df_retrieval,
            df_answer,
            entries,
            0,
            str(dataset_path),
            "<div style='padding:20px; color:#999;'>No example loaded.</div>",
            "0 / 0",
            f"Run saved to: {run_dir}",
        )

    for idx, ex in enumerate(tests):
        progress((idx + 1) / total, desc=f"Evaluating example {idx + 1}/{total}...")

        answer_log = None
        if not retrieval_only:
            # Generate answer - let answer_question handle retrieval and reranking
            a_eval, generated_answer, retrieved_docs, answer_pricing, judge_pricing = evaluate_answer(
                ex, k=k, structured_context=structured_context, prompt_version=prompt_version,
                rerank=rerank, final_k=final_k
            )
            pricing_totals.add_answer_call(answer_pricing)
            pricing_totals.add_judge_call(judge_pricing)

            # Calculate retrieval metrics using the SAME docs that were used for answering
            r_metrics, docs, _facts = retrieval_details(ex, k=k, rerank=False, final_k=final_k, docs=retrieved_docs)

            answer_log = AnswerLog(generated_answer=generated_answer, judge=a_eval.model_dump())
        else:
            # Retrieval only mode - use the dedicated retrieval function
            from implementation.answer import retrieve_documents

            retrieved_docs = retrieve_documents(ex.question, k=k, rerank=rerank, final_k=final_k)

            r_metrics, docs, _facts = retrieval_details(ex, k=k, rerank=False, final_k=final_k, docs=retrieved_docs)

        retrieved_logs = [_doc_to_log(rank=i + 1, doc=d) for i, d in enumerate(docs)]
        retrieval_log = RetrievalLog(k=k, metrics=r_metrics.model_dump(), retrieved=retrieved_logs)

        entry = ExampleLog(example_index=idx, example_id=ex.id, retrieval=retrieval_log, answer=answer_log)
        entries.append(entry)

        _append_jsonl(results_path, asdict(entry))

    avg_recall, avg_mrr, avg_ndcg = _summarize_overall_retrieval(entries)
    avg_acc, avg_comp, avg_rel, n_answered = _summarize_overall_answer(entries)

    run_finished_at = datetime.now()
    duration_s = (run_finished_at - run_started_at).total_seconds()
    duration_human = _format_duration(duration_s)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["pricing"] = pricing_totals.to_meta()
    meta["summary"] = {
        "examples": len(entries),
        "retrieval": {
            "k": k,
            "avg_recall": avg_recall,
            "avg_mrr": avg_mrr,
            "avg_ndcg": avg_ndcg,
        },
        "answer": {
            "n_answered": n_answered,
            "avg_accuracy": avg_acc,
            "avg_completeness": avg_comp,
            "avg_relevance": avg_rel,
        },
    }
    meta["duration"] = {
        "started_at": run_started_at.isoformat(timespec="seconds"),
        "finished_at": run_finished_at.isoformat(timespec="seconds"),
        "seconds": duration_s,
        "human": duration_human,
    }
    _write_json(meta_path, meta)

    models = _current_models_meta()
    mode = "retrieval_only" if retrieval_only else "retrieval_and_answer"
    mode_info = f"Mode: <code>{_escape(mode)}</code>"
    if rerank and not retrieval_only:
        mode_info += f" | Re-rank: <b>ON</b> (final_k={_escape(final_k)})"
    else:
        mode_info += " | Re-rank: <b>OFF</b>"

    pricing = meta.get("pricing", {}) or {}
    total_cost = pricing.get("total_cost_usd", None)
    answer_cost = (pricing.get("answer", {}) or {}).get("cost_usd", None)
    judge_cost = (pricing.get("judge", {}) or {}).get("cost_usd", None)

    pricing_line = ""
    if total_cost is not None and not retrieval_only:
        pricing_line = (
            "<div style='font-size: 13px; color: #334; margin-top: 6px;'>"
            f"Cost: <code>${_escape(total_cost)}</code>"
            f" (answer=${_escape(answer_cost)} | judge=${_escape(judge_cost)})"
            "</div>"
        )

    duration_line = f"<div style='font-size: 13px; color: #334; margin-top: 6px;'>Duration: <code>{_escape(duration_human)}</code></div>"

    answer_summary_html = ""
    if not retrieval_only:
        answer_summary_html = f"""
        <div style="margin-top: 10px;">
          <div style="font-size:16px;"><b>Answer metrics (overall)</b> <span style="color:#666; font-size: 12px;">(n={_escape(n_answered)})</span></div>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px;">
            {format_metric_html("Accuracy (avg)", avg_acc, "accuracy", score_format=True)}
            {format_metric_html("Completeness (avg)", avg_comp, "completeness", score_format=True)}
            {format_metric_html("Relevance (avg)", avg_rel, "relevance", score_format=True)}
          </div>
        </div>
        """

    summary_html = f"""
    <div style="padding: 0;">
        <div style="margin: 10px 0; padding: 10px; background-color: #eef; border-radius: 6px; border: 1px solid #dde;">
            <div style="font-size: 14px; color: #334;">Run: <code>{_escape(run_id)}</code> saved to <code>{_escape(run_dir)}</code></div>
            <div style="font-size: 14px; color: #334;">Dataset: <code>{_escape(dataset_path)}</code></div>
            <div style="font-size: 14px; color: #334;">{mode_info} | Limit: first <b>{_escape('all' if n_first is None else n_first)}</b> examples | k=<b>{_escape(k)}</b></div>
            <div style="font-size: 13px; color: #334; margin-top: 6px;">
              Models:
              <code>embed={_escape(models['embedding_model'])}</code>,
              <code>answer={_escape(models['answer_model'])}</code>,
              <code>judge={_escape(models['judge_model'])}</code>
            </div>
            <div style="font-size: 13px; color: #334; margin-top: 6px;">Prompt version: <code>{_escape(prompt_version)}</code></div>
            {duration_line}
            {pricing_line}
        </div>

        <div style="margin-top: 10px;">
          <div style="font-size:16px;"><b>Retrieval metrics (overall)</b></div>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px;">
            {format_metric_html("Supporting-Facts Recall@10", avg_recall, "recall")}
            {format_metric_html("Mean Reciprocal Rank (MRR@10)", avg_mrr, "mrr")}
            {format_metric_html("Normalized DCG (nDCG@10)", avg_ndcg, "ndcg")}
          </div>
        </div>

        {answer_summary_html}

        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {len(entries)} examples</span>
        </div>
    </div>
    """

    df_retrieval = pd.DataFrame(
        [
            {"Metric": "Recall@10", "Value": avg_recall},
            {"Metric": "MRR@10", "Value": avg_mrr},
            {"Metric": "nDCG@10", "Value": avg_ndcg},
        ]
    )
    df_answer = pd.DataFrame(
        [
            {"Metric": "Accuracy (avg)", "Value": avg_acc},
            {"Metric": "Completeness (avg)", "Value": avg_comp},
            {"Metric": "Relevance (avg)", "Value": avg_rel},
        ]
    )

    html_view, cnt = render_current_example(entries, dataset_path, 0)

    return summary_html, df_retrieval, df_answer, entries, 0, str(dataset_path), html_view, cnt, f"Run saved to: {run_dir}"


def load_run(*, run_dir: Path, dataset_override_path: Path | None):
    meta_path = run_dir / "meta.json"
    results_path = run_dir / "results.jsonl"

    if not meta_path.exists() or not results_path.exists():
        raise FileNotFoundError(f"Invalid run dir (expected meta.json and results.jsonl): {run_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    dataset_path = Path(meta["dataset_path"]).expanduser().resolve()
    if dataset_override_path is not None:
        dataset_path = dataset_override_path

    warn = ""
    try:
        actual = sha256_file(dataset_path)
        expected = meta.get("dataset_sha256", "")
        if expected and actual != expected:
            warn = "Dataset SHA256 mismatch vs run metadata. Results may not align with questions/facts."
    except Exception:
        warn = "Could not compute dataset hash for verification."

    _ = load_dev(dataset_path=dataset_path)  # validate readable

    rows = _load_jsonl(results_path)
    entries = [_entry_from_dict(r) for r in rows]

    avg_recall, avg_mrr, avg_ndcg = _summarize_overall_retrieval(entries)
    avg_acc, avg_comp, avg_rel, n_answered = _summarize_overall_answer(entries)

    embed_m = meta.get("embedding_model", "")
    answer_m = meta.get("answer_model", "")
    judge_m = meta.get("judge_model", "")
    mode = meta.get("mode", "")
    prompt_version = meta.get("prompt_version", "unknown")

    pricing = meta.get("pricing", {}) or {}
    total_cost = pricing.get("total_cost_usd", None)
    answer_cost = (pricing.get("answer", {}) or {}).get("cost_usd", None)
    judge_cost = (pricing.get("judge", {}) or {}).get("cost_usd", None)

    pricing_line = ""
    if total_cost is not None:
        pricing_line = (
            "<div style='font-size: 13px; color: #334; margin-top: 6px;'>"
            f"Cost: <code>${_escape(total_cost)}</code>"
            f" (answer=${_escape(answer_cost)} | judge=${_escape(judge_cost)})"
            "</div>"
        )

    duration = meta.get("duration", {}) or {}
    duration_human = duration.get("human", "")
    duration_line = ""
    if duration_human:
        duration_line = f"<div style='font-size: 13px; color: #334; margin-top: 6px;'>Duration: <code>{_escape(duration_human)}</code></div>"

    answer_summary_html = ""
    if mode != "retrieval_only":
        answer_summary_html = f"""
        <div style="margin-top: 10px;">
          <div style="font-size:16px;"><b>Answer metrics (overall)</b> <span style="color:#666; font-size: 12px;">(n={_escape(n_answered)})</span></div>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px;">
            {format_metric_html("Accuracy (avg)", avg_acc, "accuracy", score_format=True)}
            {format_metric_html("Completeness (avg)", avg_comp, "completeness", score_format=True)}
            {format_metric_html("Relevance (avg)", avg_rel, "relevance", score_format=True)}
          </div>
        </div>
        """

    summary_html = f"""
    <div style="padding: 0;">
        <div style="margin: 10px 0; padding: 10px; background-color: #eef; border-radius: 6px; border: 1px solid #dde;">
            <div style="font-size: 14px; color: #334;">Loaded run: <code>{_escape(meta.get('run_id',''))}</code> from <code>{_escape(run_dir)}</code></div>
            <div style="font-size: 14px; color: #334;">Dataset: <code>{_escape(dataset_path)}</code></div>
            <div style="font-size: 14px; color: #334;">Mode: <code>{_escape(mode)}</code></div>
            <div style="font-size: 14px; color: #b45309;">{_escape(warn)}</div>
            <div style="font-size: 13px; color: #334; margin-top: 6px;">
              Models:
              <code>embed={_escape(embed_m)}</code>,
              <code>answer={_escape(answer_m)}</code>,
              <code>judge={_escape(judge_m)}</code>
            </div>
            <div style="font-size: 13px; color: #334; margin-top: 6px;">Prompt version: <code>{_escape(prompt_version)}</code></div>
            {duration_line}
            {pricing_line}
        </div>

        <div style="margin-top: 10px;">
          <div style="font-size:16px;"><b>Retrieval metrics (overall)</b></div>
          <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 8px;">
            {format_metric_html("Supporting-Facts Recall@10", avg_recall, "recall")}
            {format_metric_html("Mean Reciprocal Rank (MRR@10)", avg_mrr, "mrr")}
            {format_metric_html("Normalized DCG (nDCG@10)", avg_ndcg, "ndcg")}
          </div>
        </div>

        {answer_summary_html}

        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Loaded: {len(entries)} examples</span>
        </div>
    </div>
    """

    df_retrieval = pd.DataFrame(
        [
            {"Metric": "Recall@10", "Value": avg_recall},
            {"Metric": "MRR@10", "Value": avg_mrr},
            {"Metric": "nDCG@10", "Value": avg_ndcg},
        ]
    )
    df_answer = pd.DataFrame(
        [
            {"Metric": "Accuracy (avg)", "Value": avg_acc},
            {"Metric": "Completeness (avg)", "Value": avg_comp},
            {"Metric": "Relevance (avg)", "Value": avg_rel},
        ]
    )

    html_view, cnt = render_current_example(entries, dataset_path, 0)

    return (
        summary_html,
        df_retrieval,
        df_answer,
        entries,
        0,
        str(dataset_path),
        html_view,
        cnt,
        f"Loaded run from: {run_dir}",
    )


def main():
    parser = argparse.ArgumentParser(description="Gradio RAG evaluation dashboard + run viewer.")
    parser.add_argument("--first", type=int, default=0, help="Evaluate only the first N examples. 0 = all.")
    parser.add_argument("--k", type=int, default=10, help="Top-k for retrieval metrics/logging.")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory to store run folders.")
    parser.add_argument("--run-name", type=str, default="", help="Optional run name prefix.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DEV_FILE),
        help="Default dataset path (can be overridden in UI).",
    )

    # Model overrides (CLI)
    parser.add_argument("--embedding-model", type=str, default="", help="Override embedding model name.")
    parser.add_argument("--answer-model", type=str, default="", help="Override answer LLM model name.")
    parser.add_argument("--judge-model", type=str, default="", help="Override judge LLM model name.")

    # Run mode (CLI)
    parser.add_argument("--retrieval-only", action="store_true", help="If set, run retrieval-only evaluation.")

    # Re-ranking options
    parser.add_argument("--rerank", action="store_true", help="If set, use LLM to re-rank retrieved documents.")
    parser.add_argument("--final-k", type=int, default=None, help="Number of documents after re-ranking (default: same as k).")

    # Structured context flag
    parser.add_argument(
        "--structured-context",
        action="store_true",
        help="If set, annotate context chunks (#N, Title=...) in answer prompting",
    )

    # Prompt version tracking
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v2",
        help="Prompt version identifier for tracking (e.g., 'v1', 'v2', 'v2-concise'). Saved in run metadata.",
    )

    # Server configuration
    parser.add_argument("--port", type=int, default=None, help="Port for Gradio server (default: auto-assign starting from 7860).")

    args = parser.parse_args()

    # Set default final_k to k if not specified
    final_k = args.final_k if args.final_k is not None else args.k

    # Apply CLI overrides (also updates os.environ)
    eval_set_model_overrides(
        embedding_model=args.embedding_model or None,
        answer_model=args.answer_model or None,
        judge_model=args.judge_model or None,
    )
    answer_set_model_overrides(
        embedding_model=args.embedding_model or None,
        answer_model=args.answer_model or None,
        prompt_version=args.prompt_version,
        rerank=args.rerank,
        final_k=final_k,
    )

    n_first = None if args.first == 0 else args.first
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    default_dataset_path = Path(args.dataset).expanduser().resolve()

    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# RAG Evaluation Dashboard (HotpotQA)")

        models = _current_models_meta()
        gr.Markdown(
            f"""
**Defaults**
- Dataset: `{default_dataset_path}`
- Limit: first `{args.first}` examples (0 = all) | k=`{args.k}` | final_k=`{final_k}`
- Re-ranking: `{'ON' if args.rerank else 'OFF'}`
- Models: `embed={models['embedding_model']}` | `answer={models['answer_model']}` | `judge={models['judge_model']}`
- Prompt version: `{args.prompt_version}`
"""
        )

        with gr.Row():
            run_button = gr.Button("Run Evaluation (Retrieval + Answer)", variant="primary")
            retrieval_only_button = gr.Button("Run Retrieval Only", variant="secondary")
            load_button = gr.Button("Load Run", variant="secondary")
            refresh_runs_button = gr.Button("Refresh runs list", variant="secondary")

        run_choices = _list_run_dirs(runs_dir)
        default_run_value = run_choices[0] if run_choices else None

        with gr.Row():
            dataset_override = gr.File(label="Dataset override (.jsonl)", file_types=[".jsonl"])
            run_dropdown = gr.Dropdown(
                label="Run to load (from local runs folder)",
                choices=run_choices,
                value=default_run_value,
            )

        status = gr.Markdown("")

        with gr.Row():
            with gr.Column(scale=1):
                summary_html = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Run or load an evaluation to see summary.</div>"
                )
            with gr.Column(scale=1):
                with gr.Row():
                    retrieval_chart = gr.BarPlot(
                        x="Metric",
                        y="Value",
                        title="Overall Retrieval Metrics",
                        y_lim=[0, 1],
                        height=220,  # smaller
                    )
                    answer_chart = gr.BarPlot(
                        x="Metric",
                        y="Value",
                        title="Overall Answer Metrics",
                        y_lim=[1, 5],
                        height=220,  # smaller
                    )

        gr.Markdown("## Example Browser")

        entries_state = gr.State([])  # list[ExampleLog]
        index_state = gr.State(0)
        dataset_state = gr.State(str(default_dataset_path))

        with gr.Row():
            prev_btn = gr.Button("Prev", variant="secondary")
            next_btn = gr.Button("Next", variant="secondary")
            counter = gr.Markdown("0 / 0")

        example_view = gr.HTML("<div style='padding: 20px; text-align: center; color: #999;'>No example loaded.</div>")

        def refresh_runs():
            choices = _list_run_dirs(runs_dir)
            value = choices[0] if choices else None
            return gr.update(choices=choices, value=value)

        def on_run_full(dataset_file, progress=gr.Progress()):
            ds_override = _resolve_uploaded_file_to_path(dataset_file)
            ds_path = ds_override if ds_override is not None else default_dataset_path

            return run_evaluation(
                dataset_path=ds_path,
                n_first=n_first,
                k=args.k,
                runs_dir=runs_dir,
                run_name=args.run_name or None,
                retrieval_only=bool(args.retrieval_only),
                structured_context=args.structured_context,
                prompt_version=args.prompt_version,
                rerank=args.rerank,
                final_k=final_k,
                progress=progress,
            )

        def on_run_retrieval_only(dataset_file, progress=gr.Progress()):
            ds_override = _resolve_uploaded_file_to_path(dataset_file)
            ds_path = ds_override if ds_override is not None else default_dataset_path

            return run_evaluation(
                dataset_path=ds_path,
                n_first=n_first,
                k=args.k,
                runs_dir=runs_dir,
                run_name=args.run_name or None,
                retrieval_only=True,
                structured_context=args.structured_context,
                prompt_version=args.prompt_version,
                rerank=False,  # No re-ranking for retrieval-only
                final_k=final_k,
                progress=progress,
            )

        def on_load(run_dir_str, dataset_file):
            empty_df_retrieval = pd.DataFrame(
                [
                    {"Metric": "Recall@10", "Value": 0.0},
                    {"Metric": "MRR@10", "Value": 0.0},
                    {"Metric": "nDCG@10", "Value": 0.0},
                ]
            )
            empty_df_answer = pd.DataFrame(
                [
                    {"Metric": "Accuracy (avg)", "Value": 0.0},
                    {"Metric": "Completeness (avg)", "Value": 0.0},
                    {"Metric": "Relevance (avg)", "Value": 0.0},
                ]
            )

            if not run_dir_str:
                return (
                    "<div style='padding: 20px; text-align: center; color: #999;'>Run or load an evaluation to see summary.</div>",
                    empty_df_retrieval,
                    empty_df_answer,
                    [],
                    0,
                    str(default_dataset_path),
                    "<div style='padding: 20px; text-align: center; color: #999;'>No example loaded.</div>",
                    "0 / 0",
                    "**ERROR:** No runs found. Create a run first, or check --runs-dir.",
                )

            try:
                run_dir = Path(run_dir_str)
                ds_override = _resolve_uploaded_file_to_path(dataset_file)
                return load_run(run_dir=run_dir, dataset_override_path=ds_override)
            except Exception as e:
                return (
                    "<div style='padding:20px;color:#b91c1c;'>Failed to load run.</div>",
                    empty_df_retrieval,
                    empty_df_answer,
                    [],
                    0,
                    str(default_dataset_path),
                    "<div style='padding:20px;color:#999;'>No example loaded.</div>",
                    "0 / 0",
                    f"**ERROR:** `{type(e).__name__}`: {e}",
                )

        def on_prev(entries, idx, ds_path_str):
            ds_path = Path(ds_path_str)
            new_idx = prev_index(entries, idx)
            html_view, cnt = render_current_example(entries, ds_path, new_idx)
            return new_idx, html_view, cnt

        def on_next(entries, idx, ds_path_str):
            ds_path = Path(ds_path_str)
            new_idx = next_index(entries, idx)
            html_view, cnt = render_current_example(entries, ds_path, new_idx)
            return new_idx, html_view, cnt

        refresh_runs_button.click(fn=refresh_runs, inputs=[], outputs=[run_dropdown])

        run_button.click(
            fn=on_run_full,
            inputs=[dataset_override],
            outputs=[
                summary_html,
                retrieval_chart,
                answer_chart,
                entries_state,
                index_state,
                dataset_state,
                example_view,
                counter,
                status,
            ],
        )

        retrieval_only_button.click(
            fn=on_run_retrieval_only,
            inputs=[dataset_override],
            outputs=[
                summary_html,
                retrieval_chart,
                answer_chart,
                entries_state,
                index_state,
                dataset_state,
                example_view,
                counter,
                status,
            ],
        )

        load_button.click(
            fn=on_load,
            inputs=[run_dropdown, dataset_override],
            outputs=[
                summary_html,
                retrieval_chart,
                answer_chart,
                entries_state,
                index_state,
                dataset_state,
                example_view,
                counter,
                status,
            ],
        )

        prev_btn.click(
            fn=on_prev,
            inputs=[entries_state, index_state, dataset_state],
            outputs=[index_state, example_view, counter],
        )

        next_btn.click(
            fn=on_next,
            inputs=[entries_state, index_state, dataset_state],
            outputs=[index_state, example_view, counter],
        )

    # Launch with specified port or auto-assign
    launch_kwargs = {"inbrowser": True}
    if args.port is not None:
        launch_kwargs["server_port"] = args.port

    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()