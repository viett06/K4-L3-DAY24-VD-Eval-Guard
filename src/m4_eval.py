from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import math
import os, sys, json
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, resolve_chat_model


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _finite_unit(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"RAGAS {name} is not a finite score in [0, 1]: {value!r}")
    return number


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS with the LLM and embedding models this call actually configures."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("RAGAS inputs must have the same length")
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings

    llm = ChatOpenAI(model=resolve_chat_model("gpt-4o-mini"), temperature=0, max_tokens=400)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    llm_id = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    embedding_id = getattr(embeddings, "model_name", None)
    if not isinstance(llm_id, str) or not llm_id.strip():
        raise ValueError("Evaluator LLM did not report a model id")
    if not isinstance(embedding_id, str) or not embedding_id.strip():
        raise ValueError("Evaluator embeddings did not report a model id")
    evaluator_models = [llm_id.strip(), embedding_id.strip()]

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=True,
    )
    df = result.to_pandas()
    if len(df) != len(questions):
        raise ValueError(f"RAGAS returned {len(df)} rows for {len(questions)} questions")
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row["question"],
            answer=row["answer"],
            contexts=list(row["contexts"]),
            ground_truth=row["ground_truth"],
            faithfulness=_finite_unit(row["faithfulness"], "faithfulness"),
            answer_relevancy=_finite_unit(row["answer_relevancy"], "answer_relevancy"),
            context_precision=_finite_unit(row["context_precision"], "context_precision"),
            context_recall=_finite_unit(row["context_recall"], "context_recall"),
        ))
    count = len(per_question)
    return {
        "faithfulness": sum(r.faithfulness for r in per_question) / count,
        "answer_relevancy": sum(r.answer_relevancy for r in per_question) / count,
        "context_precision": sum(r.context_precision for r in per_question) / count,
        "context_recall": sum(r.context_recall for r in per_question) / count,
        "per_question": per_question,
        "metadata": {"evaluator_models": evaluator_models},
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating or using unsupported claims", "Tighten prompt, cite context spans, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking, add BM25 terms, or enrich chunks with contextual text"),
        "context_precision": ("Retrieved context contains too much irrelevant text", "Tune hybrid weights, add reranking, or add metadata filters"),
        "answer_relevancy": ("Answer does not directly address the question", "Improve prompt template and answer format constraints"),
    }
    scored = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg = sum(metrics.values()) / len(metrics)
        worst_metric = min(metrics, key=metrics.get)
        diagnosis, fix = diagnostic_tree[worst_metric]
        scored.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": float(metrics[worst_metric]),
            "avg_score": float(avg),
            "diagnosis": diagnosis,
            "suggested_fix": fix,
            "error_tree": f"{worst_metric} -> {diagnosis} -> {fix}",
        })
    return sorted(scored, key=lambda item: item["avg_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "reports/ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
