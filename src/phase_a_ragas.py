from __future__ import annotations

"""Phase A: RAGAS Production Evaluation — 50q, 3 distributions, cluster analysis."""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, ANSWERS_PATH, ARTIFACTS_DIR
from src.experiment_support import write_json

Distribution = str  # "factual" | "multi_hop" | "adversarial"

DIAGNOSTIC_TREE = {
    "faithfulness":      ("LLM hallucinating", "Tighten system prompt, lower temperature"),
    "context_recall":    ("Missing relevant chunks", "Improve chunking or add BM25"),
    "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    "answer_relevancy":  ("Answer doesn't match question", "Improve prompt template"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return (self.faithfulness + self.answer_relevancy +
                self.context_precision + self.context_recall) / 4

    @property
    def worst_metric(self) -> str:
        scores = {
            "faithfulness":      self.faithfulness,
            "answer_relevancy":  self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall":    self.context_recall,
        }
        return min(scores, key=scores.get)


# ─── Đã implement sẵn ────────────────────────────────────────────────────────

@dataclass
class RagasEvaluation:
    """Task 2 output: measured rows plus model identifiers reported by m4."""
    results: list[RagasResult]
    evaluator_models: list[str]

def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    """Load 50q test set với 3 distributions."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    """Load pre-generated answers từ setup_answers.py."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"answers_50q.json không tìm thấy tại {path}\n"
            "→ Chạy trước: python scripts/setup_answers.py --live"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict]) -> dict:
    """Provided exporter: preserve the student-produced Task 3 rows verbatim."""
    per_dist: dict[str, dict] = {}
    for dist in ["factual", "multi_hop", "adversarial"]:
        subset = [r for r in results if r.distribution == dist]
        if subset:
            per_dist[dist] = {
                "count": len(subset),
                "faithfulness":      sum(r.faithfulness for r in subset) / len(subset),
                "answer_relevancy":  sum(r.answer_relevancy for r in subset) / len(subset),
                "context_precision": sum(r.context_precision for r in subset) / len(subset),
                "context_recall":    sum(r.context_recall for r in subset) / len(subset),
                "avg_score":         sum(r.avg_score for r in subset) / len(subset),
            }

    report = {
        "total_questions": len(results),
        "per_question": [asdict(r) for r in results],
        "per_distribution": per_dist,
        "failure_clusters": clusters,
        "bottom_10": bottom,
    }
    return report


def save_phase_a_report(results: list[RagasResult], clusters: dict, bottom: list[dict],
                        path: str = os.path.join(ARTIFACTS_DIR, "ragas_50q.json")) -> None:
    write_json(path, build_phase_a_report(results, clusters, bottom))



# ─── Tasks 1-4: Sinh viên implement ──────────────────────────────────────────

def _metric(record, name: str) -> float:
    value = record[name] if isinstance(record, dict) else getattr(record, name)
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} is not a finite score in [0, 1]")
    return number


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    """Task 1: Nhóm 50 câu hỏi theo 3 distributions."""
    groups = {"factual": [], "multi_hop": [], "adversarial": []}
    for item in test_set:
        groups[item["distribution"]].append(item)
    return groups


def run_ragas_50q(answers: list[dict]) -> RagasEvaluation:
    """Task 2: Chạy RAGAS 4 metrics trên toàn bộ 50 câu hỏi."""
    try:
        from src.m4_eval import evaluate_ragas
    except ImportError as exc:
        raise ValueError("Missing m4 evaluator; cannot declare its actual models") from exc

    questions = [a["question"] for a in answers]
    ans_texts = [a["answer"] for a in answers]
    contexts = [a["contexts"] for a in answers]
    ground_truths = [a["ground_truth"] for a in answers]
    raw = evaluate_ragas(questions, ans_texts, contexts, ground_truths)
    per_q = raw.get("per_question", [])
    if len(per_q) != len(answers):
        raise ValueError("m4 per_question length does not match answers")
    models = raw.get("metadata", {}).get("evaluator_models")
    if not isinstance(models, list) or not models:
        raise ValueError("m4 must report actual evaluator models")

    results = []
    for answer, metrics in zip(answers, per_q):
        results.append(RagasResult(
            question_id=answer["id"],
            distribution=answer["distribution"],
            question=answer["question"],
            answer=answer["answer"],
            contexts=answer["contexts"],
            ground_truth=answer["ground_truth"],
            faithfulness=_metric(metrics, "faithfulness"),
            answer_relevancy=_metric(metrics, "answer_relevancy"),
            context_precision=_metric(metrics, "context_precision"),
            context_recall=_metric(metrics, "context_recall"),
        ))
    return RagasEvaluation(results, list(models))


def bottom_10(results: list[RagasResult]) -> list[dict]:
    """Task 3: Lấy tối đa 10 câu có avg_score thấp nhất, xếp tăng dần."""
    ordered = sorted(results, key=lambda row: (row.avg_score, row.question_id))
    output = []
    for rank, row in enumerate(ordered[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[row.worst_metric]
        output.append({
            "rank": rank,
            "question_id": row.question_id,
            "distribution": row.distribution,
            "question": row.question,
            "avg_score": round(row.avg_score, 4),
            "worst_metric": row.worst_metric,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    """Task 4: Đếm worst_metric của mọi câu và chọn nhóm có mean thấp nhất."""
    matrix = {
        metric: {"factual": 0, "multi_hop": 0, "adversarial": 0}
        for metric in DIAGNOSTIC_TREE
    }
    for row in results:
        matrix[row.worst_metric][row.distribution] += 1

    means = {}
    for distribution in ("factual", "multi_hop", "adversarial"):
        subset = [row for row in results if row.distribution == distribution]
        if subset:
            means[distribution] = sum(row.avg_score for row in subset) / len(subset)
    if not means:
        raise ValueError("cluster_analysis needs at least one result")
    dominant_dist = min(means, key=means.get)
    dominant_metric = max(matrix, key=lambda metric: sum(matrix[metric].values()))
    diagnosis, suggested_fix = DIAGNOSTIC_TREE[dominant_metric]
    insight = (
        f"Distribution '{dominant_dist}' có điểm trung bình thấp nhất "
        f"({means[dominant_dist]:.4f}). Metric '{dominant_metric}' là worst_metric "
        f"xuất hiện nhiều nhất. Chẩn đoán: {diagnosis}. Gợi ý: {suggested_fix}."
    )
    report = {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_dist,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }
    if set(means) == {"factual", "multi_hop", "adversarial"}:
        report["distribution_avg_scores"] = means
    return report


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "a", *sys.argv[1:]]))
