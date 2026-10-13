from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str, reference: str = "") -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    prompt = f"""Bạn là chuyên gia đánh giá chất lượng câu trả lời RAG.

Câu hỏi: {question}
Chính sách/đáp án tham chiếu: {reference}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên độ chính xác so với tham chiếu, độ đầy đủ và tính súc tích.
Không ưu tiên câu trả lời vì vị trí hoặc độ dài. Trả lời duy nhất một JSON:
{{"winner":"A|B|tie","reasoning":"giải thích ngắn gọn","scores":{{"A":0.0,"B":0.0}}}}
"""
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON hợp lệ."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    if result.get("winner") not in {"A", "B", "tie"}:
        raise ValueError("Judge winner must be A, B, or tie.")
    if not isinstance(result.get("reasoning"), str) or not result["reasoning"].strip():
        raise ValueError("Judge reasoning must be a non-empty string.")
    scores = result.get("scores")
    if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
        raise ValueError("Judge scores must contain exactly A and B.")
    if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in scores.values()):
        raise ValueError("Judge scores must be numeric values in [0, 1].")
    return result


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str, reference: str = "") -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b, reference=reference)
    pass2_raw = pairwise_judge(question, answer_b, answer_a, reference=reference)
    winner_pass2 = {"A": "B", "B": "A", "tie": "tie"}[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2

    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2,
        final_winner=pass1["winner"] if consistent else "tie",
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=consistent,
        scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("Judge and human labels must be non-empty and have equal length.")
    if any(type(label) is not int or label not in (0, 1)
           for label in judge_labels + human_labels):
        raise ValueError("Cohen kappa only accepts integer labels 0 and 1.")
    n = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    chance = sum(judge_labels.count(label) * human_labels.count(label) / n**2
                 for label in (0, 1))
    if chance == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - chance) / (1 - chance)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    position_count = sum(not row.position_consistent for row in judge_results)
    a_longer = sum(row.final_winner == "A" and len(row.answer_a) > len(row.answer_b)
                   for row in judge_results)
    b_longer = sum(row.final_winner == "B" and len(row.answer_b) > len(row.answer_a)
                   for row in judge_results)
    decisive = sum(row.final_winner != "tie" for row in judge_results)
    position_rate = position_count / total if total else 0.0
    verbosity_rate = (a_longer + b_longer) / decisive if decisive else 0.0
    if not total:
        interpretation = "Chưa có cặp nào để đánh giá position hoặc verbosity bias."
    elif position_rate > 0.3:
        interpretation = "Position inconsistency cao; cần giữ swap-and-average và kiểm tra thủ công."
    else:
        interpretation = "Position inconsistency thấp; vẫn cần đọc nội dung trước khi kết luận judge đáng tin cậy."
    return {
        "total_judged": total,
        "position_bias_rate": round(position_rate, 3),
        "position_bias_count": position_count,
        "verbosity_bias": round(verbosity_rate, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_longer,
            "b_wins_b_longer": b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scripts.run_experiments import main
    raise SystemExit(main(["--phase", "b", *sys.argv[1:]]))

