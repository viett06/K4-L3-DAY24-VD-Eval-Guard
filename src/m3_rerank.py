from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, math, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                if os.getenv("USE_HEAVY_MODELS") != "1":
                    raise RuntimeError("Heavy rerank model disabled")
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = _LexicalRerankModel()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        scored = sorted(zip(scores, documents), key=lambda x: float(x[0]), reverse=True)
        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]


class _LexicalRerankModel:
    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    @classmethod
    def _score(cls, query: str, text: str) -> float:
        q_tokens = cls._tokens(query)
        d_tokens = cls._tokens(text)
        if not q_tokens or not d_tokens:
            return 0.0
        q_set = set(q_tokens)
        d_set = set(d_tokens)
        overlap = len(q_set & d_set) / len(q_set)
        tf = sum(d_tokens.count(t) for t in q_set)
        length_norm = math.sqrt(len(d_tokens))
        return overlap + tf / (length_norm + 1e-9)

    def predict(self, pairs):
        return [self._score(query, text) for query, text in pairs]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
            if self._model is None:
                self._model = Ranker()
            passages = [{"text": d["text"], "metadata": d.get("metadata", {}), "score": d.get("score", 0.0)} for d in documents]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=r["text"],
                    original_score=float(r.get("score", 0.0)),
                    rerank_score=float(r.get("score", 0.0)),
                    metadata=r.get("metadata", {}),
                    rank=i,
                )
                for i, r in enumerate(results[:top_k])
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
