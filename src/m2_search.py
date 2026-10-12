from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, math, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        if os.getenv("USE_UNDERTHESEA") != "1":
            raise RuntimeError("underthesea disabled")
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", segment_vietnamese(text).lower(), flags=re.UNICODE)


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [_tokenize(chunk["text"]) for chunk in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if not self.documents:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        if self.bm25 is not None:
            scores = list(self.bm25.get_scores(tokenized_query))
        else:
            q_terms = set(tokenized_query)
            scores = []
            for tokens in self.corpus_tokens:
                tf = sum(tokens.count(t) for t in q_terms)
                coverage = len(q_terms & set(tokens)) / max(len(q_terms), 1)
                scores.append(tf + coverage)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices
            if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None
        self._fallback_documents = []
        self._fallback_vectors = []

    def _get_encoder(self):
        if self._encoder is None:
            if os.getenv("USE_HEAVY_MODELS") != "1":
                raise RuntimeError("Heavy embedding model disabled")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    @staticmethod
    def _bow_vector(text: str) -> dict[str, int]:
        vector: dict[str, int] = {}
        for token in _tokenize(text):
            vector[token] = vector.get(token, 0) + 1
        return vector

    @staticmethod
    def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
        dot = sum(a[t] * b[t] for t in set(a) & set(b))
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return dot / (norm_a * norm_b + 1e-9)

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._fallback_documents = chunks
        self._fallback_vectors = [self._bow_vector(c["text"]) for c in chunks]
        if not chunks:
            return
        try:
            from qdrant_client.models import Distance, VectorParams, PointStruct
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            texts = [c["text"] for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=False)
            points = [
                PointStruct(
                    id=i,
                    vector=v.tolist(),
                    payload={**chunks[i].get("metadata", {}), "text": chunks[i]["text"]},
                )
                for i, v in enumerate(vectors)
            ]
            self.client.upsert(collection_name=collection, points=points)
        except Exception as exc:
            print(f"  ⚠️  Qdrant dense index failed, BM25/bag-of-words fallback will be used: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        try:
            query_vector = self._get_encoder().encode(query).tolist()
            # qdrant-client 1.9 exposes search(), not query_points().
            hits = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
            )
            return [
                SearchResult(
                    text=(pt.payload or {}).get("text", ""),
                    score=float(pt.score),
                    metadata={k: v for k, v in (pt.payload or {}).items() if k != "text"},
                    method="dense",
                )
                for pt in hits
            ]
        except Exception as exc:
            print(f"  ⚠️  Qdrant dense search failed: {exc}")
            q_vector = self._bow_vector(query)
            scored = [
                (self._cosine(q_vector, vector), doc)
                for vector, doc in zip(self._fallback_vectors, self._fallback_documents)
            ]
            scored.sort(key=lambda item: item[0], reverse=True)
            return [
                SearchResult(text=doc["text"], score=float(score), metadata=doc.get("metadata", {}), method="dense")
                for score, doc in scored[:top_k]
                if score > 0
            ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in merged[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
