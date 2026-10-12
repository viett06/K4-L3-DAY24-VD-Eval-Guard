from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re, math
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?。])\s+|\n{2,}', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    def lexical_vector(sentence: str) -> dict[str, int]:
        tokens = re.findall(r"\w+", sentence.lower(), flags=re.UNICODE)
        vector: dict[str, int] = {}
        for token in tokens:
            vector[token] = vector.get(token, 0) + 1
        return vector

    def lexical_cosine(a: dict[str, int], b: dict[str, int]) -> float:
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return dot / (norm_a * norm_b + 1e-9)

    try:
        if os.getenv("USE_HEAVY_MODELS") != "1":
            raise RuntimeError("Heavy semantic model disabled")
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(sentences)

        def similarity(i: int) -> float:
            return float(dot(embeddings[i - 1], embeddings[i]) / (norm(embeddings[i - 1]) * norm(embeddings[i]) + 1e-9))
    except Exception:
        vectors = [lexical_vector(s) for s in sentences]

        def similarity(i: int) -> float:
            return lexical_cosine(vectors[i - 1], vectors[i])

    chunks: list[Chunk] = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        if similarity(i) < threshold and current:
            chunks.append(Chunk(
                text=" ".join(current).strip(),
                metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
            ))
            current = []
        current.append(sentences[i])

    if current:
        chunks.append(Chunk(
            text=" ".join(current).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)},
        ))
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs and text.strip():
        paragraphs = [text.strip()]

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > parent_size and current:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
            ))
            current = ""
        if len(para) > parent_size:
            for start in range(0, len(para), parent_size):
                part = para[start:start + parent_size].strip()
                if part:
                    pid = f"parent_{len(parents)}"
                    parents.append(Chunk(
                        text=part,
                        metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
                    ))
        else:
            current += para + "\n\n"

    if current.strip():
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)},
        ))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        child_parts = [p.strip() for p in parent.text.split("\n\n") if p.strip()]
        buffer = ""
        for part in child_parts:
            if len(buffer) + len(part) + 2 > child_size and buffer:
                children.append(Chunk(
                    text=buffer.strip(),
                    metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                    parent_id=pid,
                ))
                buffer = ""
            if len(part) > child_size:
                for start in range(0, len(part), child_size):
                    segment = part[start:start + child_size].strip()
                    if segment:
                        children.append(Chunk(
                            text=segment,
                            metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                            parent_id=pid,
                        ))
            else:
                buffer += part + "\n\n"
        if buffer.strip():
            children.append(Chunk(
                text=buffer.strip(),
                metadata={**metadata, "chunk_type": "child", "child_index": len(children)},
                parent_id=pid,
            ))
    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if current_header or body:
            section = current_header.lstrip("#").strip() if current_header else "root"
            chunk_text = f"{current_header}\n\n{body}".strip() if current_header else body
            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={**metadata, "section": section, "strategy": "structure", "chunk_index": len(chunks)},
                ))
        current_lines = []

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+$", line):
            flush()
            current_header = line.strip()
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        chunks.append(Chunk(text=text.strip(), metadata={**metadata, "section": "root", "strategy": "structure", "chunk_index": 0}))
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
