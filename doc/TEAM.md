# Nhóm thực hiện K4 Track 3A

Tên nhóm: VD

Repo chung: repo local `K4-L3-DAY24-VD-Eval-Guard`. Link nộp chưa có.

Trưởng nhóm/người nộp: Lê Văn Việt MSSV 2A202602504.

Ngày lab/deadline áp dụng: 2026-10-12, Asia/Ho_Chi_Minh (UTC+07:00). Deadline chính thức theo thông báo lớp.

Commit dùng để chấm: chưa commit bản nộp.

Mỗi nhóm nộp một repo. Hiện mới có một mục INDIVIDUAL cho phần đã làm. Teammate thêm hàng của mình khi làm Phase B và Phase C.

## Thành viên và phân công

| STT | Họ và tên   | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|-------------|---|---|---|
| 1 | Lê Văn Việt | 2A202602504 | Chuẩn bị pipeline, sinh 50 answers, Phase A Tasks 1–4 | `src/m1_chunking.py` đến `src/m5_enrichment.py`, `src/phase_a_ragas.py`, `artifacts/answers_50q.json`, `tests/test_phase_a.py` |

Người tích hợp/chạy batch/nộp link: Le Van Viet đã chạy setup và batch Phase A trên máy này. Batch Phase A chưa tạo `artifacts/ragas_50q.json` vì OpenRouter hết credit. Link nộp chưa gửi.

## Môi trường và chuẩn bị trước lớp

OS/Python/compiler: macOS, Python 3.11.15 trong `venv`.
Virtualenv/dependencies/imports: `python -m pip install -r requirements/requirements.txt -c requirements/constraints-core.txt`. `pip check` không báo gói hỏng. Import được ragas, nemoguardrails, presidio_analyzer, spacy, qdrant_client, openai. `pyarrow` được hạ về 18.1.0 vì bản 26 không chạy với NumPy 1.26.4. Model spaCy `en_core_web_lg` 3.8.0 đã tải.
Models/Qdrant/50 answers đã chuẩn bị: Qdrant `qdrant/qdrant:v1.9.7` trên cổng 6333. Embedding `BAAI/bge-m3`, reranker `BAAI/bge-reranker-v2-m3`, chạy với `USE_HEAVY_MODELS=1`. Collection `lab24_production` có 128 chunk. 50 answers ở `artifacts/answers_50q.json`, mode `llm`, model `openai/gpt-4o-mini`. Key là key OpenRouter nên client dùng `https://openrouter.ai/api/v1`. `HF_TOKEN` trống.

## INDIVIDUAL

### LeVanViet-2A202602504

- Phần được giao và phần thực tế đã làm: nửa đầu bài, gồm chuẩn bị pipeline và Phase A. Chép m1–m5 từ repo Day 18. Sửa dense search sang `search()` của qdrant-client 1.9. Thêm `metadata.evaluator_models` cho RAGAS, dùng LLM `openai/gpt-4o-mini` và embedding `BAAI/bge-m3`. Viết `group_by_distribution`, `run_ragas_50q`, `bottom_10`, `cluster_analysis`.
- Bằng chứng: `python -m pytest tests/test_phase_a.py` đạt 6 tests. `artifacts/answers_50q.json` có 50 câu, 20 factual, 20 multi_hop, 10 adversarial, không câu nào copy ground truth, mỗi câu 3 contexts. Generation mode là llm cho cả 50 câu.
- Cách kiểm tra: setup live đã enrich 128 chunk, index Qdrant, rồi sinh lại 50 câu từ index đó với `max_tokens=400`. Lệnh RAGAS `python scripts/run_experiments.py --phase a --live` chạy tới khoảng 65/200 lượt chấm rồi dừng.
- Điều đã học: retrieval hybrid và RAGAS là hai bước tách nhau. Answers có thể đúng mode llm trong khi bộ điểm 4 metric vẫn chưa có nếu lượt chấm không chạy hết.
- Giới hạn/cách xử lý: hai PDF scan, `BCTC.pdf` và nghị định 13, không có text layer nên bị bỏ qua. OpenRouter trả 402 khi RAGAS đang chạy, nên không có `artifacts/ragas_50q.json`. Không điền số metric khi chưa có file đó.
- Nguồn/AI hỗ trợ nếu có: pipeline lấy từ bài Day 18 của Le Van Viet. Grok hỗ trợ lắp môi trường, sửa tương thích Qdrant/OpenRouter và viết Tasks 1–4. Số đo generation đối chiếu với `artifacts/answers_50q.json`. Số RAGAS chưa có để kiểm chứng.

## Cập nhật sau khi nộp

Chưa nộp. Không có cập nhật sau deadline.
