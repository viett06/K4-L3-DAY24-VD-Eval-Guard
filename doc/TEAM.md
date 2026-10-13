# Nhóm thực hiện K4 Track 3A

Tên nhóm: VD

Repo chung: repo local `K4-L3-DAY24-VD-Eval-Guard`. Link nộp chưa có.

Trưởng nhóm/người nộp: Lê Văn Việt MSSV 2A202602504.

Ngày lab/deadline áp dụng: 2026-10-12, Asia/Ho_Chi_Minh (UTC+07:00). Deadline chính thức theo thông báo lớp.

Commit dùng để chấm: chưa commit bản nộp.

Mỗi nhóm nộp một repo. Nhóm gồm hai thành viên; Phase B và Phase C do Nguyễn Quang Duy tiếp tục triển khai sau phần chuẩn bị và Phase A của Lê Văn Việt.

## Thành viên và phân công

| STT | Họ và tên   | MSSV | Vai trò / công việc thật | Bằng chứng |
|---|-------------|---|---|---|
| 1 | Lê Văn Việt | 2A202602504 | Chuẩn bị pipeline, sinh 50 answers, Phase A Tasks 1–4 | `src/m1_chunking.py` đến `src/m5_enrichment.py`, `src/phase_a_ragas.py`, `artifacts/answers_50q.json`, `tests/test_phase_a.py` |
| 2 | Nguyễn Quang Duy | 2A202602426 | Phase B Tasks 5–8, Phase C Tasks 9a–12, kiểm thử và tích hợp báo cáo | `src/phase_b_judge.py`, `src/phase_c_guard.py`, kết quả unit tests; artifacts và báo cáo live sẽ bổ sung sau khi API key hợp lệ |

Người tích hợp/chạy batch/nộp link: Lê Văn Việt đã chạy setup và thử batch Phase A; Nguyễn Quang Duy hoàn thiện source Phase B/C và kiểm thử offline. Phase A dừng vì OpenRouter hết credit; key cung cấp cho lượt tiếp theo đã hết hạn nên B/C chưa chạy live. Link nộp chưa gửi.

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

### NguyenQuangDuy-2A202602426

- Phần được giao và phần thực tế đã làm: tiếp quản Phase B và Phase C; triển khai pairwise judge, swap-and-average, Cohen's kappa, phân tích position/verbosity bias, quét PII, input/output rails, adversarial suite và phép đo latency theo phạm vi `input_layers_sequential`.
- Bằng chứng: `src/phase_b_judge.py`, `src/phase_c_guard.py`; toàn bộ 86 unit tests offline đạt trên Python 3.11.9. Các hàm dùng dependency injection nên tests không gọi mạng, model hoặc API.
- Cách kiểm tra: `.venv\\Scripts\\python.exe -m pytest tests/ -m "not integration" -q --tb=short`. Đã xác nhận Presidio khởi tạo được với `en_core_web_lg` 3.8.0 và NeMo 0.10.0 đọc được cấu hình khi có biến môi trường API.
- Điều đã học: pairwise winner và nhãn calibration 0/1 là hai phép đánh giá khác nhau; swap pass thứ hai phải đổi winner và scores về không gian A/B ban đầu. Latency bắt buộc đo tuần tự cả Presidio và NeMo input sau hai lượt warm-up, không đại diện cho full request.
- Giới hạn/cách xử lý: API key được cung cấp trả lỗi `expired_secret_key`, nên chưa tạo artifact live hoặc điền số liệu báo cáo giả. Cần thay bằng key hợp lệ, sau đó chạy lại A/B/C và checker trước khi nộp.
- Nguồn/AI hỗ trợ nếu có: Codex hỗ trợ đọc đặc tả, triển khai Tasks 5–12, dựng môi trường Windows và chạy unit tests. Mọi artifact live sẽ được đối chiếu trực tiếp với JSON do runner tạo.

## Cập nhật sau khi nộp

Chưa nộp. Không có cập nhật sau deadline.

