# Phân tích lỗi Phase A

Nhóm: VD

Người phụ trách: lÊ Văn Việt, MSSV 2A202602504

Nguồn: chưa có `artifacts/ragas_50q.json`. Run timestamp: chưa có vì batch Phase A không chạy xong.
Generation mode/fallback và điều kiện chạy: 50/50 answers ở mode `llm`, model `openai/gpt-4o-mini` qua OpenRouter. Retrieval dùng `BAAI/bge-m3` và `BAAI/bge-reranker-v2-m3`, `USE_HEAVY_MODELS=1`, Qdrant `lab24_production`, 128 chunk đã enrich. Mỗi câu có 3 contexts. Không câu nào dùng ground truth làm answer. Hai PDF scan bị bỏ qua vì không có text layer: `BCTC.pdf` và `Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf`.
Models generation/evaluator: generation là `openai/gpt-4o-mini`. Evaluator dự kiến là `openai/gpt-4o-mini` và `BAAI/bge-m3`, nhưng lượt RAGAS không hoàn thành nên chưa có metadata evaluator trong report JSON.

## Điểm theo distribution

| Metric | factual (20) | multi_hop (20) | adversarial (10) |
|---|---|---|---|
| faithfulness | chưa đo | chưa đo | chưa đo |
| answer_relevancy | chưa đo | chưa đo | chưa đo |
| context_precision | chưa đo | chưa đo | chưa đo |
| context_recall | chưa đo | chưa đo | chưa đo |
| avg_score | chưa đo | chưa đo | chưa đo |

## Bottom 10 và nguyên nhân

Chưa có 10 rows từ Task 3. `bottom_10` đã được viết và qua unit test, nhưng exporter chưa chạy trên 50 kết quả RAGAS thật. Không điền diagnosis giả.

Đối chiếu answer với reference chưa làm ở mức điểm số. Answers và contexts đã có trong `artifacts/answers_50q.json`. Câu adversarial trong bộ test cố tình dùng bản policy cũ và phủ định. Việc đó mới là thiết kế dữ liệu, chưa phải kết luận từ điểm RAGAS.

## Matrix và nhóm yếu nhất

Matrix 4 metric nhân 3 distributions chưa đo. Mỗi câu sẽ được đếm theo `worst_metric` khi có đủ 50 `RagasResult`.

Nhóm yếu theo mean thấp nhất: chưa đo. Mean chưa có.
Worst metric xuất hiện nhiều nhất: chưa đo. Số câu chưa có.

Chưa chọn dominant distribution. Tổng 20/20/10 chỉ là kích thước nhóm, không dùng để kết luận nhóm yếu.

## Bẫy adversarial và giới hạn

Chưa so sánh mean adversarial với factual và multi_hop. Điểm thấp hơn, nếu sau này có, cũng không tự chứng minh pipeline nhận ra version conflict hay phủ định.

Giới hạn của lượt này: `python scripts/run_experiments.py --phase a --live` dừng quanh 65/200 lượt chấm. OpenRouter trả HTTP 402, tài khoản hết credit và chưa mua thêm credit. Vì vậy không có số faithfulness, answer relevancy, context precision hay context recall để ghi. Cần nạp credit rồi chạy lại đúng lệnh trên. Sau khi có `artifacts/ragas_50q.json`, bảng trên phải được thay bằng số trong file đó.
