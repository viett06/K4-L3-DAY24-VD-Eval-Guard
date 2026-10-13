# Blueprint CI/CD cho RAG và guardrails

Nhóm: VD

Người phụ trách: Nguyễn Quang Duy, MSSV 2A202602426

Đây là đề xuất CI/CD, không phải workflow đã được triển khai hoặc chạy. Source
Tasks 5–12 đã hoàn thiện và qua unit tests offline. Các batch live A/B/C chưa
hoàn thành vì API key hết hạn; do đó chưa có timestamp, artifact live hoặc số đo
để công bố. Generation source hiện có ghi model `openai/gpt-4o-mini`; evaluator,
judge và guard model chỉ được xác nhận sau một lượt runner thành công.

## Kiến trúc và failure action

Luồng đề xuất là Input → Presidio → NeMo input → RAG → NeMo output → Response.
Request thực dừng sớm khi Presidio phát hiện PII hoặc NeMo input rail chặn. PII
được redact trong log; raw input/output nhạy cảm không được ghi vào telemetry.
Nếu RAG hoặc provider lỗi, hệ thống trả thông báo lỗi an toàn thay vì tự tạo câu
trả lời không có context. Output rail thay câu trả lời bị chặn bằng response của
rail và giữ nguyên câu trả lời sạch.

Secrets được đưa qua secret manager của môi trường chạy, không nằm trong source,
artifact hoặc log. CI chỉ chạy unit tests không mạng; job live dùng environment
được bảo vệ và cần phê duyệt để tránh chi phí ngoài ý muốn.

## Phạm vi và số đo latency

Task 12 đo `input_layers_sequential`: Presidio và NeMo input luôn được gọi tuần
tự, không short-circuit. Engines được dựng trước khi đo, hai lượt warm-up thật bị
loại, và percentile dùng nội suy tuyến tính từ tổng của từng raw sample.

| Layer / scope | P50 ms | P95 ms | P99 ms | Tình trạng |
|---|---:|---:|---:|---|
| Presidio | chưa đo | chưa đo | chưa đo | Chờ batch live |
| NeMo input | chưa đo | chưa đo | chưa đo | Chờ batch live |
| Input guard sequential | chưa đo | chưa đo | chưa đo | Chờ batch live |
| RAG | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| NeMo output | chưa đo | chưa đo | chưa đo | Ngoài phạm vi bắt buộc |
| Full guard / short-circuit request | chưa đo | chưa đo | chưa đo | Không suy từ input guard |

Target input-layer P95 là 500 ms nhưng `latency_budget_ok` chưa xác định khi chưa
có mẫu. Sau khi có key, cần lấy ít nhất 20 mẫu timed sau hai warm-up. Muốn đo RAG,
output và full request phải tạo benchmark riêng, ghi timestamp ở ranh giới từng
stage và phân tầng blocked/allowed để không trộn hai đường chạy.

## CI gates đề xuất

- Job offline: cài test lockfile, chạy `python -m pytest tests/ -m "not integration" -q`.
- Job eval được phê duyệt: chạy Phase A và yêu cầu faithfulness ≥0.75,
  `avg_score` ≥0.65 trên đủ 50 câu.
- Job guard được phê duyệt: yêu cầu attacks ≥18/20, clean 8/8 allowed và cả bốn
  output cases đúng. Mốc bắt buộc của lab vẫn là attacks ≥15/20.
- Chỉ áp dụng gate full-guard P95 <500 ms sau khi thực sự đo full request; không
  dùng `input_guard_ms` để đại diện cho chỉ số này.
- Mỗi gate phải validate schema/provenance trước khi đọc metric và trả exit code
  khác 0 khi thiếu artifact, stale hash hoặc không đạt ngưỡng.

Các mục trên là thiết kế đề xuất; repository chưa có workflow live đã chạy và
các biến `MIN_*` tự thân không tạo quality gate.

## Monitoring và cải tiến

Production monitoring nên theo dõi tỷ lệ block theo category, false-positive
rate trên clean canary, P50/P95/P99 từng layer, lỗi provider, RAGAS theo
distribution và drift của query. Cảnh báo khi clean false-positive vượt 1%,
attack pass rate giảm dưới gate hoặc input P95 vượt 500 ms trong hai cửa sổ liên
tiếp. Chỉ lưu nội dung đã redact; raw PII cần bị loại hoặc mã hóa với thời hạn lưu
ngắn và quyền truy cập có audit.

Kết quả RAGAS mean/worst distribution, Cohen's kappa, attacks passed và clean
false-positive rate hiện chưa đo. Sau khi có key hợp lệ, nhóm cần chạy ba phase,
đọc các case yếu/bất đồng/chặn nhầm, điều chỉnh prompt hoặc policy trên tập phát
triển, rồi chạy lại toàn bộ tập cố định đúng một lần để tránh chọn lọc kết quả.
Bộ 50/20/8/4 nhỏ nên cần bổ sung regression từ incident thực tế, versioned policy
cases, multilingual attacks và canary theo từng phiên bản model.

