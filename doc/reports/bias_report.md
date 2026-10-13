# Đánh giá LLM Judge Phase B

Nhóm: VD

Người phụ trách: Nguyễn Quang Duy, MSSV 2A202602426

Nguồn dự kiến: `artifacts/judge_results.json`. Batch live chưa chạy vì API key
được cung cấp đã hết hạn, nên chưa có timestamp hoặc metadata model judge từ một
lượt chạy thành công. Generation source hiện có dùng `openai/gpt-4o-mini` theo
`artifacts/answers_50q.json`.

## Năm cặp A/B và hai lượt

Source code đã cố định đúng nguồn theo đề: A từ
`artifacts/answers_50q.json:answer`; B từ
`fixtures/human_labels_10q.json:model_answer`; reference từ
`fixtures/test_set_50q.json:ground_truth`. Các id là 1, 5, 12, 21 và 41.

`swap_and_average` gọi judge hai lần theo thứ tự A/B rồi B/A, quy đổi winner và
scores của lượt hai về A/B gốc. Kết quả chỉ được xem là decisive khi hai lượt
đồng thuận; trường hợp không đồng thuận trở thành `tie`. Unit tests đã xác nhận
thứ tự lời gọi, phép đổi winner, phép đổi scores và consensus.

Chưa có năm rows live nên position bias count/rate, verbosity win rate và số
decisive cases đều chưa đo. Không suy luận các giá trị này từ unit tests hoặc tự
điền số nhằm vượt checker.

## Calibration đúng model_answer

Runner gửi đúng `question`, `model_answer` và reference cho helper nhị phân;
`human_label` và `human_note` chỉ được đọc sau khi judge trả kết quả. Hàm
`cohen_kappa` kiểm tra hai danh sách có cùng độ dài, chỉ nhận integer 0/1, tính
observed agreement và chance agreement, đồng thời xử lý trường hợp đồng thuận
hằng một cách xác định.

Chưa có 10 binary judgments live nên Cohen's kappa, observed agreement, chance
agreement và các case bất đồng với người chấm đều chưa đo. Unit tests chỉ xác
nhận công thức với các vector nhãn kiểm soát, không thay thế calibration thật.

## Giới hạn và cách dùng judge

Mẫu chỉ gồm năm cặp pairwise và mười câu calibration, vì vậy ngay cả sau khi có
kết quả cũng cần đọc từng reasoning thay vì xem judge là đáp án tuyệt đối.
Position inconsistency chỉ là dấu hiệu tương quan với vị trí; verbosity win rate
không chứng minh độ dài gây ra winner. Khi có key hợp lệ, chạy
`python scripts/run_experiments.py --phase b --live`, kiểm tra provenance trong
JSON rồi thay phần “chưa đo” bằng dữ liệu tạo ra từ batch đó.

