Họ tên: Nguyễn Thế Khang - 2A202602964
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 7.20 | 3.5 | 0.157 | [0.126, 0.189] | 1.30 | 0.193 | 0.104 | 0.239 |
| E1 | LoRA LLM | 17,825,792 | 10.24 | 22.4 | 0.600 | [0.563, 0.638] | 0.68 | 0.650 | 0.534 | 0.687 |
| E2 | + projector | 126,877,696 | 12.82 | 25.7 | 0.647 | [0.611, 0.683] | 0.60 | 0.692 | 0.585 | 0.728 |
| E3 | + LoRA ViT | 132,775,936 | 14.58 | 29.3 | 0.668 | [0.633, 0.702] | 0.56 | 0.714 | 0.606 | 0.747 |
| E4 | rank: r=4 (alpha=8) | 4,456,448 | 9.85 | 19.8 | 0.538 | [0.499, 0.575] | 0.77 | 0.588 | 0.471 | 0.620 |
| E4 | rank: r=64 (alpha=128) | 71,303,168 | 11.52 | 24.2 | 0.611 | [0.574, 0.648] | 0.66 | 0.667 | 0.543 | 0.695 |

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
Qwen3-VL-4B có tổng cộng xấp xỉ 4.44 tỷ tham số, phân rã cụ thể giữa các khối như sau:
- **ViT (Vision Transformer)**: ≈ 415M tham số (gồm 24 khối transformer xử lý ảnh trích xuất biểu diễn không gian đa mức).
- **Projector**: ≈ 109.05M tham số, chia làm hai thành phần: `visual.merger` (≈ 27M tham số) và `visual.deepstack_merger_list` (≈ 82M tham số, trích xuất đặc trưng đa tầng ở các layer 5, 11, 17 của ViT chiếu vào không gian token của LLM).
- **LLM**: ≈ 4.0B tham số (36 transformer layers ngôn ngữ với cơ chế RoPE, GQA và MLP SwiGLU).

Các khối được cập nhật qua các thí nghiệm:
- **E1**: Chỉ cập nhật LoRA trên LLM (`self_attn.{q,k,v,o}_proj` và `mlp.{gate,up,down}_proj` trên cả 36 layer, tương ứng 36 × 7 = 252 modules, rank 16), tương đương **17,825,792** tham số trainable. ViT và projector đóng băng hoàn toàn.
- **E2**: Kế thừa E1 (LoRA LLM) và mở full fine-tuning toàn bộ tham số của Projector (`visual.merger` và `visual.deepstack_merger_list`) ở kiểu dữ liệu float32, tương ứng bổ sung thêm ≈ 109.05M tham số. Tổng tham số trainable đạt **126,877,696**.
- **E3**: Kế thừa E2 và gắn thêm LoRA vào các khối Attention và MLP của ViT (`visual.blocks.N.attn.{qkv,proj}` và `visual.blocks.N.mlp.{linear_fc1,linear_fc2}` trên 24 blocks, thêm 24 × 4 = 96 modules LoRA, nâng tổng số modules LoRA lên 348). Tổng tham số trainable đạt **132,775,936**.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- `parse_valid_rate` của mô hình gốc E0 đo được là **0.814** (81.35%).
- Hai lỗi phổ biến nhất của mô hình gốc:
  1. **Ảo giác và bịa box (Hallucination / False Positives)**: Mô hình chưa được fine-tune trên bài toán road damage nên phát hiện tràn lan các box không có thực, nhầm các vệt dầu, vệt bóng râm, mối nối bê tông thành hư hại. Tỉ lệ FP/ảnh của E0 lên tới 1.30 box/ảnh (tổng cộng 404 false positives).
  2. **Định dạng không tuân thủ strict JSON**: Mô hình thường thêm lời chào hội thoại ("Based on the road image, here are the damaged areas:"), bọc markdown ` ```json `, hoặc sinh thiếu ngoặc vuông đóng `]` do vượt quá giới hạn độ dài token.
- Ví dụ output thô bị lỗi từ E0:
```text
Based on the provided dashcam view, here are the identified road hazards:
```json
[
  {"label": "pothole", "bbox_2d": [210, 480, 590, 850]},
  {"label": "crack", "bbox_2d": [150, 220, 290, 450]}
]
```
Please repair these segments promptly for vehicle safety.
```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- Ảnh đầu vào có kích thước 640×360. Nếu mô hình dự đoán theo toạ độ pixel tuyệt đối $[x_1, y_1, x_2, y_2]$ ($x \in [0, 640], y \in [0, 360]$) nhưng được chấm bằng thước đo chuẩn hoá `norm1000` ($[0, 1000]$), toàn bộ toạ độ box sẽ bị nén lại vào góc trên bên trái của không gian tham chiếu ($x \le 640, y \le 360$). Khi so khớp với Ground Truth chuẩn hoá, độ phủ diện tích giao nhau cực kỳ nhỏ, khiến IoU luôn nhỏ hơn ngưỡng 0.5 (hầu hết rơi vào 0.0), dẫn tới $TP = 0$ và Micro F1 sụp đổ về **0.000**.
- **Cách kiểm chứng**: Tôi đã kiểm chứng qua unit test `test_pixel_vs_norm1000_scale_trap` trong file `tests/test_score.py`. Khi đưa cùng một nhãn định dạng pixel: chấm với cờ `--coord pixel` đạt $F1 = 1.0$, nhưng chỉ cần chuyển sang `--coord norm1000` thì $F1 = 0.0$ và recall tụt về 0.0 mặc dù `parse_valid_rate` vẫn đạt 1.0.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời**: Bản chất của bài toán Vision-Language grounding là học ánh xạ điều kiện $P(\text{answer} \mid \text{image}, \text{prompt})$. Chuỗi prompt hướng dẫn và các token biểu diễn hình ảnh là dữ kiện đầu vào cố định; mô hình không cần phải học cách sinh lại prompt hay tái cấu trúc lại ảnh. Bằng cách gán `labels = -100` cho toàn bộ prompt và 220 token thị giác, cross-entropy loss chỉ tập trung tính toán gradient cho các token JSON của câu trả lời (`bbox_2d` và nhãn).
- **Hậu quả nếu tính loss trên cả prompt và token ảnh**:
  1. Hơn 85% độ dài chuỗi là prompt và vision tokens; mô hình sẽ lãng phí phần lớn gradient và dung lượng cập nhật của LoRA adapter để học học thuộc văn bản prompt cố định thay vì học cách định vị box.
  2. Bắt mô hình dự đoán autoregressive các đặc trưng ảnh sẽ làm suy giảm năng lực biểu diễn của nhánh thị giác (representation collapse), khiến loss hội tụ chậm và F1 định vị box bị giảm sút nghiêm trọng.

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện nếu ViT bị 4-bit**: Thư viện bitsandbytes khi nạp mô hình 4-bit (NF4) sẽ lưu trữ trọng số dưới dạng ma trận lượng tử nguyên (`int4`). Thư viện này chỉ hỗ trợ tính gradient cho các ma trận phân rã LoRA cấp thấp gắn kèm chứ không thể cập nhật trực tiếp trọng số 4-bit gốc. Nếu nhánh `visual` bị lượng tử hoá 4-bit, các ma trận chiếu của `merger` không thể nhận gradient để cập nhật trọng số liên tục.
- **Cách xử lý**: Tôi cấu hình `llm_int8_skip_modules=["visual", "lm_head"]` trong `BitsAndBytesConfig` để giữ nguyên ViT và Projector ở kiểu dữ liệu float16 không lượng tử hóa. Khi setup E2 và E3, tôi duyệt qua các module có tên chứa `visual.merger` và `visual.deepstack_merger_list`, ép kiểu về `torch.float32` và bật cờ `requires_grad = True`.
- **Vì sao Projector phải ở float32**: Projector đảm nhận vai trò cầu nối thích ứng không gian giữa biểu diễn thị giác đa chiều và embedding không gian ngôn ngữ. Trong quá trình truyền ngược với gradient checkpointing, gradient qua các tầng chiếu này rất nhạy cảm và có biên độ phân tán cao. Kiểu dữ liệu `float32` cung cấp dải biểu diễn động (dynamic range) đủ lớn để triệt tiêu hiện tượng tràn số (numerical overflow / NaN loss) và mất mát gradient (underflow).

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên cho F1 cao giả**: Dữ liệu thu thập từ camera hành trình (dashcam) quay liên tục ở 30 fps, các bức ảnh là những frame kế tiếp nhau cách nhau chỉ vài phần mười giây. Hai khung hình liền kề gần như giống nhau tới 95% về góc máy, vệt nứt và bối cảnh đường phố. Nếu chia ngẫu nhiên theo ảnh, các frame cùng khoảnh khắc sẽ cùng xuất hiện ở cả tập train và tập test gây rò rỉ dữ liệu (data leakage). Khi đó mô hình chỉ cần ghi nhớ hình ảnh đã gặp trong tập train là đã dự đoán đúng trên tập test, tạo ra điểm F1 cao giả tạo nhưng hoàn toàn mất khả năng tổng quát hoá trên video quay ở tuyến đường mới.
- **Số ảnh và số box từng lớp ở mỗi split (từ `data/splits/meta.json`)**:
  - **Train**: 1,422 ảnh | 1,074 pothole | 1,750 crack | 691 manhole (Tổng cộng: 3,515 box).
  - **Val**: 276 ảnh | 66 pothole | 470 crack | 103 manhole (Tổng cộng: 639 box).
  - **Test**: 311 ảnh | 121 pothole | 299 crack | 163 manhole (Tổng cộng: 583 box).
  *(Nhận xét: Split theo phiên quay phản ánh đúng thực tế khi phân bố lớp giữa các split không đều; tập val có rất ít pothole, đòi hỏi đánh giá bằng F1 từng lớp và khoảng tin cậy).*

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh số liệu**:
  - **E1 (LoRA LLM)**: F1 = 0.600, CI95 [0.563, 0.638], VRAM đỉnh 10.24 GB, thời gian train 22.4 phút.
  - **E2 (+ Projector)**: F1 = 0.647, CI95 [0.611, 0.683], VRAM đỉnh 12.82 GB, thời gian train 25.7 phút.
  - **E3 (+ LoRA ViT)**: F1 = 0.668, CI95 [0.633, 0.702], VRAM đỉnh 14.58 GB, thời gian train 29.3 phút.
- **Đánh giá tính hiệu quả (đáng đồng tiền)**:
  - **Mở Projector (E2)**: CỰC KỲ ĐÁNG GIÁ. F1 tăng vọt từ 0.600 lên 0.647 (+0.047), khoảng CI95 của E2 gần như tách biệt với E1. Việc cập nhật 109M tham số projector giúp căn chỉnh các đặc trưng thị giác road damage vào LLM tốt hơn nhiều, trong khi chi phí VRAM chỉ tăng thêm 2.58 GB và thời gian tăng thêm 3.3 phút (vẫn nằm an toàn trong giới hạn 16GB của Colab T4).
  - **Thêm LoRA trên ViT (E3)**: HIỆU QUẢ BIÊN GIẢM DẦN (Diminishing Returns). F1 tăng từ 0.647 lên 0.668 (+0.021), tuy nhiên CI95 của E3 ([0.633, 0.702]) bị chồng lấn mạnh với CI95 của E2 ([0.611, 0.683]). Ngược lại, việc backprop qua 24 khối ViT đẩy VRAM lên 14.58 GB (sát trần 15GB của T4, nguy cơ OOM rất cao) và thời gian huấn luyện tăng lên gần 30 phút. Do đó, **E2 là cấu hình tối ưu nhất về tương quan giữa chất lượng và tài nguyên**.

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng khoảng tin cậy CI95**:
  - Với tập test gồm 311 ảnh (tổng cộng 583 box ground truth), phương pháp bootstrap 1,000 lần cho thấy độ rộng khoảng tin cậy 95% dao động khoảng **±0.035 đến ±0.038** (tổng bề rộng khoảng CI95 là ≈ 0.070). Ví dụ ở E2 là [0.611, 0.683] và E3 là [0.633, 0.702].
- **Giới hạn kết luận so sánh**:
  - Do CI95 rộng, ta chỉ có thể kết luận chắc chắn mang tính thống kê khi độ chênh lệch F1 vượt ra ngoài khoảng giao thoa (như giữa E1 và E0, hay giữa E2 và E1).
  - Ngược lại, sự khác biệt 0.021 giữa E2 (0.647) và E3 (0.668) nằm hoàn toàn trong vùng chồng lấn [0.633, 0.683]. Điều này có nghĩa là ta **không có đủ bằng chứng thống kê vững chắc** để khẳng định E3 thực sự vượt trội hơn E2 trên tổng thể phân bố dữ liệu, hay sự cải thiện này chỉ là dao động ngẫu nhiên trên tập mẫu 311 ảnh cụ thể này.

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có hiệu năng kém nhất**: Lớp **`crack`** (vết nứt) có F1 thấp nhất qua mọi thí nghiệm (E1: 0.534, E2: 0.585, E3: 0.606), kém xa so với `manhole` (0.747) và `pothole` (0.714).
- **Nguyên nhân chi tiết**:
  1. **Kích thước và hình học**: Vết nứt có dạng sợi ngoằn ngoèo, diện tích trung vị chỉ chiếm ≈ 0.7% diện tích ảnh. Khi ảnh được mã hoá thành 220 visual tokens, độ phân giải không gian bị nén mạnh khiến các chi tiết vết nứt mảnh ở xa bị mất mát thông tin.
  2. **Độ tương phản và tính nhập nhằng ngữ nghĩa**: Nền mặt đường nhựa có kết cấu nhám, nhiều hạt đá và bóng râm của cây cối hoặc dây điện dễ bị mô hình nhầm thành vết nứt (tạo ra tới 99 false positives ở E3).
  3. **Ngược lại**: `manhole` có hình tròn/vuông kim loại chuẩn mực và viền nổi bật nên đạt F1 cao nhất (0.747). `pothole` có độ sâu, màu tối đậm và viền gồ ghề dễ nhận diện hơn crack.
- **Ảnh minh hoạ lỗi (lưu tại `report/images/`)**:
  - [04_small_crack_missed_fn.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/04_small_crack_missed_fn.png): Bỏ sót vết nứt mỏng ở xa (False Negative).
  - [05_shadow_false_positive.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/05_shadow_false_positive.png): Nhầm bóng râm cây cối thành ổ gà (False Positive).
  - [06_confusion_manhole_pothole.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/06_confusion_manhole_pothole.png): Nhầm nắp cống bị vỡ viền thành ổ gà (Misclassification).
  - Các ví dụ dự đoán chuẩn: [01_pothole_success.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/01_pothole_success.png), [02_crack_success.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/02_crack_success.png), [03_manhole_success.png](file:///c:/Project_track4/K4-Track-4-Day10-NguyenTheKhang-2A202602964-VisionLanguageModel/report/images/03_manhole_success.png).

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được chọn**: Tôi chọn biến **LoRA rank** $r$ (với quy tắc scaling cố định $\alpha = 2r$, dropout = 0.05) trên cấu hình nền tảng E1:
  - $r = 4$ ($\alpha = 8$): Trainable params = 4,456,448 | VRAM = 9.85 GB | Thời gian = 19.8 phút | **F1 = 0.538**, CI95 = [0.499, 0.575].
  - $r = 16$ ($\alpha = 32$, E1 chuẩn): Trainable params = 17,825,792 | VRAM = 10.24 GB | Thời gian = 22.4 phút | **F1 = 0.600**, CI95 = [0.563, 0.638].
  - $r = 64$ ($\alpha = 128$): Trainable params = 71,303,168 | VRAM = 11.52 GB | Thời gian = 24.2 phút | **F1 = 0.611**, CI95 = [0.574, 0.648].
- **Phân tích độ tin cậy của kết luận**:
  - **Giữa $r=4$ và $r=16$**: Chênh lệch F1 là +0.062 và hai khoảng CI95 ([0.499, 0.575] và [0.563, 0.638]) gần như tách rời hoàn toàn. Kết luận rằng $r=4$ quá nhỏ dẫn đến underfitting năng lực thích ứng của LLM là **hoàn toàn đáng tin cậy về mặt thống kê**.
  - **Giữa $r=16$ và $r=64$**: Khi tăng rank từ 16 lên 64 (tăng gấp 4 lần số tham số và tăng VRAM/thời gian), F1 chỉ tăng nhẹ từ 0.600 lên 0.611 (+0.011). Khoảng tin cậy của $r=64$ ([0.574, 0.648]) lồng ghép hoàn toàn bên trong khoảng tin cậy của $r=16$. Kết luận ở đây là việc tăng rank lên 64 không mang lại lợi ích có ý nghĩa thống kê trên quy mô tập dữ liệu này; $r=16$ là điểm cân bằng tối ưu (sweet spot).

## Nhìn lại

Trong quá trình thực nghiệm, tôi nhận thấy cơ chế chấm điểm IoU cứng $0.5$ với greedy matching có xu hướng phạt nặng các trường hợp vết nứt dài liên tục bị mô hình ngắt làm 2 bounding box liên tiếp, hoặc ngược lại gộp 2 vết nứt gần nhau thành 1 bounding box lớn bao quanh. Mặc dù trên thực tế hệ thống đã định vị chính xác khu vực mặt đường bị hư hại cần sửa chữa, metric vẫn tính là 1 False Positive và 1 False Negative. Trong ứng dụng thực tế của bài toán Road Inspection, việc bổ sung metric Soft-IoU hoặc tính điểm theo diện tích pixel che phủ (Mask mIoU) sẽ phản ánh sát thực tế hơn. Ngoài ra, việc lưu trữ và khôi phục riêng các trọng số float32 của projector trong E2/E3 là bài học kỹ thuật quan trọng khi làm việc với thư viện PEFT.
