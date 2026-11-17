# Audit CUBIT-InSeg — 16/11/2026

**Kết luận: đủ để chạy pilot kiểm tra pipeline có điều kiện; chưa đủ để công bố benchmark hoặc khả năng tổng quát sang công trình/phiên chụp độc lập.** Integrity và mapping hiện tại đạt, nhưng có nội dung hình ảnh trùng vùng nứt giữa các split, thiếu metadata nhóm vật lý, và annotation/background chưa được xác nhận đầy đủ. Audit không sửa dataset, manifest, config hoặc trọng số model.

Phạm vi là bản CUBIT cục bộ tại `D:/structural-defect-detection/data/CUBIT-InSeg` và protocol binary-crack `crack_v1`, không bao gồm CIF. Áp dụng skill `data-engineer` cho hợp đồng dữ liệu, lineage và quality gates; áp dụng pha ML-pipeline của `ai-ml` cho supervision, transforms và đánh giá. Các ảnh/nhãn test được xem để audit chất lượng và leakage, không dùng để train model, chọn threshold hoặc đo/chọn hyperparameter.

| Split | Ảnh/nhãn nguồn | Mẫu trong manifest |
|---|---:|---:|
| Train | 5.596 | 2.523 |
| Validation | 699 | 299 |
| Test | 701 | 317 |
| Tổng | 6.996 | 3.139 |

Mapping nguồn là `0=crack`, `1=spalling`. Semantic mask là `0=background`, `128=ignore`, `255=crack`. Toàn bộ mẫu retained có crack được gán nhãn. Các ảnh spalling-only không được coi tự động là crack-negative.

**[HIGH] Có nội dung thị giác trùng lắp giữa train và holdout trong manifest đang dùng.**

Quét 6.996 thumbnail và 24.468.510 cặp bằng pHash/dHash phát hiện 20 cặp ứng viên xuyên split mà cả hai ảnh đều nằm trong manifest: 12 train–test, 8 train–validation. Chúng liên quan 37 ảnh riêng biệt: 20 train, 8 validation, 9 test. Đây là danh sách ứng viên theo ngưỡng, không phải tuyên bố 20 cặp là byte-duplicate hay đã có physical group IDs được xác minh.

Đã xem cả 20 cặp trên bảy contact sheet. Các ví dụ rõ về cùng vùng nứt/cấu trúc bất quy tắc gồm:

| Train | Holdout | Quan sát |
|---|---|---|
| `CR1_1426.JPG` | test `CR1_1427.JPG` | Cùng các nhánh nứt và góc cửa, framing dịch nhẹ |
| `CR1_1438.JPG` | validation `CR1_1439.JPG` | Cùng junction và đường nứt ngang phía trên cửa |
| `CR1_1440.JPG` | validation `CR1_1441.JPG` | Cùng hình dạng nứt trên cửa, góc chụp thay đổi nhỏ |
| `CR1_2196.JPG` | test `CR1_2197.JPG` | Cùng đường nứt và vết phong hóa bất quy tắc trên cornice |

![Các ví dụ train/holdout trùng vùng hình ảnh](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/near_duplicates/prepared_cross_split_review_01.png)

Bằng chứng: [20 cặp ứng viên](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/near_duplicates/prepared_cross_split_candidates.csv), [ghi nhận visual review](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/near_duplicates/visual_review.json), [phương pháp và thống kê](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/near_duplicates/summary.json). Hai hash không bằng nhau vẫn có thể là ảnh cùng cảnh hoặc vùng chụp trùng nhau. Quan sát này làm mất cơ sở diễn giải holdout là cảnh độc lập; mức tăng/giảm điểm model do leakage chưa được đo.

Không có `structure_id` hay `capture_session_id` thực cho bất kỳ mẫu nào trong 3.139 mẫu. `source_image_id` hiện là hash ảnh đã decode, giúp bảo vệ exact-content identity nhưng không chứng minh độc lập theo công trình hay phiên chụp. Vì vậy `samples_without_group_metadata=0` trong validator không được diễn giải là đã có physical grouping đầy đủ. Hành vi liên quan nằm tại [manifest.py](D:/structural-defect-detection/src/structural_defect_detection/data/manifest.py:144) và [prepare_training_data.py](D:/structural-defect-detection/scripts/prepare_training_data.py:200).

Ưu tiên xử lý: review nguồn ảnh ở độ phân giải gốc và metadata tác giả, thiết lập nhóm cảnh/công trình/phiên chụp, rồi tạo protocol nội bộ có version với mỗi nhóm nằm trong một split. Giữ bản phát hành và assignments official để truy vết; không âm thầm thay benchmark. Nguyên tắc đánh giá theo nhóm độc lập được giải thích trong [tài liệu chính thức scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

**[HIGH] Background/full-crack coverage chưa được chứng nhận độc lập.**

`full_crack` trong manifest là giả định được pipeline công khai: vùng ngoài annotation được xem là background trong ảnh có crack annotation. Mỗi ảnh nguồn hiện chỉ có annotation cho một trong hai lớp; điều đó không chứng minh lớp còn lại không xuất hiện trong ảnh. Retained masks có tổng ignored pixels bằng 0, phù hợp với việc chỉ giữ ảnh crack-labelled, nhưng không chứng minh mọi pixel còn lại sạch defect.

Xem [policy mapping](D:/structural-defect-detection/configs/datasets/cubit_cif_crack.yaml:13), [prepare_record](D:/structural-defect-detection/scripts/prepare_training_data.py:199), [contract evidence](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/contracts_evidence.json). README chính thức và abstract truy cập được chưa xác lập annotation exhaustive; full article không truy cập được trong audit (HTTP 403), nên audit không kết luận paper thiếu protocol. Cần kiểm annotation hướng dẫn gốc và review vùng chưa gán nhãn trước khi coi background là negative đã xác nhận.

**[MEDIUM] Supervision của sampler hiện tại thưa; đây là rủi ro hiệu quả học, chưa phải bằng chứng accuracy thấp.**

Đo train-only trên 128 ảnh, chọn 32 ảnh từ mỗi quartile theo mật độ foreground (seed 42), tái hiện toàn bộ 20 epoch cấu hình: **1.879/2.560 crop không có pixel crack được gán nhãn, tương đương 73,3984%**. Ước lượng weighting theo kích thước quartile là 73,4038%; khoảng bootstrap theo cụm ảnh trên mẫu là 71,2182–75,5920%. Khoảng này không bao phủ annotation uncertainty hoặc tổng quát sang công trình mới.

| Quartile mật độ annotation | Ảnh sampled | Crop không có crack có nhãn |
|---|---:|---:|
| Q1 thấp nhất | 32 | 89,06% |
| Q2 | 32 | 77,97% |
| Q3 | 32 | 66,72% |
| Q4 cao nhất | 32 | 59,84% |

Trong 20 crop/ảnh, 5/128 ảnh sampled không đưa một pixel crack nào vào train crop. Median tỷ lệ pixel crack của mỗi ảnh từng được nhìn thấy là 9,46%; median vùng ảnh từng được nhìn thấy là 9,28%. Đây là exposure của sampler, không phải coverage của nhãn hoặc metric model. Prepared train có 0,7695% pixel foreground; mẫu raw 64 ảnh có 0,3780% crack theo pixel-weighted aggregation, là hai population/denominator khác nhau.

![Đo crop train không có crack có nhãn](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/training_crop_exposure.png)

16 probe trên 8 cặp ảnh/mask thật tại epoch 0 và 19 khớp chính xác transform production về crop coordinates, target sau flip, foreground/valid counts. Active crop/tile giữ độ phân giải gốc, augmentation chỉ flip; không có bước downsize trong các mode đang dùng. Chưa kiểm thin-feature learnability bên trong model. Bằng chứng và toàn bộ 2.560 crop: [ML notes](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/ml_sampling_notes.md), [JSON](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/ml_sampling.json), [CSV](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/ml_sampling_crops.csv).

Nên so sánh sampler hiện tại với một protocol foreground-aware có giữ background crops, trong pilot có version và split đã xử lý grouping. Không đổi sampler ngầm giữa các experiment, và không chọn policy bằng test score.

**[MEDIUM] Population đánh giá chỉ gồm ảnh crack-positive.**

Cả 3.139 mẫu retained có foreground trong ground truth. Segmentation metrics vẫn tính pixel false positives trên vùng được xem là background, nhưng không đo image-level false alarms trên cảnh được xác nhận crack-free hoặc spalling-only. Cần một protocol negative/ignore đã được review nếu deployment cần chỉ số đó; không đổi nhãn spalling-only thành zero-mask một cách tự động.

**[MEDIUM] Quarantine cả ảnh làm mất annotation khác còn hợp lệ.**

Có 13 polygon bị parser đánh dấu signed-area: 12 thẳng hàng, một bow-tie tự giao tại `train/CR1_2993:1`. Cả 13 cho raster rỗng với COCO hiện tại ở kích thước gốc; không suy điều này chỉ từ signed area bằng 0. Năm polygon khác được parser chấp nhận nhưng COCO raster rỗng: `CR1_1446:33`, `CR1_1447:31`, `CR1_1448:31`, `CR1_2777:27`, `CR1_2778:23`.

Năm trường hợp sau có diện tích hình học khoảng 0,5005–1,0033 pixel², cao khoảng 1,002 pixel, gần mép ảnh. Đây là trường hợp biên rasterization, chưa chứng minh annotation sai về ngữ nghĩa. Cả 18 ảnh bị loại vẫn có annotation khác có raster dương. Trong 15 ảnh crack bị loại, có **280 instance khác có raster dương**, union **951.363 pixel**; ba ảnh bị loại còn lại là spalling-labelled. Giữ quarantine hiện tại cho baseline có thể truy vết; chỉ phục hồi/correct trong phiên bản dữ liệu mới sau review. Không tự xóa polygon nhỏ để hợp thức hóa dữ liệu. Xem [chi tiết 18 mẫu](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/annotation_issue_detail.csv) và [probe rasterization](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/annotation_semantics.json).

**Các kiểm tra đã đạt và giới hạn tương ứng.**

| Kiểm tra | Kết quả |
|---|---|
| Fresh raw inventory, pairing | 6.996 ảnh/6.996 nhãn; không thiếu/orphan/rỗng |
| Header và thumbnail decode | 6.996/6.996 đạt; thumbnail không thay thế full-resolution decoding |
| Raw image/label SHA256 | Cả 6.996 cặp khớp snapshot dùng để mapping trước đó |
| Root/prepared manifest, mapping, validation SHA | Khớp provenance; không đổi trong audit |
| Prepared full validation | Tái sử dụng lượt validate toàn bộ 3.139 mẫu hoàn tất ngay trước audit; không quét lại mọi full-resolution pixel trong lượt này |
| Exact-file duplicates ở raw | 589 nhóm; 222 nhóm xuyên split/444 bản ghi |
| Exact duplicates lọt vào prepared | 0; các nhóm raw trên thuộc ảnh spalling-only đã bị loại |
| pHash/dHash positive control | Thu hồi toàn bộ 589 cặp byte-identical đã biết |
| Near-duplicate candidates trong prepared holdout | 20 cặp/37 ảnh; đã xem thumbnail cả 20, có ví dụ trùng vùng nứt rõ |
| Sampling implementation | 16 actual-transform probes đạt; train-only measurements |

Fresh scan chi tiết và biểu đồ: [raw EDA](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/raw_eda/eda_summary.json), [raw inventory](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/raw_eda/image_inventory.csv), [split/class chart](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/raw_eda/eda_split_class_incidence.png). Tổng bản ghi bị loại khỏi manifest là 3.857: 3.839 spalling-only, 13 parser-quarantined, 5 rasterizer edge cases.

**Chi phí chạy cần đo trước experiment dài.**

Config hiện tại tạo 83.720 tile windows cho mỗi validation pass trên 299 ảnh, so với 2.523 train crops/epoch. Đây là đếm workload xác định từ size/overlap, không phải latency hoặc throughput đã đo. Chạy pilot và đo latency/VRAM trước full 20 epochs; mọi thay đổi resolution/validation protocol cần được khóa và ghi version.

**Tái lập và bằng chứng.**

Các lệnh chạy từ project root, chọn output mới vì script từ chối ghi đè:

```powershell
.\.venv\Scripts\python.exe scripts/eda_cubit_inseg.py --root data/CUBIT-InSeg --output reports/cubit_raw_eda_rerun --mask-samples 64 --overlay-samples 8 --seed 42
.\.venv\Scripts\python.exe reports/cubit_audit_2026-11-16/near_duplicate_audit.py --output reports/cubit_near_duplicates_rerun
.\.venv\Scripts\python.exe reports/cubit_audit_2026-11-16/ml_sampling_audit.py --output reports/cubit_sampling_rerun
```

Near-duplicate script dùng inventory snapshot tại `reports/cubit_audit_2026-11-16/raw_eda`, từ JPEG draft256 đến grayscale LANCZOS, pHash DCT 32×32/top8×8 bỏ DC, median63, và dHash ngang 9×8. Chọn ứng viên khi pHash Hamming ≤8 và dHash Hamming ≤10. Không thay ngưỡng theo model/test score. Đây là heuristic không bao phủ mọi crop, rotation, perspective, biến đổi exposure hoặc ảnh cùng phiên chụp. Bảy contact sheet chỉ trình bày 20 prepared cross-split candidates; CSV giữ toàn bộ 2.238 cặp qua ngưỡng trong raw, bao gồm 589 exact-file pairs và 1.649 nonexact candidates.

Machine-readable kết luận: [audit_summary.json](D:/structural-defect-detection/reports/cubit_audit_2026-11-16/audit_summary.json). Các bảng crop và probe rasterization nằm cùng thư mục. Không thêm dependency và không tạo model/training run trong audit này.

Nguồn sơ cấp cho provenance/annotation: [repository CUBIT-InSeg chính thức](https://github.com/BenyunZhao/CUBIT-InSeg), [CUHK publication abstract](https://research.cuhk.edu.hk/en/publications/from-instance-segmentation-to-physical-quantification-high-resolu/). Ảnh minh họa được dẫn từ CUBIT-InSeg, Benyun Zhao và cộng sự; dataset công bố CC BY 4.0. Không áp taxonomy/metadata của các bản CUBIT mở rộng khác vào release cục bộ 6.996 ảnh này.

Chưa xác minh exhaustive annotation, physical group IDs, PII toàn corpus, accuracy hoặc mức score inflation do leakage; chưa train model. Các giới hạn này phải được giữ trong mọi diễn giải kết quả pilot.
