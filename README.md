# Structural Defect Detection

Framework nghiên cứu Python/PyTorch cho binary crack semantic segmentation. Baseline: **CNN → Projection → Swin blocks → Decoder**, có multiscale CNN skips và **Dice + Focal + persistent-homology loss**. CLI `sdd`, YAML config và YAML sweep hỗ trợ các run độc lập trên nhiều máy.

Thiết kế: [architecture](docs/RESEARCH_FRAMEWORK_ARCHITECTURE.md), [spec](docs/spec.md), [implementation plan](docs/implementation_plan.md). Kết quả kiểm chứng và giới hạn: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## 1. Cài đặt

Khuyến nghị Python 3.12 trong virtual environment riêng. Máy hiện tại đã có `.venv` với PyTorch CUDA 12.8; tránh dùng Python Anaconda global đang có xung đột OpenMP. Trên máy mới:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dev]" -c requirements.txt
sdd doctor --device cuda:0
```

Để dùng CPU, thay index của lệnh cài torch bằng `https://download.pytorch.org/whl/cpu`, chọn `runtime.device: cpu` và `runtime.amp: false`. Linux kích hoạt bằng `source .venv/bin/activate`. Các phiên bản đã kiểm nằm trong `requirements.txt`; máy khác vẫn cần kiểm driver/hardware.

## 2. Chạy thử trên dữ liệu thật

Xem [runbook](docs/runbook.md) để biết cách chạy và từng đường dẫn output. CLI dùng `train --dry-run` thay cho demo tạo dữ liệu. Chưa có manifest thì đọc trực tiếp CUBIT YOLO trong bộ nhớ:

```powershell
sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --dry-run --layout configs/datasets/cubit_yolo.local.yaml --max-samples 2 --run-id cubit-check
```

Đã có manifest thì bỏ `--layout`. Dry-run đọc tối đa N mẫu train, kiểm transforms/model/Dice/Focal/PH/backward và lưu `report.json`, `resolved.yaml`, previews tại `.cache/runs_3070ti/dry-runs/cubit-check/`. Không optimizer update/checkpoint; không đọc test split; không fallback dữ liệu giả. Mỗi lần dùng ID mới. Kiểm trên mẫu nhỏ không thay full dataset audit; annotation lỗi được báo, không tự skip.

## 3. Chuẩn bị CUBIT-InSeg

Dataset đầu tiên người dùng chọn là [CUBIT-InSeg](https://github.com/BenyunZhao/CUBIT-InSeg). Nhãn đã giải nén tại `data/CUBIT-InSeg/{train,val,test}/labels` chứa **YOLO polygons**: mỗi dòng là `class_id x1 y1 x2 y2 ...`, tọa độ chuẩn hóa. File `data/CUBIT-InSeg/data.yaml` do người dùng cung cấp xác định **0 = crack, 1 = spalling**, `nc: 2`. Ngày 2026-10-22 đã kiểm đường dẫn và ghép đủ 5.596 train, 699 validation, 701 test. Repository công bố CC BY 4.0.

Kiểm nhãn phát hiện **13 file có polygon suy biến** (8 train, 1 validation, 4 test); 13 polygon được phát hiện đều rasterize thành 0 pixel ở kích thước ảnh gốc. Adapter hiện từ chối những polygon này, nên bước `data prepare` còn bị chặn. Chi tiết file/dòng, hash của `data.yaml` và phạm vi kiểm tra nằm trong [.cache/cubit_yaml_check.json](.cache/cubit_yaml_check.json). Chưa sửa hoặc bỏ nhãn nguồn; kiểm tra này chưa thay thế audit coverage, overlay và leakage.

1. Layout hiện tại đã đúng: mỗi split có `images` và `labels`. Khi dùng máy khác, sửa `roots.cubit-inseg` trong machine YAML theo dataset root thực.
2. Dùng [cubit_yolo.local.yaml](configs/datasets/cubit_yolo.local.yaml), đã điền mapping và nguồn `data.yaml`; target là crack, spalling được ignore. `coverage: full_crack` vẫn là giả định cần audit theo annotation protocol: YAML tên lớp không chứng minh mọi vùng còn lại là negative. File `.example.yaml` giữ mapping rỗng cho các release khác.
3. Xử lý các polygon suy biến theo policy có ghi nhận trước khi chuẩn hóa. File nhãn thiếu bị từ chối; file `.txt` rỗng chỉ hợp lệ với coverage đã xác nhận. Bounding boxes không phải polygon masks.
4. Nếu có metadata, cung cấp `groups_csv` với cột `image,structure_id,capture_session_id,source_image_id`. Cột `image` tương đối với dataset root, giống manifest.
5. Sau khi giải quyết các annotation bị từ chối, chuẩn hóa và kiểm tra:

```powershell
sdd data prepare --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --adapter cubit-yolo --layout configs/datasets/cubit_yolo.local.yaml
sdd data validate --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --output .cache/cubit_audit.json
```

Adapter hợp các instances crack thành foreground; defect khác và overlap với defect khác được ignore. Xuất mask, valid mask, manifest và preparation report; giữ official splits. Polygon được rasterize ở kích thước gốc bằng COCO rasterizer; cần kiểm overlay biên nhãn trước full training. Không suy đoán nguồn công trình/phiên chụp/ảnh gốc từ tên ảnh. Nếu phát hiện leakage trong official split, audit và lưu protocol internal riêng, không tự chia lại benchmark.

Layout/manifest paths tương đối với dataset root. Output masks dùng thư mục mới, không ghi đè annotation gốc. Release, class mapping và hash annotation nguồn được lưu. Nếu conversion bị ngắt, dùng output directory mới sau khi kiểm tra lỗi. Region outputs dùng pixel; không suy ra mm khi thiếu calibration.

Adapter `cubit-coco` vẫn hỗ trợ release/conversion **COCO JSON polygons/RLE** với `configs/datasets/cubit_ingest.example.yaml`; JSON phải chứa tên category rõ ràng. Crowd/ignore annotations của COCO cũng được ignore.

Có thêm adapters `paired` cho ảnh/mask trùng relative stem và `deepcrack` cho layout `train_img/train_lab`, `test_img/test_lab`; yêu cầu `--release` và `--coverage full_crack`. Với manifest `unassigned`, `sdd data split --config ... --output manifest_split.csv` tạo manifest mới; cập nhật source config sau khi kiểm tra.

## 4. Train, resume và đánh giá

```powershell
sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --run-id cubit-ph-seed42
sdd train --config configs/experiments/hybrid_no_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --run-id cubit-no-ph-seed42
sdd train --config .cache/runs_3070ti/cubit-ph-seed42/resolved.yaml --resume .cache/runs_3070ti/cubit-ph-seed42/checkpoints/last.pt
```

Checkpoint selection dùng macro dataset mean của per-image validation Dice. Training không tự chạy test. Gọi `evaluate --split test` sau khi khóa protocol; `external_test` cho nguồn bên ngoài được khai báo rõ. Evaluation kiểm checkpoint/config/data identity và exact-image leakage từ train/validation sang external test. Không dùng test để chọn threshold, checkpoint hoặc hyperparameter.

Override tường minh: `--set training.seed=17`, `--set training.epochs=30`. Cấu hình mới tạo run mới. Resume yêu cầu experiment/data/AMP tương thích và thực hiện ở ranh giới epoch; CLI không kéo dài run cũ bằng cách đổi epochs.

Machine YAML chỉ đổi `runtime`, `output.root` và dataset `roots`. Effective batch phải chia hết cho micro-batch; accumulation được suy ra. Các profile 8/12/24 GB có micro-batch 1/2/4, effective batch 8; cần đo trên hardware thực. CNN dùng GroupNorm để giảm phụ thuộc micro-batch. AMP mặc định `bfloat16`; `runtime.amp_dtype: float16` là lựa chọn tường minh có GradScaler. Dtype được lưu và kiểm khi resume.

OOM/non-finite làm run failed, không tự giảm resolution/bỏ PH. `--retry` dành cho run failed; có checkpoint cuối epoch thì resume từ đó. Run lock chống trùng output; các run CUDA từ cùng checkout dùng device lock. Không quản lý các chương trình GPU ngoài framework hoặc distributed lock giữa máy khác nhau. Khi process bị kill cứng, kiểm process đã dừng trước khi dọn `.lock`/đánh dấu run failed; framework không tự cướp lock.

Mỗi run lưu resolved config, state, environment, data fingerprints, initialization report, history, best/last checkpoints, validation metrics/predictions và result. Checkpoints chứa optimizer/RNG state; chỉ load artifact của mình hoặc nguồn tin cậy.

## 5. YAML sweep trên nhiều máy

```powershell
sdd sweep plan --config configs/sweeps/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --output .cache/cubit_plan.json
sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx3070ti_8gb.yaml --shard-index 0 --shard-count 2
# Máy thứ hai: sao chép cùng plan, chỉnh machine roots rồi chạy shard 1/2.
sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx4090_24gb.yaml --shard-index 1 --shard-count 2
sdd sweep collect --plan .cache/cubit_plan.json --root results/machine-a --root results/machine-b --output .cache/cubit_summary
```

`--root` là thư mục chứa run directories đã thu từ từng máy; thay ví dụ bằng đường dẫn thực. Plan khóa configs/seeds/data hashes trước chạy; không sửa JSON. Đổi dataset sources bằng plan riêng; machine paths có thể khác nếu data content/splits giống nhau.

`search.method`: `grid` hoặc `random`; random thêm `seed` và `trials` (số lượt lấy mẫu candidate hữu hạn). `search.groups` là các nhánh parameter space dùng chung `search.parameters`. Trial trùng bị loại, gồm PH settings không hoạt động khi weight 0; số trial thực có thể ít hơn random sampling budget. Các `seeds` được ghép vào cấu hình còn lại.

Runner chạy tuần tự; completed được skip, failed chỉ retry với `--retry-failed`. Có thể chọn `--run-id` thay shard. Collect từ chối duplicate IDs hoặc sai plan/config/data hashes; báo missing/failed và giữ từng dataset/domain/seed trước aggregate. Dùng output roots mới khi tạo plan mới có configs trùng với plan cũ.

## 6. Model và PH implementation

Baseline khởi tạo từ đầu: CNN channels 32/64/128, strides 1/2/4; projection 96; 2 Swin blocks alternating regular/shifted windows, 3 heads, window 8; decoder có CNN skips. Không thêm patch merging. Đây là cấu hình ban đầu, chưa tối ưu trên CUBIT. `model.init_weights` nhận tensor-only state dict tương thích và báo layers loaded/reinitialized/unused; không giả nhận pretrained Swin RGB cho CNN features.

PH dùng GUDHI cubical superlevel filtration qua `1 - probability`, coface pairing và optimal diagram assignment có diagonal. Pairing/assignment chạy CPU; loss gather critical-cell values từ tensor gốc nên gradient về model. Essential H0 cap tại filtration 1; H0 mặc định, H1 tùy chọn. Không ép một connected component cho mọi ảnh.

Domain PH là closed windows không overlap/resize. Window có ignore pixel được bỏ qua; số windows bỏ qua và pairs bị cắt được log. Mặc định lọc persistence ≤0.01, giữ tối đa 64 pairs/window/dimension. Matching dùng squared L2, lấy mean window/dimension trong từng ảnh rồi mean theo ảnh để giữ loss nhất quán khi đổi micro-batch. Ảnh không có window hợp lệ đóng góp PH bằng 0; một epoch không có window nào khi PH chạy mỗi step sẽ báo lỗi. Weight, warmup và frequency có YAML.

Diagram loss piecewise differentiable, gradient nằm ở critical pixels; target feature unmatched có thể đóng góp hằng số. PH không bảo đảm tạo một nhánh hoàn toàn vắng mặt; Dice/Focal vẫn cung cấp pixel supervision. Window topology không mô tả kết nối ngoài cửa sổ. Các lựa chọn này cần ablation trên annotation thực.

Tham khảo API: [GUDHI cubical persistence](https://gudhi.inria.fr/python/latest/cubical_complex_ref.html), [torchvision Swin](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/swin_transformer.html).

## 7. Kiểm thử và profiling

```powershell
python -m pytest -q --data-root data/CUBIT-InSeg --data-layout configs/datasets/cubit_yolo.local.yaml
ruff check src tests
sdd profile --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --layout configs/datasets/cubit_yolo.local.yaml --iterations 10 --output .cache/profile.json
```

Profiler báo model/pipeline latency và forward/backward probe có PH. Pipeline gồm normalization, resize/tiling, model và khôi phục ảnh; không gồm đọc file/ghi overlay. Probe chưa có optimizer state, không đại diện throughput full training. Input là crop từ training image/annotation thật, batch 1; report có provenance. Bỏ `--layout` khi dùng prepared manifest. Không suy ra throughput ảnh 4K hay hiệu năng drone. Bộ test cũng lấy crop từ official train và làm việc trên bản sao tạm; thiếu dữ liệu sẽ báo lỗi. Energy unavailable khi thiếu telemetry.

`hybrid_efficient_cnn_ph.yaml` dùng depthwise CNN variant giữ strides; `cnn_only_ph.yaml` thay Swin bằng identity trên projected features, giữ decoder/PH. Variants có shape/backward tests; so sánh accuracy/compute trên ảnh thật, multiclass và edge/export thuộc experiment tiếp theo.
