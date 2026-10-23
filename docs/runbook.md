# Runbook — Chạy framework và tìm output

Đối chiếu source code ngày 2026-10-22. Framework giải bài toán **binary crack semantic segmentation** bằng CNN → Projection → Swin → Decoder, với Dice + Focal + PH khi training. Tài liệu này hướng dẫn chạy từ source bằng PowerShell trên Windows; mỗi lệnh bên dưới chạy từ thư mục gốc repository.

Thiết kế: [architecture](RESEARCH_FRAMEWORK_ARCHITECTURE.md), [spec](spec.md), [implementation plan](implementation_plan.md). Phạm vi đã kiểm chứng: [IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md).

## 1. Mở môi trường chạy

### Máy hiện tại đã có `.venv`

Mỗi khi mở terminal mới, chạy:

```powershell
Set-Location D:\structural-defect-detection
$sdd = (Resolve-Path .\.venv\Scripts\sdd.exe).Path
$sddPython = (Resolve-Path .\.venv\Scripts\python.exe).Path
& $sdd --help
& $sdd doctor --device cuda:0
```

`& $sdd` gọi CLI của package trong virtual environment; không cần activate hoặc đổi execution policy. `doctor` in JSON về Python, PyTorch, CUDA, device và môi trường ra terminal, **không tự lưu file**. Với CPU, dùng `doctor --device cpu`.

### Cài trên máy mới

Dùng Python 3.12; nếu chưa có Python Launcher `py`, thay bằng đường dẫn Python 3.12 đã cài. Không tạo lại `.venv` của máy đang dùng.

```powershell
Set-Location D:\structural-defect-detection
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -e ".[dev]" -c requirements.txt
```

Sau đó chạy các lệnh khai báo `$sdd`, `$sddPython` ở trên. `requirements.txt` chứa các phiên bản đã dùng để kiểm thử. Máy CPU dùng wheel index `https://download.pytorch.org/whl/cpu` và cấu hình `runtime.device: cpu`, `runtime.amp: false`. Trên Linux dùng executable tương ứng trong `.venv/bin/`; các tham số CLI giữ nguyên.

Package được cài editable bằng `-e`: chỉnh Python source sẽ được dùng ở lần gọi CLI tiếp theo. Không chạy trực tiếp `src/.../cli.py`; có thể gọi module bằng `.venv\Scripts\python.exe -m structural_defect_detection --help`.

## 2. Quy tắc cấu hình và đường dẫn

Thứ tự ưu tiên: default trong code → YAML `base` → experiment YAML → machine YAML → `--set KEY=VALUE`. File `resolved.yaml` trong mỗi run ghi cấu hình cuối cùng sau khi gộp.

| Loại đường dẫn | Được tính từ đâu? | Ví dụ |
|---|---|---|
| `base` bên trong YAML | Thư mục chứa YAML đó | `configs/experiments/hybrid_ph.yaml` kế thừa `../base.yaml` |
| `data.sources[].root`, `output.root`, checkpoint, input, CLI `--output` thông thường | Working directory nếu là đường dẫn tương đối | `.cache/runs_3070ti` → `D:\structural-defect-detection\.cache\runs_3070ti` |
| `source.manifest`; image/mask/valid-mask trong manifest | Dataset root | `manifest.csv` → `data/CUBIT-InSeg/manifest.csv` |
| `images`, `labels`, `output_dir`, `groups_csv` trong CUBIT layout | Dataset root | `prepared/crack_semantic_v1` → `data/CUBIT-InSeg/prepared/crack_semantic_v1` |
| **Riêng `data split --output`** | Dataset root | `manifest_split.csv` nằm trong dataset root |

Tên `data/CUBIT-InSeg/data.yaml` là metadata của dataset, **không phải experiment config để truyền vào `--config`**. Adapter nhận mapping/đường dẫn qua `--layout configs/datasets/cubit_yolo.local.yaml`.

| Cấu hình | Device / micro-batch | Thư mục chứa các run |
|---|---|---|
| Không truyền `--machine`, dùng base hiện tại | CPU / 2 | `.cache/runs/` |
| `configs/machines/rtx3070ti_8gb.yaml` | CUDA / 1 | `.cache/runs_3070ti/` |
| `configs/machines/rtx3060_12gb.yaml` | CUDA / 2 | `.cache/runs_3060/` |
| `configs/machines/rtx4090_24gb.yaml` | CUDA / 4 | `.cache/runs_4090/` |

Machine YAML chỉ thay runtime, output root và dataset roots. Effective batch mặc định là 8; gradient accumulation tương ứng là 8/4/2 với ba GPU profiles. Profile 3060 và 4090 chưa được đo trên máy vật lý. Dùng đường dẫn có dấu `/` trong YAML để tránh escape của chuỗi Windows.

## 3. Chạy thử bằng dữ liệu thật với `--dry-run`

`train --dry-run` kiểm tra một số ít **ảnh thuộc training split thật**: đọc ảnh/annotation → tạo mask → training transforms → model → Dice/Focal/PH → backward. Không cập nhật trọng số, không tạo optimizer state/checkpoint, không chạy epoch đầy đủ và không đọc test split. Không có cơ chế tự tạo ảnh/nhãn thay thế khi thiếu dữ liệu.

### Chưa có manifest: dùng trực tiếp YOLO polygons

Có thể chạy ngay sau khi giải nén CUBIT và xác minh mapping; không cần chuẩn hóa toàn bộ dataset trước:

```powershell
& $sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --dry-run --layout configs/datasets/cubit_yolo.local.yaml --max-samples 2 --run-id cubit-dry-run
```

`--layout` chọn chế độ rasterize nhãn của những ảnh được chọn **trong bộ nhớ**. Nhãn gốc và dataset root không bị sửa; không tạo manifest nghiên cứu. Mapping/policy giống bước prepare ở mục 4. Gặp nhãn lỗi trong mẫu được chọn thì dry-run failed, không tự bỏ qua để tìm mẫu khác.

### Đã có manifest và semantic masks

Bỏ `--layout` để dùng pipeline manifest:

```powershell
& $sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --dry-run --max-samples 2 --run-id cubit-prepared-dry-run
```

Mặc định lấy tối đa 2 mẫu train theo thứ tự sample ID/tên ảnh đã sắp xếp; nhiều nguồn dùng thứ tự trong config, vì vậy giới hạn nhỏ có thể chưa bao phủ mọi nguồn. `--max-samples` phải lớn hơn 0; nếu dữ liệu ít hơn giới hạn, report ghi số mẫu thực đọc. Dùng micro-batch/precision và transforms của config hiện tại; việc đọc mẫu diễn ra trực tiếp trong process, không kiểm DataLoader workers, scheduler hoặc checkpoint/resume. Crop ngẫu nhiên có thể chỉ chứa background; xem `foreground_pixels` trong report. Đây là kiểm khả năng chạy, không phải metric accuracy hay full leakage audit.

CPU: bỏ `--machine` để dùng base CPU hiện tại. Dùng run ID mới mỗi lần; không kết hợp `--dry-run` với `--resume` hoặc `--retry`. `--layout`/`--max-samples` trên lệnh train chỉ hợp lệ khi có `--dry-run`.

### Output dry-run

Terminal trả JSON gồm `kind: real_data_dry_run`, `status`, `samples_checked`, `optimizer_steps: 0` và đường dẫn `output`. Với lệnh đầu:

```text
.cache/runs_3070ti/dry-runs/cubit-dry-run/
├── resolved.yaml
├── report.json
└── previews/
    ├── <key>_input.png       # Crop RGB thật sau transform
    ├── <key>_target.png      # Ground truth crack của crop: 0/255
    ├── <key>_valid.png       # 255 = valid, 0 = ignore
    ├── <key>_mask.png        # Prediction của model đang khởi tạo
    ├── <key>_overlay.png
    └── <key>.json
```

`report.json` ghi ảnh/nhãn và hashes nguồn, mapping khi đọc YOLO, crop/flips, loss components, PH counters/gradient norm, model gradient norm, shape logits, thời gian forward/backward và môi trường. Tọa độ preview là **pixel của training patch đã transform**, không phải ảnh UAV nguyên gốc; metadata giữ kích thước/crop gốc để truy vết. Preview của model chưa train không chứng minh khả năng nhận diện crack.

Nếu lỗi sau khi tạo thư mục, report có `status: failed`, error và traceback. Folder không có `state.json`/`result.json` của experiment hoặc checkpoint `.pt`; sweep collect không lấy dry-run làm kết quả nghiên cứu. Chạy thử thành công trên 2 mẫu **không có nghĩa cả 6.996 annotation đều hợp lệ**.

## 4. Chuẩn bị CUBIT-InSeg

### Vì sao cần bước này dù đã tải dataset chính thống?

Tải dataset chỉ cung cấp ảnh và annotation theo định dạng của tác giả. CUBIT đang lưu mỗi instance dưới dạng YOLO polygon (`class_id x1 y1 x2 y2 ...`, tọa độ 0–1), còn loss semantic segmentation của framework nhận mask theo từng pixel cùng ảnh. Dataset chính thống vẫn cần chuyển sang hợp đồng dữ liệu của model.

`data prepare` làm bốn việc:

1. Ghép đúng ảnh–file nhãn, đọc mapping **0 = crack / 1 = spalling**.
2. Rasterize polygon ở kích thước ảnh gốc và hợp các crack instances thành một foreground mask.
3. Tạo valid mask để **ignore spalling và overlap**, phù hợp scope v1 crack/background. Số `0` trong YOLO là class ID crack; số `0` trong PNG semantic mask lại là background, nên không thể dùng trực tiếp class ID làm giá trị mask.
4. Ghi manifest với đường dẫn, kích thước, release, metadata và **giữ nguyên train/validation/test** của dataset; lưu report truy vết chuyển đổi.

`data validate` là bước sau prepare: kiểm mask/ảnh alignment, giá trị lớp, coverage, exact duplicates và group leakage nếu có metadata. Nó không tạo lại nhãn hoặc chia lại benchmark. Nếu dataset đã có semantic masks và manifest đúng schema, không cần chạy lại prepare; chỉ validate rồi train.

Dry-run đọc YOLO trực tiếp ở mục 3 dùng cùng phép rasterize nhưng chỉ trên vài ảnh trong bộ nhớ. Training đầy đủ vẫn cần bước chuẩn hóa này để DataLoader đọc mask/manifest ổn định và kiểm audit toàn bộ nguồn.

### Trạng thái dữ liệu hiện tại

`data.yaml` người dùng cung cấp xác định `0: crack`, `1: spalling`, `nc: 2`. Layout đã kiểm khớp 5.596/699/701 cặp ảnh–nhãn train/validation/test:

```text
data/CUBIT-InSeg/
├── data.yaml
├── train/images/    + train/labels/
├── val/images/      + val/labels/
└── test/images/     + test/labels/
```

**Bước prepare hiện còn bị chặn bởi 13 file chứa polygon suy biến**: 8 train, 1 validation, 4 test. Những polygon được phát hiện rasterize thành 0 pixel. Xem báo cáo cục bộ [.cache/cubit_yaml_check.json](../.cache/cubit_yaml_check.json) để biết file/dòng. Chưa sửa hoặc bỏ nhãn nguồn. Cần xử lý theo policy có ghi nhận trước khi chạy prepare; không diễn giải việc chuẩn hóa thất bại là lỗi mapping lớp.

Layout [cubit_yolo.local.yaml](../configs/datasets/cubit_yolo.local.yaml) đã điền mapping. `coverage: full_crack` vẫn là giả định cần kiểm theo annotation protocol; tên lớp trong YAML không chứng minh vùng chưa đánh dấu đều là background. `groups_csv` có thể bổ sung metadata công trình/phiên chụp/ảnh gốc để kiểm leakage.

### Chuẩn hóa sau khi xử lý annotation bị từ chối

```powershell
& $sdd data prepare --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --adapter cubit-yolo --layout configs/datasets/cubit_yolo.local.yaml
& $sdd data validate --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --output .cache/cubit_audit.json
```

`prepare` là bước chuyển annotation, không train model. Với cấu hình hiện tại, output là:

```text
data/CUBIT-InSeg/
├── manifest.csv
└── prepared/crack_semantic_v1/
    ├── preparation_report.json
    ├── train/<sample>.png + <sample>_valid.png
    ├── validation/<sample>.png + <sample>_valid.png
    └── test/<sample>.png + <sample>_valid.png
```

| Artifact | Ý nghĩa |
|---|---|
| `manifest.csv` | Một dòng mỗi ảnh: sample/dataset ID, split, đường dẫn ảnh/mask/valid-mask, kích thước và metadata |
| `<sample>.png` | Semantic mask: 255 = crack, 0 = nền hoặc vùng không được giám sát; phải đọc cùng valid mask |
| `<sample>_valid.png` | 255 = pixel được giám sát, 0 = ignore; spalling và vùng overlap với spalling bị ignore |
| `preparation_report.json` | Mapping, layout, release, số instances, ignored pixels, hash nhãn nguồn và policy chuyển đổi |
| `.cache/cubit_audit.json` | Số mẫu/split, manifest/split hashes, exact duplicates, thiếu group metadata; được tạo bởi lệnh validate |

`prepare` in preparation report dạng JSON. `validate` in audit JSON; nếu bỏ `--output` thì không lưu audit thành file. Validate kiểm ảnh đã decode, alignment/mask values và leakage theo exact image/group IDs hiện có; near-duplicates vẫn cần audit riêng. Khi gặp lỗi, lệnh dừng và có thể chưa tạo report hoàn chỉnh.

`prepare` từ chối manifest hoặc output directory đã tồn tại. Nếu conversion bị ngắt, thư mục mask có thể đã được tạo một phần: giữ để kiểm lỗi, chọn `output_dir` mới trong layout khi chạy lại; nếu manifest cũng đã tồn tại, chọn manifest mới trong source config. Không ghi đè annotation nguồn. `data validate` đọc toàn bộ manifest nên trên ảnh 4K có thể mất thời gian dù terminal chưa có tiến trình từng ảnh.

### Dataset khác hoặc chia split mới

CLI còn có `--adapter cubit-coco`, `paired`, `deepcrack`; xem `& $sdd data prepare --help`. Với nhiều source, `prepare` và `split` cần chọn một source bằng `--source`; `validate` luôn kiểm tất cả source trong config.

Chỉ với manifest có mẫu `unassigned`, tạo manifest mới bằng lệnh sau; không dùng để tự chia lại benchmark CUBIT:

```powershell
& $sdd data split --config configs/experiments/hybrid_ph.yaml --source cubit-inseg --output manifest_split.csv --ratios 0.7 0.15 0.15
```

Output nằm tại `data/CUBIT-InSeg/manifest_split.csv`; terminal trả `manifest`, `samples`, `next`. Split giữ assignments có sẵn, xử lý group/exact-duplicate linkage và chỉ phân bổ mẫu chưa gán. Sau khi audit, cập nhật `source.manifest` sang file mới.

## 5. Train baseline và đọc artifacts

Chỉ chạy sau khi prepare/validate thành công và kiểm overlay annotation. Hai lệnh sau tạo hai run độc lập, cùng seed và cấu hình nền; đối chứng chỉ tắt PH:

```powershell
& $sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --run-id cubit-ph-seed42
& $sdd train --config configs/experiments/hybrid_no_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --run-id cubit-no-ph-seed42
```

Đường dẫn training là **`<output.root>/<run-id>/`**. Ví dụ run đầu lưu tại `D:\structural-defect-detection\.cache\runs_3070ti\cubit-ph-seed42\`. Nếu bỏ `--run-id`, framework sinh ID theo UTC timestamp và experiment hash; ID được trả ra terminal.

Để kiểm khả năng chạy trước experiment dài, dùng dry-run riêng:

```powershell
& $sdd train --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --dry-run --max-samples 2 --run-id cubit-ph-check --set output.root=.cache/cubit-checks
```

Lệnh này đọc tối đa 2 mẫu train từ manifest, không cập nhật trọng số; output là `.cache/cubit-checks/dry-runs/cubit-ph-check/`. Full training hiện mặc định 20 epochs; hyperparameters vẫn là candidate chưa được xác minh bằng experiment dài trên CUBIT.

Terminal in `epoch=... loss=... val_dice=... ph=...`, rồi JSON với `run_id`, `status`, `best_validation_dice`. Có thể chưa có dòng epoch trong khi kiểm toàn bộ dữ liệu hoặc chạy epoch đầu. Đây không phải stdout chỉ chứa JSON.

| File/thư mục trong run | Nội dung và thời điểm ghi |
|---|---|
| `resolved.yaml` | Config cuối cùng của run; dùng lại khi evaluate/predict/resume |
| `state.json` | `running`, `completed` hoặc `failed`; khi failed có error, loại lỗi và traceback nếu lỗi xảy ra trong run |
| `data_report.json` | Audit dữ liệu và fingerprints đã dùng để train |
| `environment.json` | Phiên bản, hardware, seed-related identity, batch/accumulation, AMP và resolution |
| `environment_resume_<timestamp>.json` | Metadata bổ sung khi chạy lại và `environment.json` đã có |
| `initialization.json` | Trạng thái khởi tạo/load weights; tạo khi khởi tạo run, không tạo lại khi resume |
| `history.json` | Lịch sử cập nhật sau mỗi epoch: loss components, PH statistics, validation Dice, thời gian, optimizer steps; `epoch` trong file bắt đầu từ 0 |
| `validation_metrics.json` | Metrics chi tiết validation của epoch vừa đánh giá; bị thay thế ở lần đánh giá sau |
| `best_validation_metrics.json` | Metrics tại epoch cải thiện checkpoint selection |
| `checkpoints/last.pt` | Trạng thái ở cuối epoch gần nhất đã hoàn tất để resume |
| `checkpoints/best.pt` | Trạng thái tại epoch có validation selection score cao nhất |
| `best/validation_metrics.json` | Đánh giá lại best model sau khi hoàn tất training |
| `best/predictions/` | Mask, overlay và JSON cho tối đa `evaluation.prediction_samples` ảnh validation; mặc định 4 |
| `result.json` | Kết quả hoàn tất: best validation, history, checkpoint paths, identity, elapsed time và peak GPU memory |
| `.lock` | Khóa tạm trong lúc run; bình thường được xóa khi run thoát |

Run lỗi sớm có thể chỉ có một phần artifacts; `result.json` chỉ được tạo khi training hoàn tất. `best_checkpoint` và `last_checkpoint` trong result là đường dẫn **tương đối với thư mục run**. `.pt` chứa model, optimizer, scheduler, scaler, RNG và tiến độ epoch; đây không phải file ONNX/export cho drone.

Checkpoint selection dùng `selection.crack_dice`: trung bình Dice theo ảnh trong từng dataset, sau đó trung bình giữa các dataset. Không dùng test để chọn checkpoint. `overall.dice` có thể khác selection score nếu có nhiều dataset với số ảnh khác nhau.

## 6. Resume và retry

Resume run bị gián đoạn từ cuối epoch gần nhất:

```powershell
& $sdd train --config .cache/runs_3070ti/cubit-ph-seed42/resolved.yaml --resume .cache/runs_3070ti/cubit-ph-seed42/checkpoints/last.pt
```

Output tiếp tục nằm trong **run chứa checkpoint**, kể cả `output.root` trong config khác. Restore model/optimizer/scheduler/scaler/RNG và lịch sử; phần epoch chưa hoàn tất sẽ chạy lại. Config, dataset fingerprints và AMP phải tương thích. Đổi `training.epochs` không phải cách kéo dài run cũ: thay đổi đó làm identity không khớp.

Nếu `state.json` là `failed`, có thể dùng:

```powershell
& $sdd train --config .cache/runs_3070ti/cubit-ph-seed42/resolved.yaml --run-id cubit-ph-seed42 --retry
```

`--retry` tự dùng `last.pt` nếu có; nếu chưa có checkpoint thì khởi tạo lại trong cùng run. Không retry một run `completed`. Run thuộc sweep phải retry qua `sweep run --retry-failed` để giữ `plan_id`.

Nếu process bị kill cứng, `.lock` hoặc trạng thái `running` có thể còn lại. Kiểm tra process trước khi xử lý lock; framework không tự lấy quyền sở hữu run đang chạy. Sửa nguyên nhân OOM/non-finite trước khi retry; framework không tự giảm resolution hoặc tắt PH. Khi đổi protocol, dùng run mới.

## 7. Evaluate và predict

### Evaluate với segmentation ground truth

Dùng config của chính run và checkpoint đã chọn. Chỉ mở test sau khi khóa protocol:

```powershell
& $sdd evaluate --config .cache/runs_3070ti/cubit-ph-seed42/resolved.yaml --checkpoint .cache/runs_3070ti/cubit-ph-seed42/checkpoints/best.pt --split test --output .cache/evaluations/cubit-ph-seed42-test
```

Terminal trả JSON `split`, `metrics` (overall), `output`. File chi tiết nằm tại:

```text
.cache/evaluations/cubit-ph-seed42-test/
├── test_metrics.json
└── predictions/<key>_mask.png + <key>_overlay.png + <key>.json
```

Metrics gồm Dice, IoU, precision, recall, confusion counts và `window_betti0_mae`; báo theo `samples`, `datasets`, `domains`, `overall`, kèm threshold/policy và fingerprints. `window_betti0_mae` đo sai lệch số connected components trong các cửa sổ hợp lệ; giá trị thấp hơn tốt hơn. Ignore pixels không tham gia metrics; ảnh toàn ignore bị loại, prediction/target đều rỗng có Dice/IoU = 1.

Toàn bộ split được chấm điểm, nhưng chỉ tối đa **4 ảnh** được lưu visualization theo cấu hình mặc định. `--split validation` tạo `validation_metrics.json`; `--split external_test` tạo `external_test_metrics.json`. External test cần manifest/config nguồn ngoài khai báo rõ và giữ evaluation protocol; framework kiểm exact-image overlap với training/validation. Không đổi threshold/tile size qua `--set` rồi kỳ vọng load cùng checkpoint: các trường này tham gia kiểm experiment identity.

### Predict ảnh mới, không cần ground truth

```powershell
& $sdd predict --config .cache/runs_3070ti/cubit-ph-seed42/resolved.yaml --checkpoint .cache/runs_3070ti/cubit-ph-seed42/checkpoints/best.pt --input data/CUBIT-InSeg/test/images --output .cache/predictions/cubit-ph-seed42
```

`--input` nhận một file ảnh hoặc thư mục (quét đệ quy). Thư mục input hiện nhận `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`. Thay bằng folder ảnh mới của bạn khi inference thực. Lệnh không cần đọc segmentation manifest; model/config phải khớp checkpoint.

Terminal trả `images`, `output`. Tất cả ảnh input được xuất vào folder `--output`:

| File | Ý nghĩa |
|---|---|
| `<key>_mask.png` | Mask nhị phân 0/255, threshold mặc định 0.5, kích thước ảnh gốc |
| `<key>_overlay.png` | Ảnh gốc với vùng crack tô đỏ bán trong suốt |
| `<key>.json` | Metadata sample, threshold, shape `[H,W]`, số components, các vùng `xywh_pixels` và `area_pixels` |
| `prediction_run.json` | Hash checkpoint, config và số ảnh đã xử lý; ghi sau khi hoàn tất toàn bộ input |

`key` là 20 ký tự hash từ dataset ID và sample ID, không phải tên ảnh. Tra `sample.sample_id`/`sample.image` trong JSON để tìm ảnh nguồn. Các tọa độ là pixel của ảnh gốc, không phải mm. Predict không trả Dice/IoU, không tự lưu probability map và không chạy PH loss. Chọn folder output mới cho mỗi lần đánh giá/dự đoán để tránh ghi đè hoặc lẫn file cũ.

## 8. YAML sweep và nhiều máy

Sweep cần manifest đã chuẩn hóa và validate được. File [hybrid_ph.yaml](../configs/sweeps/hybrid_ph.yaml) hiện có 2 learning rates × 3 nhánh PH (off, window 32, window 64) × 3 seeds = **18 trials**. Các run dùng một GPU/run và chạy tuần tự trên mỗi máy.

### Tạo plan cố định một lần

```powershell
& $sdd sweep plan --config configs/sweeps/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --output .cache/cubit_plan.json
```

Terminal trả `plan_id`, `trials`, `path`; file `.cache/cubit_plan.json` lưu danh sách trial configs, run IDs, seeds và data/split hashes. Lệnh này kiểm dữ liệu nhưng chưa train. Plan không được ghi đè/sửa tay; tạo tên file mới khi đổi search space.

Để random search, tạo YAML sweep khác: đặt `search.method: random`, thêm `search.seed` và `search.trials`, giữ các danh sách giá trị ứng viên. `search.trials` là số lượt lấy mẫu trước khử trùng và ghép training seeds; số run cuối cùng xem ở JSON trả về. PH-off được khử trùng theo các tham số PH không hoạt động.

### Chạy một máy hoặc chia hai máy

Một máy xử lý toàn bộ plan:

```powershell
& $sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx3070ti_8gb.yaml
```

Hoặc dùng hai shards thay cho lệnh trên. Máy A:

```powershell
& $sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx3070ti_8gb.yaml --shard-index 0 --shard-count 2
```

Máy B, sau khi sao chép **cùng plan**, source code và dataset đã chuẩn hóa; chỉnh dataset root trong machine YAML của máy B:

```powershell
& $sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx4090_24gb.yaml --shard-index 1 --shard-count 2
```

Máy A lưu `.cache/runs_3070ti/<trial-run-id>/`; máy B lưu `.cache/runs_4090/<trial-run-id>/`. Đây là các run độc lập, không phải distributed training của một model. Nội dung dữ liệu/splits phải giống plan dù đường dẫn máy khác nhau.

`sweep run` in tiến trình từng run rồi JSON có các **danh sách** `completed`, `skipped`, `failed`; không có file sweep-status riêng, trạng thái nằm trong từng `state.json`. Chạy lại cùng shard sẽ skip completed. Retry failed bằng:

```powershell
& $sdd sweep run --plan .cache/cubit_plan.json --machine configs/machines/rtx3070ti_8gb.yaml --shard-index 0 --shard-count 2 --retry-failed
```

Cũng có thể chọn một hoặc nhiều `--run-id` lấy từ plan, nhưng không kết hợp lựa chọn này với sharding. Khi plan mới có cùng trial configs với plan cũ, dùng output roots riêng để tránh xung đột `plan_id`.

### Thu và tổng hợp kết quả

Sao chép đầy đủ các run folders từ máy A vào `results/machine-a/`, từ máy B vào `results/machine-b/` trên máy tổng hợp. Cấu trúc phải là `results/machine-a/<run-id>/state.json`, không lồng thêm một tầng output root. Collect không tự kết nối hoặc tải kết quả từ máy khác.

```powershell
& $sdd sweep collect --plan .cache/cubit_plan.json --root results/machine-a --root results/machine-b --output .cache/cubit_summary
```

Nếu chỉ chạy một máy, thay các `--root` bằng `--root .cache/runs_3070ti`.

| Output | Nội dung |
|---|---|
| `.cache/cubit_summary/summary.json` | Per-run rows, mean/std theo experiment family và dataset/domain, statuses, số completed/failed, danh sách run chưa có state |
| `.cache/cubit_summary/runs.csv` | Metrics của từng run/seed/dataset hoặc domain; chỉ được tạo nếu có rows từ completed runs |
| Terminal JSON | `completed`, `failed` là số lượng; `missing_or_pending` là danh sách; `output` là folder report |

Collect dùng **best validation metrics**, không gom kết quả test/evaluate riêng. Run failed/running không được đưa vào mean/std; trạng thái xem trong `summary.json.statuses`. Field `missing_or_pending` hiện chứa IDs chưa tìm thấy `state.json`, không bao gồm run đã có trạng thái `running`. `std` là sample standard deviation, `null` khi chỉ có một giá trị hợp lệ. Collect từ chối run ID trùng giữa các roots hoặc sai plan/config/data hashes; dùng output folder mới để tránh giữ lại CSV từ lần collect cũ không có rows.

## 9. Profiling và kiểm thử

```powershell
& $sdd profile --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --layout configs/datasets/cubit_yolo.local.yaml --iterations 10 --warmup 2 --output .cache/profiles/hybrid-ph-3070ti.json
& $sddPython -m pytest -q --data-root data/CUBIT-InSeg --data-layout configs/datasets/cubit_yolo.local.yaml
& $sddPython -m ruff check src tests
```

`profile` in và lưu cùng report JSON tại đường dẫn `--output`: parameters, model/pipeline mean/p95 latency và FPS, PH statistics, training forward/backward probe, peak GPU memory (bytes), môi trường và cấu hình đo. Input là một **training crop thật** (lấy mẫu đầu tiên theo cùng quy tắc dry-run), batch 1; inference FP32, training probe theo AMP config. Report lưu source hashes và transform. Dùng `--layout` khi đọc raw YOLO; bỏ cờ này nếu đã có manifest. Pipeline không gồm đọc ảnh/rasterize/ghi overlay; probe không cập nhật trọng số, không có optimizer state. Không dùng kết quả trên crop này làm throughput ảnh UAV 4K, full training hoặc hiệu năng drone.

`pytest`/`ruff` trả kết quả trên terminal. Bộ test dùng crop từ ảnh và polygon **official train split** của CUBIT, với provenance/hashes trong thư mục tạm pytest. Các partition train/validation/test nhỏ của fixture chỉ để kiểm phần mềm (resume, sweep, evaluation), không sửa split gốc hay dùng official test để chọn model. Thiếu dataset sẽ báo lỗi, không fallback sang dữ liệu giả. Negative tests chỉ gây lỗi trên bản sao annotation. File demo generator và script overfit cũ đã được loại bỏ; chạy thử model dùng `train --dry-run`.

## 10. Kết quả terminal, mã thoát và lỗi thường gặp

Phần lớn lệnh trả JSON khi thành công; riêng train/sweep còn có log trước JSON. API nội bộ trả Python dict/arrays, nhưng CLI biểu diễn kết quả bằng stdout và lưu artifacts như các mục trên. Không redirect toàn bộ stdout của train thành `.json` để dùng thay `result.json`.

Các lỗi được CLI xử lý thường trả **exit code 2** và thông báo `sdd: ...` trên stderr; argparse sai tham số cũng trả 2. `sweep run` trả **1** khi summary có failed runs; thành công trả **0**. Lỗi chưa được xử lý hoặc process bị ngắt có thể có mã khác. `--debug` phải đặt trước subcommand để xem traceback:

```powershell
& $sdd --debug data validate --config configs/experiments/hybrid_ph.yaml --machine configs/machines/rtx3070ti_8gb.yaml --output .cache/cubit_audit_debug.json
$LASTEXITCODE
```

| Hiện tượng | Kiểm tra / cách xử lý |
|---|---|
| Không thấy `sdd` hoặc lỗi OpenMP khi import torch | Gọi executable trong `.venv` như mục 1; tránh Python Anaconda global của máy hiện tại |
| `manifest.csv` chưa tồn tại | Hoàn tất prepare; kiểm source root/manifest, không truyền dataset `data.yaml` làm experiment config |
| `degenerate polygon` | Xem file/dòng nhãn; hiện đã phát hiện 13 file CUBIT cần policy xử lý |
| `Prepared masks already exist` / `Manifest exists` | Kiểm kết quả lần trước; đổi output_dir/manifest cho conversion mới, không ghi đè source |
| `Image content leakage` / `Group leakage` | Audit split/nguồn ảnh và group metadata; ghi protocol nội bộ nếu cần, không tự sửa benchmark để bỏ lỗi |
| `Run exists` | Dùng run ID mới, hoặc resume/retry đúng trường hợp ở mục 6 |
| `Checkpoint/config mismatch` | Dùng `resolved.yaml` của checkpoint; chỉ thay runtime/path qua machine config phù hợp |
| CUDA OOM / non-finite loss hoặc gradients | Xem `state.json`, hardware, precision và batch; đổi protocol tường minh trong run mới nếu cần |
| Không có PH windows hợp lệ | Kiểm ignore coverage, crop size và PH window; xem PH counters trong history |
| Không có `runs.csv` sau collect | Kiểm `summary.json`: CSV chỉ xuất khi có completed metric rows |

Artifacts trong `.cache/` và dữ liệu tải về thường bị Git ignore. Khi chuyển máy hoặc lưu kết quả nghiên cứu, sao lưu dataset/manifest, plan và toàn bộ run folders; `git push` không tự mang theo checkpoint hay predictions. Xem đường dẫn thực trong config và `state.json` trước khi kết luận một run đã hoàn tất.
