# Runbook chạy ablation phân đoạn vết nứt

Hướng dẫn này dùng PowerShell 7 trên Windows và profile RTX 3060 12 GB. Kết quả gồm hai manifest `crack_v1`, audit chuyển đổi, ba run đối chứng trên CUBIT-InSeg và metric test. CIF Full dùng để đánh giá chuyển miền sau khi chọn checkpoint trên CUBIT.

## 1. Cài môi trường

Clone repo hoặc mở checkout đã có. Kiểm tra checkout có commit `d0f2b8c` hoặc bản mới hơn chứa script prepare và các config ablation. Chạy toàn bộ lệnh từ thư mục gốc repo.

```powershell
git clone https://github.com/Zephyrus000/structural-defect-detection.git
Set-Location structural-defect-detection
git rev-parse HEAD
py -3.14 -m venv .venv
```

```powershell
$python = ".\.venv\Scripts\python.exe"
$sdd = ".\.venv\Scripts\sdd.exe"
$hf = ".\.venv\Scripts\hf.exe"
$machine = "configs/machines/rtx3060_12gb.yaml"
$baseline = "configs/experiments/cubit_cnn_no_ph.yaml"

& $python --version
& $python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
& $python -m pip install -e ".[dev]" -c requirements.txt
& $python -m pip install pyarrow==26.0.1 huggingface_hub==2.4.0 -c requirements.txt
& $python -m pip check
& $sdd doctor --device cuda:0
```

`doctor` cần nhận GPU đích, Python 3.14.x và CUDA runtime 13.0. Trên GPU khác, chọn profile trong `configs/machines/`. Dùng cùng profile cho ba ablation để so thời gian. [Wheel PyTorch chính thức](https://pytorch.org/get-started/previous-versions/#v2110)

## 2. Tải dữ liệu gốc

### CUBIT-InSeg

Tải ảnh và nhãn polygon YOLO cho cả ba split qua Google Drive hoặc Baidu do tác giả cung cấp. Giải nén vào `data/CUBIT-InSeg`. Đối chiếu đường dẫn thực với layout. Nếu ảnh validation được giải nén trực tiếp vào `val/images`, đổi `splits.validation.images` thành `val/images` trong YAML. [Nguồn tải CUBIT-InSeg](https://github.com/BenyunZhao/CUBIT-InSeg#cubit-inseg-dataset) · [Layout địa phương](configs/datasets/cubit_yolo.local.yaml)

| Split gốc | Thư mục ảnh trong layout | Thư mục nhãn | Số cặp dự kiến |
|---|---|---|---:|
| Train | `train/images` | `train/labels` | 5.596 |
| Validation | `val/images/images` | `val/labels` | 699 |
| Test | `test/images` | `test/labels` | 701 |

Mỗi ảnh cần một file `.txt` có cùng relative stem trong thư mục nhãn tương ứng. Kiểm metadata đi kèm bản tải để xác nhận class ID `0 = crack` và `1 = spalling`.

### CIF Full

Lệnh tải cố định revision và chỉ chọn các shard `full`. Bản Parquet chiếm khoảng 99,34 GB. Dành thêm dung lượng cho ảnh trích từ Parquet, mask, CUBIT và checkpoint. Khoảng 250 GB trống trước khi tải là dự trù cho quy trình giữ cả dữ liệu gốc và bản prepared. [Dataset CIF](https://huggingface.co/datasets/ibm-research/cif-dataset) · [CLI Hugging Face](https://huggingface.co/docs/huggingface_hub/en/guides/cli#hf-download)

```powershell
$cifRevision = "758cdc9295456971027ee44553c2db013a60993b"
& $hf download ibm-research/cif-dataset --repo-type dataset --revision $cifRevision `
    --include "data/*_full-*.parquet" --local-dir data/cif-dataset
if ($LASTEXITCODE -ne 0) { throw "Tải CIF thất bại. Chạy lại cùng lệnh để tiếp tục." }

$expected = @{ train = 188; val = 78; test = 76 }
foreach ($split in $expected.Keys) {
    $count = @(Get-ChildItem data/cif-dataset/data -File -Filter "${split}_full-*.parquet").Count
    if ($count -ne $expected[$split]) { throw "Thiếu shard CIF $split ($count/$($expected[$split]))" }
}
```

## 3. Tạo mapping, mask và manifest

Dùng mapping `crack_v1` để tái lập cùng bài toán. CUBIT dùng class `0`. CIF gộp `2 = Crack`, `3 = Net-crack` và `4 = Crack with precipitation`. Mask dùng `255` cho crack, `128` cho ignore và `0` cho background. Vùng ignore ghi đè phần chồng lấn. Script giữ ảnh CUBIT có nhãn crack, loại ảnh lỗi và xử lý ảnh trùng sau khi giải mã. Các mẫu giữ lại kế thừa split gốc. Mọi thay đổi mapping cần một lượt prepare và audit mới. [Mapping](configs/datasets/cubit_cif_crack.yaml) · [Schema CIF](https://huggingface.co/datasets/ibm-research/cif-dataset#schema)

```powershell
$audit = ".cache/cubit-cif-audit-v1"
& $python -u scripts/prepare_training_data.py --workers 4 --audit-dir $audit
if ($LASTEXITCODE -ne 0) { throw "Prepare thất bại. Kiểm tra lỗi trước khi chạy lại." }
```

Script chuẩn bị cả hai nguồn trong một lượt. Nó đọc hai đường dẫn nguồn và manifest từ `configs/experiments/cubit_cif_ph.yaml`. Bước này chỉ tạo dữ liệu, vì vậy cấu hình PH trong file đó không tham gia tính toán. [Script prepare](scripts/prepare_training_data.py)

| Đầu ra | Nơi tìm |
|---|---|
| Manifest CUBIT | `data/CUBIT-InSeg/prepared/crack_v1/manifest.csv` |
| Manifest CIF | `data/cif-dataset/prepared/crack_v1/manifest.csv` |
| Số mẫu và lý do loại theo split | `$audit/audit.json` và `$audit/inventory.csv` |
| Nhóm ảnh trùng xuyên split trước lọc | `$audit/raw_cross_split_duplicates.csv` |

Lượt prepare tham chiếu giữ 2.523 train, 299 validation, 317 test cho CUBIT và 8.760 train, 1.878 validation, 1.783 test cho CIF. Nếu lệch số lượng, kiểm revision, layout và lý do loại trong inventory trước khi so với kết quả nhóm.

Nếu prepare bị ngắt, script từ chối ghi đè thư mục đã tồn tại. Đổi tên các thư mục `prepared/crack_v1` đã tạo dở và chọn audit directory mới trước khi chạy lại. Nếu đã nhận bản prepared từ nhóm, giữ đúng đường dẫn trên và chuyển thẳng sang validate. Khi cần tái tạo mask CIF, giữ lại hoặc tải lại Parquet gốc.

## 4. Validate dữ liệu và chạy thử

```powershell
$validationReport = "$audit/manifest_validation.json"
& $sdd data validate --config configs/experiments/cubit_cif_ph.yaml --output $validationReport |
    Out-File "$audit/manifest_validation.stdout.log" -Encoding utf8
if ($LASTEXITCODE -ne 0) { throw "Validate manifest thất bại." }

Get-Content $validationReport -Raw | ConvertFrom-Json |
    Select-Object samples, splits, manifest_hash, split_hash
```

Validate giải mã toàn bộ ảnh và mask để kiểm kích thước, giá trị mask và ảnh trùng xuyên split. Với bản tham chiếu, báo cáo có 15.560 mẫu. Audit hash không bao phủ ảnh gần trùng. Nếu dùng bản prepared chia sẻ và chưa có thư mục audit, tạo bằng `New-Item -ItemType Directory -Force $audit` trước khi chạy.

```powershell
$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$variants = [ordered]@{
    baseline = $baseline
    cldice = "configs/experiments/cubit_cnn_cldice.yaml"
    ecc = "configs/experiments/cubit_cnn_ecc.yaml"
}
foreach ($entry in $variants.GetEnumerator()) {
    & $sdd train --config $entry.Value --machine $machine --dry-run --max-samples 8 `
        --run-id "preflight-$($entry.Key)-$tag"
    if ($LASTEXITCODE -ne 0) { throw "Dry-run lỗi cho $($entry.Key)." }
}
```

Mỗi dry-run cần `status: completed`. Mở `.cache/runs_3060/dry-runs/<run-id>/report.json` để xem loss, gradient và số crop chứa crack. Nếu các crop đều không có crack, dry-run mới kiểm đường chạy với nền. Dùng thêm một mẫu có foreground trước khi đánh giá loss topology.

## 5. Chạy ba ablation trên CUBIT

Dùng cùng manifest, seed, kiến trúc CNN-only, crop `256×256`, effective batch 8 và lịch optimizer. Profile RTX 3060 bật 4 workers, micro-batch 8, BF16 và tiling trên GPU. `training.epochs=20` là ngân sách cấu hình ban đầu. Nếu nhóm đổi ngân sách, chốt chung trước khi chạy vì cosine scheduler phụ thuộc số epoch.

| Run | Loss | Trọng số PH |
|---|---|---:|
| Baseline | Dice + Focal | 0 |
| Soft-clDice | Dice + Focal + `0.1 × soft-clDice` | 0 |
| ECC | Dice + Focal + `0.01 × ECC` | 0 |

```powershell
$seed = 42
$epochs = 20
$runRoot = ".cache/runs_3060"
$runIds = @{}
foreach ($entry in $variants.GetEnumerator()) {
    $runId = "cubit-$($entry.Key)-s$seed-$tag"
    $runIds[$entry.Key] = $runId
    & $sdd train --config $entry.Value --machine $machine `
        --set "training.seed=$seed" --set "training.epochs=$epochs" --run-id $runId
    if ($LASTEXITCODE -ne 0) { throw "Train lỗi ở $runId. Xem state.json." }
}
```

Các run chạy tuần tự trên một GPU. Lượt quét manifest đầu run và validation ảnh gốc mỗi epoch có thể kéo dài. Khi run hoàn tất, kiểm `state.json`, `resolved.yaml`, `history.json`, `result.json` và `checkpoints/best.pt`. Để lặp lại cho seed khác, đổi `$seed` và chạy cả ba biến thể.

Để tiếp tục run bị gián đoạn sau một epoch đã lưu, đặt `$run` thành thư mục run đó và dùng `last.pt` cùng `resolved.yaml`:

```powershell
& $sdd train --config "$run/resolved.yaml" --resume "$run/checkpoints/last.pt"
```

## 6. Đánh giá test và chia sẻ kết quả

Trainer chọn `best.pt` bằng Dice trên validation. Chốt checkpoint và ngưỡng validation trước khi chạy test. Lệnh dưới dùng cấu hình đã resolve của từng run.

```powershell
foreach ($entry in $variants.GetEnumerator()) {
    $run = "$runRoot/$($runIds[$entry.Key])"
    & $sdd evaluate --config "$run/resolved.yaml" --checkpoint "$run/checkpoints/best.pt" `
        --split test --output "$run/test"
    if ($LASTEXITCODE -ne 0) { throw "Evaluate lỗi ở $run." }
}
```

Gửi cho nhóm commit ID, seed, `resolved.yaml`, `data_report.json`, `history.json`, `environment.json`, `result.json` và `test/test_metrics.json`. So theo cùng seed và kiểm `manifest_hash`, `split_hash`, tổng optimizer steps. Báo Dice, IoU, precision, recall, `window_betti0_mae` và thời gian epoch. Nếu báo precision/recall gộp theo pixel, tính từ tổng TP/FP/FN và ghi rõ cách gộp. Metric Betti-0 trong repo tính theo cửa sổ, nên khi kết luận về độ liên tục hãy kèm ảnh dự đoán ở cùng các vị trí.

## 7. Đánh giá chuyển miền trên CIF

Dùng phần này sau khi đã chốt checkpoint trên CUBIT. Tạo manifest riêng cho các mẫu CIF test và đặt split thành `external_test`. Manifest CIF gốc vẫn là đầu vào cho các thí nghiệm CIF sau này.

```powershell
$cifRoot = "data/cif-dataset"
$externalManifest = "prepared/crack_v1/external_test.csv"
Import-Csv "$cifRoot/prepared/crack_v1/manifest.csv" |
    Where-Object split -eq test |
    ForEach-Object { $_.split = "external_test"; $_ } |
    Export-Csv "$cifRoot/$externalManifest" -NoTypeInformation -Encoding utf8

# Chọn run bằng kết quả CUBIT validation, ví dụ baseline.
$run = "$runRoot/$($runIds['baseline'])"
$externalSources = ConvertTo-Json -InputObject @(@{
    name = "cif-dataset"
    root = $cifRoot
    manifest = $externalManifest
}) -Compress
& $sdd evaluate --config "$run/resolved.yaml" --checkpoint "$run/checkpoints/best.pt" `
    --set "data.sources=$externalSources" --split external_test --output "$run/cif-test"
if ($LASTEXITCODE -ne 0) { throw "Evaluate CIF lỗi ở $run." }
```

Báo riêng metric CIF và CUBIT. CIF test đo chuyển miền từ CUBIT sang CIF theo mapping `crack_v1`. Khi so với một run train chung, giữ cùng CIF test và báo kết quả từng nguồn. Để đo khả năng tổng quát hóa sang công trình mới sau khi học từ CIF, cần tách các nhóm công trình giữa train và test. [Giao thức split CIF](https://arxiv.org/html/2605.18413v1#S3.SS2)
