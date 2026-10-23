# Trạng thái triển khai — 2026-10-22

Package OOP theo composition đã có baseline CNN → Projection → Swin → Decoder, Dice/Focal/PH, CLI/YAML, resume, sweep/sharding/collect. Theo yêu cầu mới, luồng tạo dữ liệu và các fixture ảnh/nhãn giả đã được bỏ; chạy thử dùng **dữ liệu thật + `train --dry-run`**. Hướng dẫn đầy đủ: [runbook](docs/runbook.md).

## Luồng chạy thử hiện tại

- `train --dry-run --layout ...` đọc tối đa N ảnh train và YOLO polygons thật, rasterize trong bộ nhớ bằng cùng policy của `data prepare`.
- `train --dry-run` không có layout dùng semantic masks từ manifest đã chuẩn hóa.
- Kiểm training transforms, logits, Dice/Focal/PH, backward, finite model/PH gradients. Không optimizer update/checkpoint, không chạy official test.
- Output riêng: `<output.root>/dry-runs/<run-id>/report.json`, `resolved.yaml`, `previews/`. Report ghi provenance, hashes, crop và số foreground/valid pixels. Failed được lưu với traceback; không skip nhãn lỗi.
- `profile` dùng một crop ảnh/nhãn thật; cần manifest hoặc raw YOLO layout, không sinh input thay thế.
- Đã bỏ `demo.py`, CLI `demo` và script overfit cũ. Tests không import hay dùng các artifact cũ để tạo dữ liệu.

## Kiểm chứng của phiên bản hiện tại

- **28 tests passed**, 48 upstream pycocotools/NumPy deprecation warnings; không có test skipped/failure. Lệnh: `.venv\Scripts\python.exe -m pytest -q --basetemp=.cache/pytest-real-data-final`.
- Test config/CLI, raw→prepared mask equivalence, real spalling ignore, reject polygon suy biến, mask/RLE roundtrip, alignment/tiling/padding, exact-image leakage.
- Ba model variants có forward/backward trên crop thật với kích thước lẻ. PH diagram đối chiếu GUDHI trên prediction thật, gradient về model, PH-off và all-ignore diagnostics.
- Dry-run kiểm sample limit, lỗi thiếu dữ liệu, output riêng, không có checkpoint và **mọi model state tensor không đổi** sau probe.
- Resume tại epoch boundary trên real crops đạt model tensors giống hệt run liên tục (`rtol=0`, `atol=0`); train→validation→checkpoint→evaluation hoàn tất trong phạm vi fixture.
- Grid/random sweep trên real crops kiểm kế hoạch cố định, PH-off dedup, hai shards, completed skip, failed/retry, duplicate IDs và data mismatch khi collect.
- Fixtures lấy crop từ **4 ảnh official train** với nhãn thật. Các partition nội bộ để kiểm phần mềm không phải split benchmark; không đọc ảnh/nhãn official validation/test để test model. Nguồn/hashes/crop lưu trong `provenance.json` ở thư mục tạm pytest. Thiếu dữ liệu thì fail, không fallback.
- GPU dry-run: [real-check-final](.cache/runs_3070ti/dry-runs/real-check-final/report.json), 4 ảnh train thật, crop 256², micro-batch 1, BF16, 1 crop có foreground; mọi batch có finite model/PH gradients, optimizer steps = 0. Các crop background cũng được giữ trong report.
- GPU profiler: [real-cubit-3070ti.json](.cache/profiles/real-cubit-3070ti.json), một real training crop 256², batch 1, 3 iterations/1 warmup. Chỉ là kiểm profiler, chưa đủ làm benchmark tốc độ hay hiệu năng ảnh 4K/drone.
- Môi trường lần kiểm này: Windows 11, Python **3.13.9** trong `.venv`, torch 2.11.0+cu128, torchvision 0.26.0+cu128, GUDHI 3.11.0, RTX 3070 Ti 8 GB. Phiên bản dependencies trong [requirements.txt](requirements.txt). Các máy 3060/4090 và Linux chưa kiểm vật lý.

## Quyết định và điều kiện còn mở

| Mã | Trạng thái |
|---|---|
| DEC-01 | CUBIT-InSeg, mapping 0 crack / 1 spalling theo YAML được cung cấp. Đã ghép đủ 5.596/699/701 ảnh–nhãn. Có 13 file chứa polygon suy biến (8/1/4), adapter vẫn từ chối; chưa sửa source labels. Coverage, group metadata và near-duplicates còn cần audit. |
| DEC-02 | Môi trường hiện tại chạy được như evidence trên; Python 3.12 từng dùng trước đó không phải phiên bản của lần kiểm này. |
| DEC-03 | Baseline 608.295 parameters, CNN channels 32/64/128, strides 1/2/4, projection 96, 2 Swin blocks/3 heads/window 8, decoder 32. Random initialization; pretrained backbone chưa chọn. |
| DEC-04 | GUDHI superlevel `1-p`, H0 mặc định, H1 opt-in; pairing CPU và critical values giữ graph. Backend hiện tại đã kiểm trên prediction thật. Chưa tái lập toàn bộ các case H1/finite-difference của bộ test cũ bằng dữ liệu thật. |
| DEC-05 | Candidate 256², GroupNorm, effective batch 8; dry-run trên 3070 Ti BF16 đã đạt. Chưa chạy full epoch CUBIT/đo throughput dài; FP16 và các profile GPU khác cần kiểm riêng. |
| DEC-06 | Dice/Focal 1/1, PH 0.05, threshold 0.5 là candidate; checkpoint chọn bằng validation macro dataset per-image Dice. Dry-run không chọn checkpoint/threshold. |
| DEC-07 | Efficient CNN và CNN-only có checks trên real crops. Nghiên cứu tối ưu, multiclass, export/container/edge còn thuộc hướng mở rộng. |

## Nghiệm thu còn thiếu

**Chưa nghiệm thu v1 trên toàn bộ dataset CUBIT.** Dry-run một số mẫu không thay full audit. Cần xử lý polygon suy biến theo policy có ghi nhận, xác minh coverage, chuẩn hóa toàn bộ mask/manifest và audit leakage. Sau đó chạy tiny real-data overfit, short controlled PH/no-PH experiments và full training theo protocol đã khóa. Những kiểm tra overfit cũ không được xem là bằng chứng của bộ test hiện tại.

Cross-machine physical execution, external-test leakage regression sau thay fixture, H1 đặc thù, finite-difference PH trên dữ liệu thật và intentional CUDA OOM handling chưa được tái kiểm trong phiên bản này. Giữ các mục này mở thay vì suy từ dry-run hoặc test crops.

Artifacts `.cache/` là cục bộ. GitNexus detect-changes theo git diff hiện chỉ nhìn thấy thay đổi tracked; nhiều source files còn untracked, nên không dùng kết quả graph đó làm bằng chứng review toàn bộ code. Các import/call sites và bộ tests được kiểm trực tiếp. Quy tắc ignore dataset đã được giới hạn thành `/data/` để package `src/structural_defect_detection/data/` không bị bỏ khỏi version control.
