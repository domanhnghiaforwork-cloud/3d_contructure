# `raamac_repro` — Tái hiện RAAMAC: video drone → point cloud 3D (.ply)

Thư mục **độc lập, tự chứa** (cách tổ chức như `professional_95/`): toàn bộ code +
kết quả hiện tại để dựng lại **point cloud từ video drone công trường Schependomlaan**
theo đúng tinh thần RAAMAC — mỗi tuần một chuyến bay → một cloud.

> **Tiêu chí thành công ở đây là HÌNH HỌC + cảm quan trực quan**, không phải điểm nhãn.
> Deliverable là file `.ply` để **bạn tự mở viewer và đánh giá**. Bản đồ căn chỉnh còn
> chưa đạt nên việc chấm theo nhãn (built/not-built) sẽ gây hiểu lầm — nhánh đánh giá
> nhãn chuẩn nằm ở `professional_95/`.

## Kết quả hiện tại (commit trong gói này)

- **Tuần 26** đã dựng bằng COLMAP (một cloud, SIFT SfM + MVS dense) từ `week26.mp4`:
  `4,451,437` điểm, sau căn chỉnh vào hệ tham chiếu cho **coverage_ref@0.2 ≈ 8.0%**
  (scale 0.378, Chamfer ≈ 2.20 m, median recon→ref ≈ 0.18 m). Đây là mức phủ thấp vì
  nguồn là **một video drone marketing** có nhiều góc khuất/nội thất không quay được.
- File `.ply` trong `results/` là **bản giảm mẫu còn 2,000,000 điểm** (~54 MB). Lý do:
  GitHub **chặn file > 100 MB**, nên `fused.ply` gốc (120 MB) không push trực tiếp được.
  Chạy lại pipeline (`--recon`) sẽ sinh cloud đầy đủ.
- Nhánh **đa-shot** (tách theo hard-cut rồi hợp nhất các cloud đã căn chỉnh, để phủ
  rộng hơn) **đã có code xong** (`run_colmap_shots.py`, `remote_colmap_shots.py`,
  `merge_shots.py`) nhưng **chưa chạy tới kết quả** (đã dừng theo yêu cầu) → tuần 27–30
  chưa có cloud.

## Cách xem kết quả (không cần cài gì)

Mở `viewer/viewer_repro_week26.html` bằng trình duyệt (file://, offline, không CDN):
- Chuột trái kéo = xoay · chuột phải / Shift+kéo = tinh chỉnh khung · con lăn = zoom.
- Phím `1`/`2` đổi cloud, `O` bật/tắt **chồng tham chiếu** (cloud chuyên nghiệp mờ),
  `R` đặt lại góc. Ảnh chụp tự kiểm tra 4 góc nằm trong `results/renders_raw/` và
  `results/renders_aligned/`.

## Chạy lại pipeline

```bash
# 1) Tải dữ liệu nặng (tham chiếu week26.ply/model.ifc + 5 video YouTube)
bash download_data.sh
# 2a) Chỉ dựng hình học CPU khi ĐÃ có cloud (artifacts/week{N}/fused*.ply): căn + render
PYTHON=/đường/dẫn/python3 bash reproduce.sh --weeks 26
# 2b) Dựng lại từ đầu trên Colab T4 (cần GPU): trích frame -> COLMAP đa shot -> merge -> viewer
PYTHON=/đường/dẫn/python3 bash reproduce.sh --weeks 26 --recon
```

`--recon` gọi COLMAP trên **Google Colab** qua CLI chính thức (`colab`); biến môi trường
`COLAB_CLI` trỏ tới binary đó. Đầu vào video nặng + đầu ra `fused.ply` đầy đủ nằm ngoài
git (xem `.gitignore`); chạy xong mới có cloud để căn/render.

## Pipeline (ánh xạ 1‑1 từ RAAMAC sang công cụ mã nguồn mở)

| Khâu RAAMAC (luận văn van Schaijk 2016) | Công cụ mở trong gói này | Script (`code/`) |
|---|---|---|
| Trích ảnh chuyến bay | ffmpeg/PyAV, 1080p, 5fps, lọc mờ | `extract_all_frames.py` → `extract_frames.py` |
| Video marketing ghép nhiều cảnh | phát hiện hard-cut bằng histogram | `detect_shots.py` |
| SfM họ SIFT + bundle adjustment | **COLMAP** (SIFT GPU; theo shot dùng sequential matching) | `remote_colmap.py`, `remote_colmap_shots.py` |
| MVS dày | COLMAP PatchMatch CUDA + `stereo_fusion` | (trong `remote_colmap*.py`) |
| Ghép nhiều góc | căn từng shot vào ref rồi **hợp nhất sau căn chỉnh** (tránh lệch scale) | `merge_shots.py` |
| Đăng ký hệ tham chiếu (quaternion Horn / ICP) | PCA coarse + ICP similarity | `align_and_compare.py` |
| Xem trực quan | viewer WebGL tự chứa + render matplotlib | `build_viewer_final.py`, `render_cloud.py` |

D4AR (công cụ gốc của RAAMAC) không công khai mã nguồn → **COLMAP là thay thế mở cùng họ
SfM SIFT**. Dựng nhiều video/đa góc là cách trung thực nhất để bắt kịp độ phủ của cloud
chuyên nghiệp từ một video nén, một quỹ đạo drone.

## Cấu trúc thư mục

```
raamac_repro/
  code/                 # toàn bộ pipeline + các script nhánh tái dùng (self-contained)
  data/                 # watermark_mask.png, cloud_attributes.npz (ref) + raw/ videos/ (download_data.sh)
  viewer/               # viewer_repro_week26.html (offline, có overlay tham chiếu)
  results/              # KẾT QUẢ HIỆN TẠI: .ply giảm mẫu, metrics, verify_summary, renders, shots manifest
  README.md             # file này
  reproduce.sh          # chạy lại (mặc định CPU; --recon cho GPU Colab)
  download_data.sh      # tải tham chiếu + video nguồn
```

## Giới hạn trung thực

- **Một quỹ đạo drone / video nén**: mặt khuất và nội thất không quay được → coverage
  thấp hơn cloud chuyên nghiệp nhiều góc. Đây là giới hạn **dữ liệu nguồn**, không phải
  lỗi pipeline; luận văn RAAMAC cũng thừa nhận không chụp được nội thất.
- **Khác không gian GUID**: `raamac_comparison.xlsx` của RAAMAC dùng IFC as-planned
  (export Synchro) có GlobalID khác `model.ifc` thiết kế → đối chiếu theo tên, xấp xỉ
  (`results/raamac_missing_target.json`: tuần 26=9, 27=50, 28=89, 29=18, 30=4 phần tử thiếu).
- **Không tune trên tập test đóng băng**; mọi cải thiện chỉ nhắm chất lượng dựng
  (số frame, matching, MVS, hợp nhất đa shot).
