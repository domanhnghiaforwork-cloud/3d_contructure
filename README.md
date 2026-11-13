# 3d_contructure — Pipeline tự động Video → 3D → căn chỉnh BIM → đánh giá RAAMAC

Pipeline **tự động một lệnh** đưa ảnh/video vào dựng point cloud 3D (trên GPU
Colab), căn chỉnh vào mô hình BIM (IFC), rồi đánh giá tiến độ thi công theo
phương pháp RAAMAC trên cohort nhãn độc lập đã đóng băng.

## Kết quả chính (test cohort đóng băng, 21 phần tử, không tune trên test)

| Nguồn cloud | Test acc | Built recall | Partial recall | Thắng baseline đa số (66.7%)? |
|---|---|---|---|---|
| Cloud chuyên nghiệp `week26.ply` | **20/21 = 95.24%** | 5/6 | 1/1 | **có** |
| Video → DUSt3R (khử nhiễu, 113 frame@384) | 14/21 = 66.7% | 0/6 | 0/1 | không (bằng) |
| Video → MASt3R metric (57 frame@256) | 14/21 = 66.7% | 0/6 | 0/1 | không (bằng) |

**Kết luận trung thực (đính chính):** mục tiêu ≥80% **chỉ đạt với cloud chuyên
nghiệp** `week26.ply` (95.24%, phát hiện đúng 5/6 phần tử built). Với cloud **chỉ
từ video**, cả DUSt3R lẫn MASt3R **không phát hiện đúng phần tử built nào (recall
0/6)** và partial 0/1. Accuracy 66.7% (cả DUSt3R khử nhiễu lẫn MASt3R) nghe có vẻ
"đạt 60%" nhưng thực chất
**bằng hoặc dưới baseline đoán `not_built` hàng loạt** (66.7%, vì 14/21 nhãn test
là not_built) — `beats_majority_baseline = false`. Nói cách khác, pipeline video
**chưa thực sự phát hiện tiến độ thi công**; nó chỉ đoán gần như toàn bộ là
not_built. Nút thắt là chất lượng dữ liệu nguồn (một quỹ đạo drone, nhà trắng ít
texture, video nén) khiến cloud dựng lại thiếu/rỗng ở các phần tử đã xây.

> **Đính chính so với bản trước:** bảng cũ ghi "50✓ 60✓" cho video là **gây hiểu
> lầm** — accuracy cao chỉ do lớp đa số not_built, không phải nhận diện tiến độ.
> Cloud video trong viewer cũ cũng bị **xoay/phóng sai** (ước lượng scale sai);
> đã căn lại (MASt3R rigid scale=1, overlap cloud chuyên nghiệp 78%; DUSt3R tìm
> kiếm đa scale, 45%). Xem `video_cloud_research/research_log.md` mục cuối.

## Viewer 3D (xem kết quả dựng lại)

Thư mục `viewer/` chứa viewer 3D **tự chứa** (không cần internet/CDN) để xem cloud
video dựng lại đối chiếu BIM:
- `viewer/index.html` — trang chọn kịch bản + bảng kết quả trung thực (per-class).
- `viewer/professional/` — **thư mục riêng** viewer cloud chuyên nghiệp:
  `viewer_3d.html` + `viewer_data.js` (week26.ply đã đăng ký IFC, test 95.24%, built 5/6).
- `viewer/dust3r.html` + `data_dust3r.js` — DUSt3R khử nhiễu (test 66.7%, built recall 0/6).
- `viewer/mast3r.html` + `data_mast3r.js` — MASt3R metric (test 66.7%, built recall 0/6).
- `viewer/results/` — summary + metrics JSON từng kịch bản (gồm `beats_majority_baseline`,
  `per_class`, `confusion_matrix`).

Mở `viewer/index.html` trong trình duyệt sau khi clone. Kéo chuột để xoay, lăn để
phóng to; bật/tắt lớp cloud video và BIM theo trạng thái (built/partial/not_built/
unobserved). Dữ liệu viewer sinh bởi `build_viewer_video.py` (cloud video khung ref
→ áp G3 ref→IFC → downsampling ~120k điểm + mẫu bề mặt BIM). Cloud video đã căn
chỉnh lại cho khớp IFC; tuy nhiên **lớp built trong viewer gần như rỗng** vì cloud
dựng lại từ video thiếu các phần tử đã xây — đây là hạn chế dữ liệu nguồn, không
phải lỗi hiển thị.

## Cấu trúc repo

```
.
├── progress_baseline.py / apply_semantic.py / evaluate_progress.py / semantic_groups.py
│       # pipeline đánh giá RAAMAC đóng băng (coverage plane + ngữ nghĩa v5)
├── prepare_schependomlaan.py        # tiện ích chuẩn bị dữ liệu
├── video_cloud_research/
│   ├── scripts/
│   │   ├── run_full_pipeline.py     # ★ MỘT LỆNH end-to-end
│   │   ├── run_dust3r_colab.py      # orchestrator Colab (new→upload→run→download→STOP)
│   │   ├── remote_dust3r.py / remote_mast3r.py   # dựng 3D trên GPU Colab
│   │   ├── extract_frames.py        # trích frame từ video (PyAV + độ nét)
│   │   ├── align_and_compare.py / align_to_bim.py / align_coverage.py  # căn chỉnh
│   │   ├── estimate_normals.py      # ước lượng pháp tuyến (PCA cục bộ)
│   │   ├── evaluate_raamac.py       # đánh giá cohort đóng băng
│   │   └── colmap_reconstruct.py / dense_mvs.py / ...  # nhánh COLMAP
│   ├── research_log.md / PLAN.md    # nhật ký + kế hoạch nghiên cứu
│   └── artifacts/                   # (sinh ra khi chạy: frames, clouds — đã .gitignore)
├── artifacts/                       # artifacts đóng băng NHẸ (đã commit)
│   ├── cohort_v1/                   # nhãn test/validation + config ngữ nghĩa v5
│   ├── week26_registration_v3/      # transform G3 ref→IFC + registration_inputs.npz
│   └── week26_raw_diagnostics/      # cloud_attributes.npz (cloud tham chiếu)
├── data/schependomlaan/raw/         # dữ liệu gốc NẶNG (tải riêng, xem dưới)
├── requirements.txt / setup_venv.sh / download_data.sh
```

## 1. Cài đặt môi trường cục bộ

```bash
bash setup_venv.sh          # tạo .venv + cài numpy/scipy/sklearn/trimesh/ifcopenshell/av/Pillow
source .venv/bin/activate
```

## 2. Tải dữ liệu NẶNG (không commit trong repo)

Repo chỉ chứa code + artifacts nhẹ (~15 MB). Các file nặng tải riêng:

```bash
bash download_data.sh
```

Script clone nguồn công khai chính thức
[`openBIMstandards/Archive-DataSetSchependomlaan`](https://github.com/openBIMstandards/Archive-DataSetSchependomlaan)
và đặt vào `data/schependomlaan/raw/`:
- `model.ifc` — mô hình BIM (~49 MB).
- `week26.ply` — cloud chuyên nghiệp (~74 MB), dùng làm cloud tham chiếu / nhánh đánh giá chuyên nghiệp.
- Video drone gốc (cùng dataset) — dùng trích frame.

> Nếu tên file trong dataset khác, đổi thủ công về `model.ifc` / `week26.ply`.
> Kiểm tra lại nguồn trước khi tải; chỉ dùng link chính thức ở trên.

**Trọng số mô hình KHÔNG cần tải thủ công:** `remote_dust3r.py` / `remote_mast3r.py`
tự tải từ Hugging Face trên Colab khi chạy:
- DUSt3R: [`naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`](https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt) (~1.3 GB).
- MASt3R metric: [`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric) (~2.75 GB).
- Code dựng: [`github.com/naver/dust3r`](https://github.com/naver/dust3r), [`github.com/naver/mast3r`](https://github.com/naver/mast3r) (tự clone trên Colab).

### Tái tạo đăng ký week26.ply → IFC (G3, mỏ neo chuyên nghiệp)

Toàn bộ code + artifact nhẹ của bước đăng ký đã commit (chỉ raw data tải riêng):
- `register_cloud.py` — sinh G3: ICP trimmed đa khởi tạo (scale / yaw / z-offset)
  giữa mẫu cloud chuyên nghiệp và mẫu bề mặt IFC → `candidate_transform.json`.
- `verify_registration.py`, `test_registration.py`, `inspect_alignment.py` — kiểm chứng đăng ký.
- `artifacts/week26_registration_v3/` — G3 đóng băng + `registration_inputs.npz`
  (source/target) + `registration.json` (ghi ĐÚNG tham số đã dùng: scales, flip_z,
  source_bounds, normal_frame, yaw_steps, z_offsets) + `candidate_alignment.png`.
- `refine_registration.py` — ICP point-to-plane tinh chỉnh thêm (lợi ích không đáng
  kể ~0.8°, xem research_log); kết quả ở `artifacts/week26_registration_refined/`.

Tham số tái tạo G3 đọc từ `registration.json`; chạy `register_cloud.py --samples
<inspection_samples.npz> --output <dir>` với các cờ tương ứng. Cloud chuyên nghiệp
khung ref (`cloud_attributes.npz`) + G3 là mỏ neo để căn cloud video
(`clean_align_dust3r.py`, `align_and_compare.py`).

## 3. Chuẩn bị frames từ video

```bash
python video_cloud_research/scripts/extract_frames.py \
  --video data/schependomlaan/raw/<video_drone>.mp4 \
  --out video_cloud_research/artifacts/frames_4fps --fps 4
# đóng gói để upload Colab (orchestrator tự chia chunk >60MB):
cd video_cloud_research/artifacts && zip -r frames_4fps.zip frames_4fps && cd ../..
```

## 4. Cài Colab CLI (cho khâu dựng GPU)

Khâu dựng chạy trên Google Colab (T4). Cài CLI chính thức `google-colab-cli`
(cần Python ≥3.12, dùng `uv`):

```bash
uv tool install google-colab-cli      # binary tại ~/.local/bin/colab
colab --auth=oauth2 sessions          # xác thực OAuth lần đầu (mở URL consent trong browser)
```

Nếu `colab` không ở PATH, trỏ rõ: `export COLAB_BIN=/đường/dẫn/colab`.
Orchestrator tự tìm qua `COLAB_BIN` → PATH → `~/.local/bin/colab`.

## 5. Chạy pipeline MỘT LỆNH (end-to-end từ video)

```bash
python video_cloud_research/scripts/run_full_pipeline.py \
  --backend dust3r --graph logwin --niter 1000 --conf 1.5 --stride 2 --size 384 \
  --out-dir video_cloud_research/artifacts/run1
```

Tự động: new T4 → upload frames → dựng DUSt3R → tải `fused.ply` → **STOP session**
→ căn chỉnh vào cloud tham chiếu (DUSt3R `--scale-search`, MASt3R `--rigid`) →
ước lượng normals → đánh giá RAAMAC (7 types đóng băng) → `pipeline_report.json`
+ báo cáo trung thực per-class (built/partial recall) và `beats_majority_baseline`
(accuracy có thắng baseline đoán not_built hàng loạt hay không).

**Backend MASt3R metric (cloud metric, scale thật — căn rigid scale=1):**
```bash
python video_cloud_research/scripts/run_full_pipeline.py \
  --backend mast3r --size 256 --stride 4 --conf 1.0 \
  --out-dir video_cloud_research/artifacts/run_mast3r
```

**Chạy lại phần cục bộ không tốn GPU** (đã có `fused.ply` trong out-dir):
```bash
python video_cloud_research/scripts/run_full_pipeline.py \
  --backend dust3r --skip-remote --out-dir video_cloud_research/artifacts/run1
```

Tham số: `--backend {dust3r,mast3r}`, `--graph`, `--niter`, `--conf`, `--stride`,
`--size`, `--session`, `--skip-remote`.

## 6. Tài nguyên mỗi lần chạy (Colab T4)

- GPU 16 GB VRAM / RAM ~12.7 GB. DUSt3R 113 frame@384px hoặc MASt3R 57 frame@256px vừa RAM.
- Đĩa VM tạm thời: weights (1.3–2.75 GB) + frames (~138 MB) + cloud output (~144 MB).
- Orchestrator **tự STOP session** khi xong hoặc lỗi (tiết kiệm compute units).

## Ghi chú

- **Cohort đóng băng:** `artifacts/cohort_v1/test_labels_frozen.csv` (21) +
  `validation_labels.csv` (44). Không tune trên test; giữ đủ phần tử (unobserved
  tính sai), không loại dự đoán thiếu để thổi accuracy.
- **Khung tọa độ:** `progress_baseline` áp transform G3 (ref→IFC) lên cloud +
  normals; đầu vào căn chỉnh ở khung ref (`alignment_ref`).
- **Giấy phép:** DUSt3R/MASt3R (naver) CC BY-NC-SA — nghiên cứu phi thương mại.
  Dataset Schependomlaan công khai. Kiểm tra giấy phép từng nguồn trước khi dùng.
- **URL:** chỉ dùng nguồn chính thức (Google Colab, github.com/naver,
  github.com/openBIMstandards, huggingface.co/naver); không dùng proxy tạm.