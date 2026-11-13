# 3d_contructure — Pipeline tự động Video → 3D → căn chỉnh BIM → đánh giá RAAMAC

Pipeline **tự động một lệnh** đưa ảnh/video vào dựng point cloud 3D (trên GPU
Colab), căn chỉnh vào mô hình BIM (IFC), rồi đánh giá tiến độ thi công theo
phương pháp RAAMAC trên cohort nhãn độc lập đã đóng băng.

## Kết quả chính (test cohort đóng băng, 21 phần tử, không tune trên test)

| Nguồn cloud | Test accuracy | Mốc 50/60/70/80% |
|---|---|---|
| Cloud chuyên nghiệp `week26.ply` | **20/21 = 95.24%** | đạt cả 4 |
| Video → DUSt3R (logwin, 113 frame@384) | 13/21 = 61.9% | 50✓ 60✓ 70✗ 80✗ |
| Video → MASt3R metric (57 frame@256) | 14/21 = 66.7% | 50✓ 60✓ 70✗ 80✗ |

**Kết luận trung thực:** mục tiêu ≥80% đạt với cloud chuyên nghiệp. Với cloud
**chỉ từ video** (1 quỹ đạo drone nén, nhà trắng ít texture), trần ~66.7% — đạt
50/60%, chưa 70/80%. Nút thắt là chất lượng dữ liệu nguồn (góc nhìn hạn chế,
ít texture, video nén), không phải pipeline đánh giá. Xem
`video_cloud_research/research_log.md` để biết chi tiết thí nghiệm.

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
→ căn chỉnh vào cloud tham chiếu → ước lượng normals → đánh giá RAAMAC (7 types
đóng băng) → `pipeline_report.json` + bảng mốc 50/60/70/80.

**Backend MASt3R metric (kết quả video tốt nhất ~66.7%):**
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