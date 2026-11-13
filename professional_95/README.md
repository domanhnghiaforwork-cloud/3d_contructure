# professional_95 — Nhánh map cloud CHUYÊN NGHIỆP week26.ply → IFC (RAAMAC 95.24%)

Package **độc lập, tự chứa** của nhánh đạt mục tiêu: dùng cloud laser chuyên nghiệp
`week26.ply` (bộ Schependomlaan công khai) đăng ký vào mô hình BIM `model.ifc`, rồi
đánh giá tiến độ thi công theo RAAMAC trên cohort nhãn độc lập **đã đóng băng**.

**Kết quả (test đóng băng, 21 phần tử, không tune trên test):**
`20/21 = 95.24%` · built recall **5/6** · partial recall **1/1** ·
`beats_majority_baseline = true`. Đây là nhánh DUY NHẤT thực sự phát hiện tiến độ
(hai nhánh video DUSt3R/MASt3R chỉ bằng baseline đoán not_built, built recall 0/6 —
xem repo chính / research_log).

## Cấu trúc

```
professional_95/
├── code/                  # toàn bộ code nhánh chuyên nghiệp
│   ├── register_cloud.py          # sinh G3: đăng ký week26.ply → IFC (ICP trimmed đa khởi tạo)
│   ├── verify_registration.py / test_registration.py / inspect_alignment.py / inspect_cloud.py
│   ├── refine_registration.py     # ICP point-to-plane tinh chỉnh thêm (lợi ích ~0.8°, không đáng kể)
│   ├── progress_baseline.py       # coverage v3 (plane, 0.1/0.2, 2000 mẫu, seed 42)
│   ├── apply_semantic.py          # ngữ nghĩa v5 đóng băng
│   ├── evaluate_progress.py       # đối chiếu cohort → accuracy/per-class/confusion
│   ├── semantic_groups.py / define_cohort.py / reproduce_v5.py
│   ├── prepare_schependomlaan.py  # chuẩn bị dữ liệu
│   └── build_viewer_3d.py         # dựng viewer_data.js (viewer đã dựng sẵn kèm theo)
├── data/
│   ├── raw/                       # week26.ply (74MB) + model.ifc (49MB)  [NẶNG]
│   ├── week26_alignment/inspection_samples.npz   # mẫu source/target cho register_cloud
│   ├── week26_raw_diagnostics/cloud_attributes.npz  # cloud chuyên nghiệp khung ref + normals + colors
│   ├── week26_registration_v3/    # G3 đóng băng + registration_inputs.npz + registration.json + png
│   ├── week26_registration_refined/  # transform tinh chỉnh + metrics
│   └── cohort_v1/                 # nhãn test/validation đóng băng + config ngữ nghĩa v5 + metrics
├── viewer/                  # FILE DỰNG SẴN: mở là chạy, không cần internet
│   ├── viewer_3d.html
│   └── viewer_data.js
├── results/                 # test_metrics_v5.json (95.24%) + repro_v5_test_metrics.json
├── reproduce.sh             # chạy lại đánh giá từ data có sẵn
├── download_data.sh         # tải raw nặng khi clone từ git (raw bị gitignore)
└── README.md
```

## Xem kết quả 3D

Mở `viewer/viewer_3d.html` trong trình duyệt (tự chứa, không CDN/server). Kéo chuột
xoay, lăn phóng to; bật/tắt lớp cloud chuyên nghiệp và bề mặt IFC theo trạng thái
(built / partial / not_built / unobserved); click cấu kiện để xem coverage + nhãn.

## Chạy lại đánh giá (từ data có sẵn)

```bash
bash reproduce.sh          # progress_baseline → apply_semantic → evaluate_progress (test+validation)
```

Hoặc thủ công (đường dẫn tương đối trong package):

```bash
python code/progress_baseline.py --ifc data/raw/model.ifc \
  --cloud data/raw/week26.ply --attributes data/week26_raw_diagnostics/cloud_attributes.npz \
  --transform data/week26_registration_v3/candidate_transform.json --scan-id week26 \
  --types IfcWall IfcWallStandardCase IfcSlab IfcRoof IfcBeam IfcColumn IfcCovering \
  --output out/baseline.csv --samples 2000 --distance 0.1 --lateral 0.2 --seed 42
python code/apply_semantic.py --predictions out/baseline.csv \
  --config data/cohort_v1/validation_semantic_config_v5.json --ifc data/raw/model.ifc \
  --attributes data/week26_raw_diagnostics/cloud_attributes.npz \
  --transform data/week26_registration_v3/candidate_transform.json --output out/predictions.csv
python code/evaluate_progress.py --labels data/cohort_v1/test_labels_frozen.csv \
  --predictions out/predictions.csv --split test --output out/metrics_test.json
```

Tái tạo đăng ký G3 (nếu muốn chạy lại từ đầu): `code/register_cloud.py --samples
data/week26_alignment/inspection_samples.npz --output <dir>` với tham số ghi trong
`data/week26_registration_v3/registration.json` (scales, flip_z, source_bounds,
normal_frame, yaw_steps, z_offsets).

## Dữ liệu nặng khi clone từ git

`data/raw/week26.ply` + `data/raw/model.ifc` có sẵn khi làm việc cục bộ, nhưng bị
`.gitignore` trong repo (giữ repo nhẹ). Sau khi clone, chạy:

```bash
bash download_data.sh      # clone nguồn công khai openBIMstandards/Archive-DataSetSchependomlaan
```

## Ghi chú trung thực

- Cohort đóng băng: `data/cohort_v1/test_labels_frozen.csv` (21) + `validation_labels.csv` (44).
  Không tune trên test; giữ đủ phần tử (unobserved tính sai), không loại dự đoán thiếu.
- Khung tọa độ: `progress_baseline` áp G3 (ref→IFC) lên cloud + normals; đầu vào ở khung ref.
- Giấy phép: dataset Schependomlaan công khai; kiểm tra giấy phép từng nguồn trước khi dùng.
- URL: chỉ nguồn chính thức (github.com/openBIMstandards); không proxy tạm.