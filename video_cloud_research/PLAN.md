# Nhánh nghiên cứu: Dựng point cloud từ video drone và đánh giá RAAMAC

Ngày bắt đầu: 2026-11-12. Thư mục này độc lập với pipeline chính (đã đóng băng, test 95.24%).

## Mục tiêu

1. Tự dựng point cloud từ video drone week26 (SfM + MVS bằng COLMAP, Colab T4).
2. Đánh giá chất lượng cloud dựng lại so với cloud gốc `data/schependomlaan/raw/week26.ply`
   (Chamfer distance, coverage hai chiều, mật độ) sau khi căn chỉnh similarity.
3. Dùng cloud dựng lại thay cloud gốc trong pipeline đánh giá tiến độ RAAMAC
   (coverage v3 + config ngữ nghĩa v5 ĐÃ ĐÓNG BĂNG) trên test cohort ĐÃ ĐÓNG BĂNG.
4. Chinh phục lần lượt các mốc accuracy: **50% → 60% → 70% → 80%**.

## Ràng buộc (kế thừa từ nghiên cứu chính)

- KHÔNG tune trên test đóng băng (`artifacts/cohort_v1/test_labels_frozen.csv`,
  sha 6cdde870…). Mọi tinh chỉnh chỉ trên validation.
- Config ngữ nghĩa v5 đóng băng (`artifacts/validation_semantic_config_v5.json`) giữ nguyên
  để so sánh công bằng giữa cloud gốc và cloud dựng lại.
- 51 phần tử lỗi hình học vẫn là `unobserved` (tính là sai khi đánh giá).
- Không dùng nhãn do RAAMAC sinh ra làm chân lý độc lập.
- Chỉ tài nguyên miễn phí (Colab T4), không dừng runtime có sẵn của người dùng.
- Nhật ký tiếng Việt trong `research_log.md` của thư mục này.

## Chuỗi xử lý dự kiến

```
data/schependomlaan/videos/week26.mp4 (55 s, 25 fps, 1920x1080)
  → scripts/extract_frames.py (PyAV, ~2 fps, lọc mờ)      [local]
  → scripts/colmap_reconstruct.py                          [Colab T4]
      feature_extractor → sequential_matcher → mapper
      → image_undistorter → patch_match_stereo → stereo_fusion
      → artifacts/colmap_run/fused.ply (+ sparse cloud, cameras)
  → scripts/align_and_compare.py                           [local]
      similarity alignment (PCA thô + ICP có scale) vào hệ week26.ply
      → transform riêng; đo Chamfer/coverage/density
  → scripts/evaluate_raamac.py                             [local]
      cloud dựng lại (đã căn chỉnh) × G3 transform → hệ IFC
      → coverage v3 (plane mode, 2000 mẫu/phần tử, seed 42)
      → apply config v5 đóng băng → so khớp test đóng băng
      → accuracy + macro-F1 + per-class, đối chiếu mốc 50/60/70/80
```

## Tiêu chí chất lượng cloud (tham khảo, không phải mục tiêu cứng)

- Coverage chiều xuôi (điểm cloud gốc có láng giềng cloud dựng lại trong 0.1 m) ≥ 60%
  được kỳ vọng là đủ để pipeline phân loại hoạt động; sẽ kiểm chứng thực nghiệm.
- Chamfer trung bình ≤ 0.1 m sau căn chỉnh.

## Phương án lặp lại nếu chưa đạt mốc

1. Tăng mật độ frame (2 → 3–4 fps), thêm video góc khác nếu cần (chỉ week26 để đúng tuần).
2. Tinh chỉnh tham số MVS (max_image_size, geom_consistency, filter).
3. Lọc nhiễu cloud dựng lại (outlier removal, statistical filter).
4. Nếu coverage quá thưa: chỉ báo cáo trung thực mốc đạt được + phân tích nguyên nhân.