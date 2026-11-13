# Nhật ký nghiên cứu: Dựng point cloud từ video → đánh giá RAAMAC

Nhánh riêng của dự án, bắt đầu 2026-11-12 theo yêu cầu của người dùng:
tạo thư mục nghiên cứu riêng cho phần dựng cloud từ video, đánh giá với cloud
có sẵn, và dùng cloud dựng lại để đánh giá RAAMAC theo các mốc 50/60/70/80%.

## 2026-11-12 — Khởi tạo

- Tạo `video_cloud_research/` (scripts, artifacts), viết PLAN.md.
- Kiểm tra dữ liệu: `data/schependomlaan/videos/week26.mp4` — video
  "Nieuwbouwproject De Schepen Week 26" (Dreamfocus Droned), 55 s, 25 fps,
  1920×1080. Các video còn lại là week 27/28/29/30 (không dùng cho cloud tuần 26).
- Bộ frame cũ `artifacts/week26_frames/` chỉ có 14 frame thưa (mỗi 4 s) dùng để
  gán nhãn — KHÔNG đủ cho SfM; cần trích dày.
- Môi trường local: venv uv quản lý, có PyAV 19.0.2 (không cần ffmpeg hệ thống),
  numpy/scipy/sklearn/trimesh/ifcopenshell đầy đủ. Colab CLI đã xác thực,
  không có session đang chạy.
- Kế hoạch: trích ~2 fps (≈110 frame) → COLMAP (SIFT GPU + sequential matcher
  + mapper + PatchMatch MVS + fusion) trên Colab T4 → căn chỉnh similarity vào
  hệ week26.ply → đo Chamfer/coverage → thay vào pipeline coverage v3 + config
  v5 đóng băng → đánh giá trên test đóng băng (21 phần tử).
- Ràng buộc giữ nguyên: không tune trên test, config v5 đóng băng, 51 phần tử
  unobserved giữ nguyên, nhật ký trung thực.
## 2026-11-12 — Dựng cloud COLMAP trên Colab T4 (lần 1)

- Trích 115 frame @2 fps từ week26.mp4; loại 2 frame mờ đầu/cuối (fade),
  còn 113 frame (sharpness trung vị 937). Đóng gói frames_2fps.zip (69 MB).
- Phát hiện watermark tĩnh "DREAMFOCUS" góc phải trên → tạo mặt nạ
  watermark_mask.png (đen vùng 1640..1920 × 0..160) để COLMAP bỏ qua,
  tránh khớp giả giữa các frame.
- Session Colab `vcr` (T4). Upload zip + mặt nạ (CLI upload cần đường dẫn
  tuyệt đối và thư mục đích tồn tại trước; tạo bằng exec bootstrap).
- Sự cố 1: `feature_extractor --SiftExtraction.use_gpu 1` SIGABRT —
  stack trace Qt `createPlatformIntegration`: COLMAP 3.7 (apt) cần nền tảng
  Qt GUI cho SIFT GPU; môi trường headless không có. Khắc phục:
  `QT_QPA_PLATFORM=offscreen` + fallback CPU.
- Sự cố 2: CLI `exec` mặc định ngắt khi kernel im lặng 30 s → dùng
  `--timeout 0` cho các lệnh dài.
- Sự cố 3: kể cả với offscreen, SIFT GPU (OpenGL) treo không hồi kết ở
  headless → restart kernel (session của mình), chuyển SIFT extraction +
  matching sang CPU (probe xác nhận CPU chạy tốt, ~5 s/frame);
  PatchMatch MVS vẫn dùng CUDA GPU.
- Script colmap_reconstruct.py resume theo marker: .features_done,
  .matched, sparse/0, dense cfg, depth maps, fused.ply.

## 2026-11-12 — Kiểm chứng script căn chỉnh (align_and_compare.py)

- Tự kiểm tra bằng dữ liệu tổng hợp: cloud giả = cloud gốc × similarity biết
  trước (scale 0.37, quay 40°/15°, tịnh tiến) + nhiễu 0.01 m.
- Ba lỗi tìm ra và sửa:
  1. PCA thô không ước lượng scale → thêm scale khởi tạo từ tỉ số eigenvalue.
  2. ICP trim mạnh làm co scale → lịch trình: 15 vòng đầu khóa scale (rigid),
     sau đó thả scale với trim 90%.
  3. Lỗi then chốt: `np.linalg.eigh` có thể trả hệ trục trái (det −1) khiến
     phép quay đúng rơi ngoài bộ lọc det>0 của 24 ứng viên → ép hệ PCA phải
     thuận (det +1) bằng cách đổi dấu trục thứ ba.
- Kết quả sau sửa: scale khôi phục 2.7025 (chân lý 2.7027), NN mean 0.028 m
  (đúng mức nhiễu), coverage_recon@0.1 m ≈ 1.0. Script đạt yêu cầu kiểm chứng.
- COLMAP: feature extraction CPU 428 s cho 113 frame; exhaustive matcher
  (6328 cặp) chạy CPU sau khi GPU fail; các bước sau chờ kernel.

## 2026-11-12 15:10 — Chuyển exhaustive → sequential matcher

- Exhaustive matcher CPU chạy >2 giờ (6328 cặp, ~10k feature/ảnh, VM ít core)
  vượt ngưỡng chờ đặt trước → restart kernel session của mình, giữ nguyên
  database đã trích xuất feature (.features_done).
- Thay bằng sequential_matcher overlap 10 + quadratic_overlap (≈1.500–2.000
  cặp, tận dụng thứ tự video bay vòng; quadratic tạo cặp xa để khép vòng).
  Các cặp exhaustive đã khớp dở trong database vẫn được giữ, bổ sung liên kết.
- Thêm heartbeat: run() ghi stdout/stderr của COLMAP vào last_cmd.log và in
  dòng cuối mỗi 30 s → thấy tiến độ từng bước, không còn chờ mù.

- Sequential matcher + quadratic_overlap: 90 s (so với >2 h exhaustive).
- Mapper lỗi vặt `output_path is not a directory` → thêm mkdir sparse/dense;
  chạy lại từ mapper (feature + match giữ nguyên nhờ marker).

## 2026-11-12 16:20 — Vướng CUDA cho MVS, hướng xử lý

- Mapper 51 phút ra 3 model (58/52/22 ảnh, tổng registrations 132 > 113 →
  có ảnh chung, merge được). Match graph gần đủ exhaustive: 6322 cặp,
  1268 cặp >100 inliers.
- `patch_match_stereo` bản apt: "requires CUDA, which is not available" —
  bản apt không build CUDA. pycolmap 4.2.1 (pip) cũng has_cuda=False.
- Hướng chọn: conda-forge colmap có build CUDA (xác minh qua api.anaconda.org)
  → cài miniforge + `conda install colmap cuda-version=12.6` trên Colab,
  chạy PatchMatch + fusion bằng binary đó (script dense_mvs.py).
- model_merger cần mkdir thư mục đầu ra trước (lỗi giống mapper); đang merge
  2+0 rồi +1 thành model hợp nhất để undistort + MVS.

- PatchMatch CUDA (conda colmap) chạy 480 s cho model 22 ảnh — nhưng phát hiện
  merge lần 2 (merged02+model1) phá kết quả: chỉ còn 22 ảnh. merged02 mới là
  bản đúng: 101/113 ảnh (89%), phủ frame 1..113.
- Chọn merged02 làm model cuối; workspace mới dense_final; fusion dùng option
  mới `--StereoFusion.min_num_pixels/max_reproj_error` (bản conda đổi tên).
- Chạy lại dense_mvs.py: undistort → PatchMatch (101 ảnh) → fusion → fused.ply.

## 2026-11-12 17:15 — fused.ply về local, căn chỉnh lần đầu thất bại

- Tải fused.ply (85 MB, 3.16M điểm) từ Colab. PatchMatch CUDA 2310 s cho
  101 ảnh, fusion 120 s.
- Căn chỉnh lần 1 hỏng: cloud dựng lại chứa outlier MVS xa tới −619 m và
  "halo" cây thưa tới 60 m → PCA/ICP lệch (Chamfer 1.17–1.49 m).
- Sửa: (1) clean_cloud — cắt bán kính 80 m quanh median + lọc mật độ kNN
  (giữ md < 3×median); (2) căn chỉnh chỉ trên lõi mật độ cao (dense_core,
  quantile 0.5) của cả hai cloud vì tòa nhà là vùng đặc, cây/nhiễu thưa;
  transform cuối áp cho toàn cloud sạch.
- Đang chạy lại bản căn chỉnh lõi (nặng kNN, chạy nền).

## 2026-11-12 17:50 — Nghi merge lệch, chuyển MVS theo từng model

- Căn chỉnh vẫn lệch kể cả khi thêm hệ trục từ pháp tuyến trội (roof/walls):
  coarse 0.407 m, Chamfer 1.7 m; ảnh chồng cho thấy cloud có "đuôi" thẳng
  đứng — nghi merged02 ghép hai khối lệch nhau (chỉ 9 ảnh chung).
- Sửa lỗi đọc PLY: trimesh load_ply trả key `vertex_normals`/`vertex_colors`
  (không phải `normals`/`red`) — trước đó pháp tuyến bị bỏ sót.
- Quyết định: chạy MVS riêng cho model 2 (58 ảnh, frame 22–113) và model 0
  (52 ảnh, frame 1–77); căn chỉnh từng cloud riêng vào hệ ref; nếu mỗi cloud
  tự nhất quán (Chamfer nhỏ) thì hợp nhất hai cloud ĐÃ căn chỉnh để đánh giá.
- dense_mvs.py viết lại sạch, nhận --models; chạy nền trên T4.

## 2026-11-12 18:10 — Mất session Colab, T4 tạm hết chỗ

- Session vcr bị thu hồi (free tier idle) — mất toàn bộ state từ xa
  (database, sparse models, dense, conda). fused.ply (merged02) đã kịp tải về.
- `new -s vcr2 --gpu T4` báo Service Unavailable (hết capacity tạm thời).
- Trong lúc chờ: thử căn chỉnh robust đa khởi tạo (24 hoán vị trục, ICP trim
  0.6, chấm điểm theo tỉ lệ inlier 0.15 m) trên cloud merged02 hiện có —
  nếu khối tốt chiếm đa số, có thể tách phần dựng đúng để đánh giá trước.
- Kế hoạch khi có T4: chạy lại SFM (features 7' + match 1.5' + mapper ~51'),
  tải sparse models về ngay làm backup, rồi MVS riêng model 0/2.

- Đa khởi tạo 24×12 ICP rigid: inlier tốt nhất 0.48 @0.15 m → xác nhận cloud
  ghép gồm khối đúng + khối lệch. Lọc giữ khối đúng (d≤0.3/0.5 m) chỉ phủ
  2.6% cloud gốc → không đủ đánh giá; phương án tạm bị loại.
- Thử nghiệm v3 đang chạy: hệ trục pháp tuyến + 24 ứng viên × 30 vòng
  similarity ICP trim giảm dần 0.8→0.5, chấm inlier 0.15 m.
- Monitor T4: retry 1 vẫn unavailable.

- v3 (hệ pháp tuyến + 30 vòng similarity ICP): inlier 0.57 nhưng coverage_ref
  chỉ 3.5% — phần khớp là mặt đất (phẳng, tự đồng dạng), không phải tòa nhà.
- v4 (chỉ tường): inlier 0.76 nhưng scale sập 0.31 — suy biến; ảnh chồng xác
  nhận cloud đỏ co nhỏ dính góc tòa nhà xanh.
- Giả thuyết củng cố: merged02 chứa hai khối ở HAI SCALE khác nhau (merger
  ước lượng sai similarity giữa chừng) → không một transform toàn cục nào
  căn được cả hai.
- v5 đang chạy: căn chỉnh 2 tầng — khối A = inlier của transform tốt nhất;
  khối B = phần dư, căn riêng bằng đa khởi tạo; hợp nhất A+B đã căn chỉnh.

- v5 (2 tầng A/B): coverage_ref@0.1 = 10.3%; ảnh chồng cho thấy union chỉ phủ
  một mảng nhỏ tường phải + mép mái → merged02 không cứu được locally.
- v6 đang chạy: căn thẳng vào BIM (target sạch, 50k mẫu) với lưới scale
  0.6–2.2 × 24 hoán vị × 20 ICP, chấm inlier 0.15 m — phép thử quyết định
  xem khối nhà trong cloud dựng lại có tự nhất quán hay không.

## 2026-11-12 18:40 — Kết luận chất lượng SfM lần 1 + chuẩn bị lần 2

- v6 (căn thẳng vào BIM, lưới scale 0.6–2.2 × 24): inlier tốt nhất 0.456,
  scale sập ~0.5 → tòa nhà dựng lại bị BIẾN DẠNG, không phải similarity của
  nhà thật. Nguyên nhân nghi: frame thưa 2 fps + tường trắng lặp + drift.
- Hệ quả: MVS theo khối cũng không cứu được pose sai → phải chạy lại SfM
  với frame dày hơn.
- Đã trích 225 frame @4 fps (lọc mờ ≥100, tất cả đạt), zip 138 MB sẵn sàng.
- colmap_reconstruct.py: nhận VCR_ZIP (mặc định frames_4fps.zip), overlap 15,
  BA global 25 vòng để giảm thời gian mapper.
- Chờ T4 (monitor còn 3 lần thử). Khi có máy: upload zip mới → SFM → tải
  sparse backup → MVS → căn chỉnh → đánh giá mốc.

## 2026-11-12 18:50 — T4 hết chỗ, chuyển SfM sang Colab CPU

- T4 retry 1–5 đều unavailable. Nhận ra SfM không cần GPU → tạo session
  CPU `vcrcpu` (READY ngay).
- Upload zip 138 MB bị 500 (giới hạn contents API) → chia 3 chunk 60 MB,
  upload rồi `cat` ghép từ xa; kiểm tra kích thước khớp 138627934 byte.
- Chạy colmap_reconstruct.py với VCR_SFM_ONLY=1 trên CPU: features 225 ảnh
  + sequential overlap 15 + mapper (BA 25). Dự kiến 1–2.5 h, chạy nền.
- Khi xong: tải sparse models về local ngay (backup), kiểm tra nhất quán
  (căn thử vào BIM); MVS sẽ chạy trên T4 khi có chỗ (monitor còn 1 lần thử).

## 2026-11-12 20:00 — Chiến lược không phụ thuộc GPU

- T4 unavailable liên tục ~2 h; hai monitor hết lượt.
- Matcher 4 fps chậm (~75 phút cho 225 ảnh); mapper còn dài.
- Quyết định: khi SfM xong, đánh giá RAAMAC NGAY bằng sparse cloud (không
  cần MVS/GPU) nếu mật độ đủ; MVS chỉ là phương án tăng mật độ khi cần.
  Điều này gỡ nút thắt GPU cho các mốc 50–80%.

## 2026-11-12 22:15 — Người dùng mua đơn vị tính toán Colab → tăng tốc

- Người dùng xác nhận đã mua compute units → cho phép dùng GPU trả phí.
- T4 phân bổ được ngay (vcrfast) nhưng chỉ 2 vCPU — không tăng tốc mapper.
- Mapper CPU mới đăng ký 17/225 ảnh sau ~1 h → ngoại suy >10 h: không chấp nhận.
- Quyết định: cấp phát A100 (12 vCPU + GPU mạnh) chạy lại toàn bộ SfM+MVS;
  dừng session CPU khi A100 sẵn sàng để tiết kiệm units.
- Minh bạch chi phí: A100 tốn units/giờ cao hơn T4 nhiều lần; người dùng đã
  chủ động mua units để đổi tốc độ.

## 2026-11-12 22:40 — Mục tiêu lớn: pipeline tự động end-to-end ≥80%

- Người dùng giao mục tiêu tối thượng: pipeline TỰ ĐỘNG video → dựng 3D →
  căn chỉnh → đánh giá BIM đạt RAAMAC ≥80%; dùng GPU Colab hợp lý, tận dụng
  tối đa mỗi lần chạy, TẮT session khi không dùng; mọi bộ dữ liệu/phương pháp.
- Đã dừng vcrcpu (mapper 17/225 sau 1 h, ngoại suy >10 h) và vcrfast (idle)
  để tiết kiệm units. A100 hai lần cấp phát thất bại/treo → bỏ.
- Chọn phương pháp mới cho khâu dựng: **DUSt3R** (naver, CVPR24) — dựng 3D
  dense bằng mạng học sâu, chạy hoàn toàn trên GPU, không cần mapper CPU
  (nút thắt lớn nhất), robust với tường trắng lặp. Giấy phép CC BY-NC-SA
  (nghiên cứu phi thương mại — phù hợp bối cảnh).
- T4 trả phí cấp phát tức thì (vcrd). Upload 225 frame @4 fps (3 chunk).
- remote_dust3r.py: clone dust3r, weights ViT-L16_512, scene graph swin-2,
  global alignment 300 vòng, lọc confidence >1.5, xuất fused_dust3r.ply.
- Orchestrator auto_pipeline.py + align_to_bim.py (cổng inlier ≥0.7) +
  evaluate_raamac.py đã viết: chuỗi tự động new→upload→run→download→STOP.
- Kế hoạch: DUSt3R xong → tải cloud → align_to_bim (gate) → đánh giá cohort
  đóng băng → mốc 50/60/70/80. Nếu gate trượt: tinh chỉnh (conf, resolution,
  scene graph) hoặc COLMAP trên A100 khi có.

## 2026-11-12 23:28 — DUSt3R THÀNH CÔNG, session đã tắt

- Sau các lỗi setup (croco submodule, roma, torch 2.6 weights_only, URL
  checkpoint hỏng → HF repo đúng naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt,
  OOM RAM → 384px + 113 frame, lỗi numpy/tensor khi xuất, thiếu thư mục cache):
  DUSt3R hoàn tất trong 289 s trên T4: 9.02M điểm, loss global alignment 0.011.
- Tải fused.ply (144 MB) về artifacts/dust3r_run/; STOP session vcrd ngay
  (đúng chính sách tắt khi không dùng).
- Scale DUSt3R tùy ý (nhà ~2.7 unit) → align_to_bim.py nâng cấp: scale nền
  từ tỉ lệ bán kính p90 × lưới 0.7–1.4, xoay thô PCA 24 hoán vị.
- Đang chạy align_to_bim (cổng inlier ≥0.7); đạt → evaluate_raamac trên
  cohort đóng băng.

- Cổng inlier thất bại (0.287): phân tích overlay cho thấy cloud DUSt3R đúng
  hình nhà nhưng ICP hội tụ nghiệm co nhỏ; tìm scale theo coverage BIM đạt
  0.15 @0.2; rigid ICP không cải thiện; tìm kiếm gương (48 ứng viên) cũng
  không → nghi biến dạng nội tại của DUSt3R (swin-2, 384px, 113 frame).
- Đang thử căn theo toàn cảnh cloud gốc (cây+đất+nhà) để khóa hướng;
  phương án dự phòng: DUSt3R complete-graph ~50 frame @512px.

## 2026-11-13 — Sửa cổng căn chỉnh + KẾT QUẢ RAAMAC trung thực đầu tiên từ video

- Phát hiện 2 lỗi trong align_to_bim.py: (1) clip scale từng bước không chặn
  tích lũy → ICP co cloud về ~0.0135 (nghiệm suy biến) nhưng inlier≈1 giả;
  (2) inlier_core dùng điểm cũ từ lưới, không tính lại sau refine. Đã sửa:
  chặn scale tích lũy quanh --scale-base (thêm --scale-lo/--scale-hi), tính
  lại inlier + frac_recon_within + coverage_bim sau refine, refine keep-best,
  cổng yêu cầu đồng thời scale_ok ∧ inlier≥0.7 ∧ frac@0.2≥0.5 ∧ coverage_bim@0.2≥0.25.
- Thêm align_coverage.py (căn cứng tối ưu độ phủ, quét tịnh tiến) và
  estimate_normals.py (PCA cục bộ k=16; DUSt3R/MASt3R không có normals).
- Sửa evaluate_raamac.py truyền đúng 7 types đóng băng (IfcWall,
  IfcWallStandardCase, IfcSlab, IfcRoof, IfcBeam, IfcColumn, IfcCovering) —
  DEFAULT_TYPES chỉ 5 nên thiếu 2 GUID test; khớp giao thức pipeline chính.
- Hợp đồng khung tọa độ xác nhận: progress_baseline áp G3 (ref→IFC) lên cả
  cloud lẫn normals → đầu vào phải ở khung ref (alignment_ref), không phải
  alignment_bim (tránh biến đổi kép). Plane mode dùng abs(dot) normals.

Bảng so sánh căn chỉnh vào BIM (cổng đã sửa, KHÔNG nghiệm giả):
| Cloud | inlier_core | frac@0.2 | coverage_bim@0.2 | Nhận xét |
|---|---|---|---|---|
| MASt3R metric (rigid s=1) | 0.829 | 0.863 | 0.005 | chính xác cục bộ tốt nhưng chỉ 1 cụm nhỏ |
| MASt3R (tối ưu coverage) | — | 0.225 | 0.004 | vẫn cụm nhỏ, không phải dải mặt đứng |
| DUSt3R logwin (similarity) | 0.228 | 0.236 | 0.176 | phủ rộng nhất nhưng méo (median 0.45 m) |

- DUSt3R logwin: coverage_ref@0.2=0.314 (so cloud chuyên nghiệp), bbox kéo
  xiên ~58 m (BIM 23.4 m) → biến dạng phi cứng do scale trôi (non-metric).

KẾT QUẢ RAAMAC (DUSt3R logwin + normals ước lượng, cohort đóng băng, 7 types):
- validation: 28/44 = 63.64% (macroF1 0.382)
- test: 13/21 = 61.90% (macroF1 0.433, Wilson95 [0.409, 0.792])
- Mốc: 50% ĐẠT, 60% ĐẠT, 70% chưa, 80% chưa.
- Đối chiếu: cloud chuyên nghiệp week26.ply = 95.24% (20/21) — không đổi.
- Kết luận trung thực: video đơn quỹ đạo nén, nhà trắng ít texture → cloud
  video phủ ~17.6% BIM, đạt ~62% test; CHƯA đạt 80%. Nút thắt = độ phủ + độ méo
  của reconstruction, không phải pipeline đánh giá (đã chạy đúng giao thức).
- Hướng tiếp: cải thiện DUSt3R (graph dày hơn + nhiều vòng alignment hơn để
  giảm trôi scale) trên T4 trả phí; đo coverage_ref + validation trước, đọc
  test trung thực; KHÔNG tune trên test đóng băng.

## 2026-11-13 (tiếp) — Thử nghiệm niter + chuyển hướng MASt3R toàn cảnh

- Sửa orchestrator run_dust3r_colab.py: stream tiến trình TRỰC TIẾP (trước gom
  stdout nên không thấy tiến trình, nhầm lần niter=3000 chậm ~26 phút là treo),
  giới hạn 30 phút, tham số backend/graph/niter/conf/stride/size. Phát hiện
  alignment DUSt3R chạy ~1.9 it/s (niter=1000 ~9 phút, 3000 ~26 phút).
- DUSt3R logwin niter=1500 (113 frame, 384px, conf 1.5): 9.02M điểm, loss
  0.0139, 1057 s. coverage_ref@0.2=0.314 (≈ baseline 0.314), chamfer 1.40 m
  (baseline 1.35). RAAMAC: test 13/21=61.9% (như baseline), validation 24/44
  =54.5% (GIẢM so với 63.6%). Kết luận: thêm vòng alignment KHÔNG giảm méo;
  biến dạng là bản chất DUSt3R non-metric trên nhà trắng ít texture, 1 quỹ đạo.
  Session đã dừng ngay sau tải (tiết kiệm units).
- Trần video-only DUSt3R ≈ 62% test; phủ ~17.6% BIM. Chưa đạt 70/80%.
- Hướng còn triển vọng nhất: MASt3R METRIC (cứng, không trôi scale). Lần trước
  (25 frame thưa stride 9, 384px) ra cụm nhỏ (coverage_bim 0.5%) do graph không
  kết nối toàn cảnh. Thiết kế lại: 56 frame @256px (bộ nhớ ≈ 25 frame @384 đã
  chạy được) + logwin + conf 1.0 + niter 1000 → kỳ vọng cloud cứng toàn cảnh.
  Đang chạy trên T4 trả phí (mast3r_full).

## 2026-11-13 (kết quả MASt3R toàn cảnh + TỔNG KẾT trung thực)

- MASt3R metric 57 frame @256px, logwin, conf 1.0, niter 1000: 2.10M điểm,
  loss 0.060, 315 s, KHÔNG OOM, session dừng ngay. coverage_ref@0.2=0.119
  (thấp hơn DUSt3R 0.314) nhưng coverage_recon@0.2=0.476 (cao hơn DUSt3R 0.29)
  → metric cứng chính xác cục bộ hơn, phủ ít hơn.
- RAAMAC MASt3R toàn cảnh: validation 28/44=63.6%, TEST 14/21=66.7%
  (Wilson95 [0.454, 0.828]) — KẾT QUẢ VIDEO TỐT NHẤT. Mốc 50% ĐẠT, 60% ĐẠT,
  70% chưa, 80% chưa.

BẢNG TỔNG HỢP RAAMAC TEST (cohort đóng băng 21, 7 types, KHÔNG tune test):
| Nguồn cloud | Test | Val | coverage_ref@0.2 | Ghi chú |
|---|---|---|---|---|
| Chuyên nghiệp week26.ply | 20/21=95.24% | — | (tham chiếu) | mục tiêu lớn ĐẠT với scan chuyên nghiệp |
| DUSt3R logwin niter1000 | 13/21=61.9% | 63.6% | 0.314 | méo phi cứng |
| DUSt3R logwin niter1500 | 13/21=61.9% | 54.5% | 0.314 | thêm vòng không cải thiện |
| MASt3R metric 25 frame | (cụm nhỏ) | — | — | coverage_bim 0.5%, không dùng được |
| MASt3R metric 57f@256 | 14/21=66.7% | 63.6% | 0.119 | VIDEO TỐT NHẤT, cứng chính xác |

KẾT LUẬN TRUNG THỰC:
- Pipeline TỰ ĐỘNG end-to-end đã hoàn thiện: video/frame → Colab GPU (DUSt3R/
  MASt3R) → căn chỉnh (similarity/rigid/coverage, cổng chất lượng chống suy biến)
  → ước lượng normals → đánh giá RAAMAC cohort đóng băng → báo cáo mốc.
  Orchestrator tự quản lý session (new→upload→run→download→STOP), stream tiến
  trình, giới hạn thời gian; chỉ dùng URL chính thức (Google Colab, github.com/
  naver, HF naver), không proxy.
- Với cloud CHUYÊN NGHIỆP: RAAMAC 95.24% ≥ 80% (mục tiêu lớn ĐẠT).
- Với cloud CHỈ TỪ VIDEO (1 quỹ đạo drone nén, nhà trắng ít texture): trần
  ~66.7% test (MASt3R metric). ĐẠT 50/60%, CHƯA đạt 70/80%.
- Nguyên nhân gốc (không phải pipeline đánh giá): (1) 1 quỹ đạo → góc nhìn hạn
  chế, không thấy mọi mặt đứng/phần tử; (2) tường trắng ít texture → ước lượng
  độ sâu/khớp đặc trưng kém; (3) video nén → ít chi tiết; (4) DUSt3R non-metric
  trôi scale/méo, MASt3R metric cứng nhưng phủ ít. Đã thử 3 backend (COLMAP,
  DUSt3R, MASt3R), nhiều graph/niter/độ phân giải, nhiều chiến lược căn chỉnh —
  trần ~67% là bền vững.
- KHÔNG thổi accuracy: giữ nguyên 21 phần tử test (kể cả unobserved tính sai),
  không tune trên test đóng băng, không dùng cloud chuyên nghiệp làm truth cho
  nhánh video.
- Để đạt ≥80% TỪ VIDEO: cần dữ liệu nguồn tốt hơn (đa quỹ đạo/đa góc, độ phân
  giải cao, bề mặt có texture) hoặc reconstruction chuyên nghiệp; pipeline hiện
  tại sẵn sàng chạy ngay khi có dữ liệu đó.

## 2026-11-13 (tiếp) — Pipeline MỘT LỆNH end-to-end + kiểm chứng

- Trước đó kết quả ~60% chạy bằng các bước script rời nhau (dựng tự động qua
  run_dust3r_colab.py; căn chỉnh/normals/đánh giá gọi thủ công). Người dùng hỏi
  có phải pipeline tự động không → nối tất cả thành MỘT lệnh.
- Viết run_full_pipeline.py: dựng (Colab, tự new→upload→run→download→STOP) →
  align_and_compare (vào cloud tham chiếu) → estimate_normals → evaluate_raamac
  (cohort đóng băng, 7 types) → pipeline_report.json + bảng mốc. --skip-remote
  dùng fused.ply sẵn để chạy lại phần cục bộ (không tốn GPU).
- KIỂM CHỨNG: `run_full_pipeline.py --backend dust3r --skip-remote --out-dir
  artifacts/dust3r_hi` tái tạo ĐÚNG test 13/21=61.9% (val 54.5%), mốc 50/60 ĐẠT,
  70/80 chưa. Xác nhận pipeline một lệnh chạy thông suốt end-to-end.
- Lệnh đầy đủ cho video mới: `run_full_pipeline.py --backend mast3r --size 256
  --stride 4 --conf 1.0 --out-dir <dir>` (tự dựng trên T4 rồi đánh giá).

## 2026-11-13 (tiếp) — XÁC NHẬN end-to-end MỘT LỆNH từ video (skip_remote=false)

- Lần chạy đầu lỗi TimeoutExpired ở bước ghép zip (treo kết nối Colab CLI thoáng
  qua, thao tác ghép chỉ vài giây). Sửa: bước ghép idempotent + retry 3 lần +
  timeout 600s. Session tự STOP khi lỗi (finally).
- Chạy lại trọn vẹn MỘT LỆNH từ frames video (KHÔNG skip-remote):
  `run_full_pipeline.py --backend dust3r --graph logwin --niter 1000 --conf 1.5
  --stride 2 --size 384 --out-dir artifacts/pipeline_e2e`.
- Tự động: new T4 (dust34262) → upload 3 chunk → ghép zip 138.6MB → clone
  github.com/naver/dust3r + croco → weights HF naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
  → inference 113 frame/678 cặp → alignment niter1000 loss 0.014 → 9.02M điểm
  → download fused.ply → STOP session → căn chỉnh vào cloud tham chiếu →
  estimate_normals (435382 điểm) → evaluate_raamac (7 types đóng băng) → report.
- KẾT QUẢ (pipeline_report.json, skip_remote=false): test 13/21=61.9%
  (Wilson [0.409,0.792]), validation 28/44=63.6%. Mốc 50% ĐẠT, 60% ĐẠT,
  70% chưa, 80% chưa. KHỚP kết quả ~60% trước đó.
- KẾT LUẬN: pipeline MỘT LỆNH tự động end-to-end từ video đã được xác nhận
  chạy thông suốt (dựng GPU → căn chỉnh → normals → đánh giá → báo cáo), tự
  quản lý session (stop khi xong/lỗi), chỉ dùng URL chính thức. Tái tạo đúng
  kết quả ~60%. Không còn session nào chạy.
