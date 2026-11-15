# raw — dữ liệu gốc Schependomlaan (cloud chuyên nghiệp các tuần + IFC)

Thư mục này chứa dữ liệu gốc của nhánh chuyên nghiệp. **Một phần đã commit trong
repo, phần quá nặng thì tải riêng** theo hướng dẫn dưới (GitHub chặn file >100MB).

## Đã commit sẵn trong repo

| File | Kích thước | Nguồn trong archive |
|---|---|---|
| `week26.ply` | 73.988.849 B (~70.6MB) | `Point Clouds/Week 26/PLY/De Schepen week 27_t.ply` |
| `model.ifc` | 49.286.967 B (~47MB) | `Design model IFC/IFC Schependomlaan.ifc` |

## KHÔNG commit (vượt giới hạn 100MB/file của GitHub) — tải riêng

| File | Kích thước | Nguồn trong archive |
|---|---|---|
| `week27.ply` | 205.962.475 B (~196MB) | `Point Clouds/Week 27/PLY/De Schepen week 27_t.ply` |
| `week28.ply` | 440.113.608 B (~420MB) | `Point Clouds/Week 28/PLY/De Schepen week 28_t.ply` |
| `week29.ply` | 475.084.870 B (~453MB) | `Point Clouds/Week 29/PLY/De Schepen week 29_t.ply` |
| `week30.ply` | 263.330.451 B (~251MB) | `Point Clouds/Week 30/PLY/De Schepen week 30_t.ply` |

## Cách tải (nguồn chính thức, release 1.1)

Nguồn: `https://github.com/openBIMstandards/Archive-DataSetSchependomlaan` — release
tag `1.1`. **Tải cả zip rồi giải nén**, không tải lẻ từng file (theo README nguồn).
Các asset của release 1.1 (đã xác minh qua GitHub API):

- Point clouds (đủ week26–30, ~743MB):
  `https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1/schependomlaan.-.Point.Clouds.zip`
- Chỉ IFC (~86MB):
  `https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1/schependomlaan.-.IFC.zip`
- Đầy đủ tất cả (~1.457GB):
  `https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1/schependomlaan.-.complete.zip`

**Cách nhanh nhất:** chạy script tự động (tải phần còn thiếu, giải nén đúng đường dẫn, đổi tên):

```bash
bash ../download_data.sh        # từ thư mục professional_95/
```

**Hoặc thủ công** (ví dụ week29, dùng asset Point.Clouds.zip):

```bash
curl -L -o pc.zip "https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1/schependomlaan.-.Point.Clouds.zip"
unzip -p pc.zip "Point Clouds/Week 29/PLY/De Schepen week 29_t.ply" > week29.ply
# IFC:
curl -L -o ifc.zip "https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/download/1.1/schependomlaan.-.IFC.zip"
unzip -p ifc.zip "Design model IFC/IFC Schependomlaan.ifc" > model.ifc
```

Đổi tên theo bảng trên (`weekNN.ply`, `model.ifc`). Nếu đường dẫn trong zip khác,
liệt kê bằng `unzip -l pc.zip | grep PLY` rồi chỉnh lại.

## Kiểm tra toàn vẹn (sha256, theo manifest nguồn)

| File | sha256 |
|---|---|
| week26.ply | `5bf756f0081c0204ac1c57119040ee986942f9d45a82e5bc26c0fb1a3783e88b` |
| week27.ply | `08ec2b2c228ededda51f564ae1b0424bc0cf14afa65ef8dfd8f3fedea50a6247` |
| week28.ply | `8b694f83d24d2ae8a8be56f3c77654d531c9c22f61da7193a46ffc4f135c5ea6` |
| week29.ply | `a11bc2c4086169d9b8c244e66868eb19a6261f0e9019e0c249142fcb1e7cf4b9` |
| week30.ply | `2907e094f72e46a53e78d8c6721faa9e8e36942570611f0ee80597574e2efc27` |
| model.ifc | `2c3565ca1904f2aa61adab92024cf3755b2c5b21a498144d3094d7cb58cebec7` |

```bash
sha256sum week29.ply   # đối chiếu bảng trên
```

## Giấy phép & lưu ý nguồn

- Giấy phép **CC BY 4.0**: giữ attribution chủ dữ liệu gốc + tài liệu nguồn
  (TU Eindhoven ISBE, Hendriks Bouw en Ontwikkeling, ROOT, TNO, RAAMAC).
- **Cảnh báo chronology:** thư mục `Week 26` trong archive chứa PLY basename
  "week 27"; đã kiểm tra chéo trước khi dùng làm week26. Đối chiếu trình tự thời
  gian bằng chứng cứ độc lập nếu dùng các tuần khác.
- File `raamac_comparison.xlsx` (nếu tải) là **output thuật toán**, KHÔNG phải nhãn
  ground-truth độc lập; không dùng làm nhãn đánh giá.
- Chỉ dùng link chính thức ở trên; không dùng proxy/mirror tạm.