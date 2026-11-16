"""Dựng viewer 3D TƯƠNG TÁC tự chứa cho cloud COLMAP repro tuần 26 (4tr điểm).

Tái dùng engine WebGL + load_reduce/quantize của build_viewer_6clouds.py.
2 cloud: (1) repro RAW + ALIGNED, (2) week26.ply chuyên nghiệp tham chiếu.
Chồng tham chiếu mờ lên bản căn chỉnh để đối chiếu. Mở file:// là chạy, không CDN.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE))
import build_viewer_6clouds as bv  # noqa: E402
from build_viewer_6clouds import HTML, load_reduce, quantize  # noqa: E402

W26 = PKG / "artifacts" / "week26"
REF_PLY = PKG / "data" / "raw" / "week26.ply"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=150000)
    ap.add_argument("--out", type=Path, default=PKG / "viewer" / "viewer_repro_week26.html")
    args = ap.parse_args()
    bv.MAX_POINTS = args.max_points
    OUT = args.out
    fused = W26 / "fused.ply"
    aligned = W26 / "alignment_ref" / "recon_aligned.ply"
    metrics = json.loads((W26 / "alignment_ref" / "metrics.json").read_text())

    rxyz, rrgb, rn, rd = load_reduce(fused)
    c = (rxyz.max(0) + rxyz.min(0)) / 2
    e = float((rxyz.max(0) - rxyz.min(0)).max())
    pos, col, gz = quantize(rxyz, rrgb, c, e)
    repro = dict(dir="raamac_repro/artifacts/week26", name="1 · COLMAP repro tuần 26",
                 cfg="268 frame @5fps 1080p · SIFT GPU 16384 · exhaustive matching · CPU BA · MVS 1200",
                 note="Bản dựng chung: phủ 8% tham chiếu, test 66,7% (=baseline), built 0/6. "
                      "Video marketing cắt 5 cảnh → SfM vỡ 4 model (chỉ lấy model lớn nhất = 1 góc nhà). "
                      "Đang khắc phục: dựng riêng 6 shot + hợp nhất cloud đã căn chỉnh (đa góc).",
                 orig=rn, shown=len(rxyz), dropped=rd, gz=gz,
                 sizeMB=round(fused.stat().st_size / 1e6, 1), pos=pos, col=col)
    axyz, argb, an, ad = load_reduce(aligned)
    repro["al"] = dict(path="raamac_repro/artifacts/week26/alignment_ref/recon_aligned.ply",
                       frame="ref", scale=round(metrics["scale"], 3),
                       cov=round(metrics["coverage_ref_at_tau"]["0.2"], 3),
                       orig=an, shown=len(axyz), dropped=ad)

    fxyz, frgb, fn, fd = load_reduce(REF_PLY)
    fc = (fxyz.max(0) + fxyz.min(0)) / 2
    fe = float((fxyz.max(0) - fxyz.min(0)).max())
    fpos, fcol, fgz = quantize(fxyz, frgb, fc, fe)
    ref = dict(dir="ref", name="2 · week26.ply chuyên nghiệp",
               cfg="Scan chuyên nghiệp RAAMAC — SfM từ video drone đa chuyến bay",
               note="Đối chiếu: 95,24% test, built 5/6. Phủ kín nhà đa góc. KHÔNG phải cloud repro.",
               orig=fn, shown=len(fxyz), dropped=fd, gz=fgz,
               sizeMB=round(REF_PLY.stat().st_size / 1e6, 1), pos=fpos, col=fcol,
               al=dict(path="data/schependomlaan/raw/week26.ply", frame="ref gốc",
                       scale=1.0, cov=None, orig=fn, shown=len(fxyz), dropped=fd))

    allal = __import__("numpy").concatenate([axyz, fxyz])
    ac = (allal.max(0) + allal.min(0)) / 2
    ae = float((allal.max(0) - allal.min(0)).max())
    for entry, xyz, rgb in [(repro, axyz, argb), (ref, fxyz, frgb)]:
        p, cc, g = quantize(xyz, rgb, ac, ae)
        entry["al"].update(pos=p, col=cc, gz=g)

    html = HTML.replace("__DATA__", json.dumps({"clouds": [repro, ref]},
                                               ensure_ascii=False, separators=(",", ":")))
    OUT.write_text(html, encoding="utf-8")
    print(f"repro raw {rn:,}->{len(rxyz):,} | aligned {an:,}->{len(axyz):,} | ref {fn:,}->{len(fxyz):,}")
    print(f"viewer {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()