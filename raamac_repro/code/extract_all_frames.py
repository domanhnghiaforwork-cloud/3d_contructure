"""Trích frame 1080p mật độ cao cho cả 5 video tuần 26-30 (tái hiện RAAMAC).

Mỗi tuần một bộ frame riêng từ đúng video tuần đó (RAAMAC dùng 1 chuyến bay/tuần).
Full 1920x1080, target 5fps (25fps gốc -> step 5), lọc frame mờ (sharpness < 200).
Tái dùng extract_frames.py của nhánh video; chạy sạch từng thư mục đầu ra.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
PY = os.environ.get("PYTHON") or sys.executable
EXTRACT = HERE / "extract_frames.py"
VIDEOS = PKG / "data" / "videos"
OUT_ROOT = PKG / "frames"

WEEK_VIDEO = {
    26: "week26.mp4",
    27: "_FA_8JYWoYQ.mp4",
    28: "gKLyBgRecLQ.mp4",
    29: "0Pev1Weh6_I.mp4",
    30: "QbBf31bY0xw.mp4",
}
FPS = 5.0
MIN_SHARPNESS = 200.0


def main():
    manifest = {}
    for week, video in WEEK_VIDEO.items():
        out = OUT_ROOT / f"week{week}"
        if out.exists():
            shutil.rmtree(out)
        cmd = [str(PY), str(EXTRACT), "--video", str(VIDEOS / video),
               "--out", str(out), "--fps", str(FPS),
               "--min-sharpness", str(MIN_SHARPNESS)]
        print(f"\n=== week {week}: {video}", flush=True)
        subprocess.run(cmd, check=True)
        meta = json.loads((out / "frames.json").read_text())
        manifest[week] = {"video": video, "frames_kept": meta["frames_kept"],
                          "source_fps": meta["source_fps"], "target_fps": FPS,
                          "min_sharpness": MIN_SHARPNESS}
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("\n=== MANIFEST ===")
    print(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()