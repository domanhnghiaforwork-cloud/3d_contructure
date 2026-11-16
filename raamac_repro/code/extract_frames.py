"""Trích frame dày từ video drone week26 bằng PyAV, kèm chỉ số độ nét."""
import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image


def sharpness(gray: np.ndarray) -> float:
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min-sharpness", type=float, default=0.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    container = av.open(args.video)
    stream = container.streams.video[0]
    src_fps = float(stream.average_rate)
    step = max(1, round(src_fps / args.fps))

    records = []
    kept = 0
    for i, frame in enumerate(container.decode(video=0)):
        if i % step != 0:
            continue
        t = i / src_fps
        img = frame.to_image()
        gray = np.asarray(img.convert("L"), dtype=np.float64)
        s = sharpness(gray)
        rec = {"index": i, "time_s": round(t, 3), "sharpness": round(s, 2)}
        if s < args.min_sharpness:
            rec["skipped"] = "blurry"
            records.append(rec)
            continue
        name = f"frame_{kept:04d}_{t:07.2f}s.jpg"
        img.save(out / name, quality=92)
        rec["file"] = name
        records.append(rec)
        kept += 1

    meta = {
        "video": str(args.video),
        "source_fps": src_fps,
        "target_fps": args.fps,
        "step": step,
        "frames_kept": kept,
        "min_sharpness": args.min_sharpness,
        "frames": records,
    }
    (out / "frames.json").write_text(json.dumps(meta, indent=1))
    sharp = [r["sharpness"] for r in records if "file" in r]
    print(f"kept={kept} sharpness min/median/max = "
          f"{min(sharp):.1f} / {np.median(sharp):.1f} / {max(sharp):.1f}")


if __name__ == "__main__":
    main()