"""Phát hiện cut (cảnh ghép) trong video drone marketing và chia frame thành shot.

Video marketing bị cắt thành nhiều shot khác góc quay -> SfM vỡ model nếu dựng chung.
Mỗi shot = một 'chuyến bay' liên tục (faithful RAAMAC đa góc). Dựng riêng từng shot
rồi hợp nhất các cloud ĐÃ căn chỉnh -> phủ đa góc như cloud chuyên nghiệp.

Cut = frame-to-frame diff > 4x median. Ghi shots.json mỗi tuần.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FRAMES = HERE.parent / "frames"
CUT_FACTOR = 4.0


def detect(frames):
    prev = None
    diffs = []
    for i, f in enumerate(frames):
        a = np.asarray(Image.open(f).convert("L").resize((160, 90)), dtype=np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(a - prev))))
        prev = a
    d = np.array(diffs)
    med = np.median(d)
    cuts = [i + 1 for i, v in enumerate(diffs) if v > CUT_FACTOR * med]
    bounds = [0] + cuts + [len(frames)]
    shots = []
    for k in range(len(bounds) - 1):
        s, e = bounds[k], bounds[k + 1]
        if e - s >= 8:  # shot quá ngắn (<8 frame) bỏ qua
            shots.append({"id": len(shots), "start": s, "end": e,
                          "files": [f.name for f in frames[s:e]]})
    return cuts, shots, float(med)


def main():
    summary = {}
    for week in [26, 27, 28, 29, 30]:
        wd = FRAMES / f"week{week}"
        frames = sorted(wd.glob("*.jpg"))
        if not frames:
            continue
        cuts, shots, med = detect(frames)
        meta = {"week": week, "n_frames": len(frames), "median_diff": round(med, 2),
                "cuts": cuts, "n_shots": len(shots), "shots": shots}
        (wd / "shots.json").write_text(json.dumps(meta, indent=1))
        summary[week] = {"n_frames": len(frames), "cuts": cuts,
                         "shot_sizes": [s["end"] - s["start"] for s in shots]}
        print(f"week {week}: {len(frames)} frames, {len(cuts)} cuts -> "
              f"{len(shots)} shots {[s['end']-s['start'] for s in shots]}", flush=True)
    (FRAMES / "shots_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()