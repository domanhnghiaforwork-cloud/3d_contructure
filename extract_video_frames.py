import argparse
import json
import math
from pathlib import Path

import av
from PIL import Image, ImageDraw

from progress_baseline import sha256


def extract(video, output, interval):
    if interval <= 0:
        raise ValueError("Frame interval must be positive")
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    next_time = 0.0
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            timestamp = float(frame.time) if frame.time is not None else index / float(stream.average_rate)
            if timestamp + 1e-6 < next_time:
                continue
            filename = f"frame_{len(frames):03d}_{timestamp:07.2f}s.jpg"
            image = frame.to_image()
            image.save(output / filename, quality=94)
            frames.append({"file": filename, "time_seconds": timestamp, "width": image.width, "height": image.height})
            next_time += interval
    if not frames:
        raise ValueError("No video frames decoded")
    tile_width, tile_height, label_height, columns = 480, 270, 24, 3
    rows = math.ceil(len(frames) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        with Image.open(output / frame["file"]) as image:
            image.thumbnail((tile_width, tile_height))
            x = index % columns * tile_width
            y = index // columns * (tile_height + label_height)
            sheet.paste(image, (x, y))
            draw.text((x + 5, y + tile_height + 4), f"{video.stem}: {frame['time_seconds']:.2f}s", fill="black")
    sheet.save(output / "contact_sheet.jpg", quality=94)
    metadata = {
        "source_video": str(video), "source_sha256": sha256(video),
        "interval_seconds": interval, "n_frames": len(frames), "frames": frames,
        "purpose": "Independent visual evidence preparation; no progress labels or predictions assigned",
    }
    (output / "frames.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=4.0)
    args = parser.parse_args()
    extract(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()
