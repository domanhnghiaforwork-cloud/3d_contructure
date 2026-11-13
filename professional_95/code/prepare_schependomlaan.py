import argparse
import json
import shutil
import zipfile
from pathlib import Path

from progress_baseline import sha256


ARCHIVE_BYTES = 1457592828
ARCHIVE_SHA256 = "23a628f46a88a1f04465c3d26d2c5d4892efa229dae6641a83542166b138a8ba"
FILES = {
    "Design model IFC/IFC Schependomlaan.ifc": "model.ifc",
    "Point Clouds/Week 26/PLY/De Schepen week 27_t.ply": "week26.ply",
    "Point Clouds/Week 27/PLY/De Schepen week 27_t.ply": "week27.ply",
    "Point Clouds/Week 28/PLY/De Schepen week 28_t.ply": "week28.ply",
    "Point Clouds/Week 29/PLY/De Schepen week 29_t.ply": "week29.ply",
    "Point Clouds/Week 30/PLY/De Schepen week 30_t.ply": "week30.ply",
    "Comparison As Planned As Built/comparison_all weeks.xlsx": "raamac_comparison.xlsx",
    "LICENSE.MD": "LICENSE.MD",
    "README.md": "README_source.md",
}


def prepare(archive, output):
    if archive.stat().st_size != ARCHIVE_BYTES or sha256(archive) != ARCHIVE_SHA256:
        raise ValueError("Archive differs from the pinned downloaded release 1.1")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as bundle:
        for source_name, target_name in FILES.items():
            info = bundle.getinfo(source_name)
            target = output / target_name
            with bundle.open(info) as source, target.open("xb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            record = {
                "archive_path": source_name, "local_file": target_name,
                "bytes": target.stat().st_size, "sha256": sha256(target),
                "zip_crc32": f"{info.CRC:08x}",
            }
            records.append(record)
            print(json.dumps(record), flush=True)
    manifest = {
        "source": "https://github.com/openBIMstandards/Archive-DataSetSchependomlaan/releases/tag/1.1",
        "archive_sha256": ARCHIVE_SHA256,
        "archive_sha256_origin": "Computed on the downloaded release; not a separately published author checksum",
        "license": "CC BY 4.0; retain original-owner attribution and source documentation",
        "files": records,
        "warnings": [
            "Week26 uses the archive's Week26 directory even though its PLY basename says week27; verify chronology against independent evidence.",
            "RAAMAC comparison is algorithm output, not independent ground-truth construction labels.",
            "The original design IFC is used, not a date-filtered as-planned model that would omit future elements.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.output)


if __name__ == "__main__":
    main()
