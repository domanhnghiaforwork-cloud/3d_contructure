import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


started = time.monotonic()
root = Path("/content")
environment = root / "bim_progress_py312"
output = root / "colab_validation.json"
report = {
    "run_time_utc": datetime.now(timezone.utc).isoformat(),
    "purpose": "Colab portability and unit/smoke validation only; not real-world progress accuracy",
    "kernel_python": platform.python_version(),
    "platform": platform.platform(),
    "status": "failed",
    "stages": [],
}


def stage(name, command, **kwargs):
    print(name, flush=True)
    result = subprocess.run(command, text=True, capture_output=True, **kwargs)
    report["stages"].append({
        "name": name, "command": command, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    })
    print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, flush=True)
    result.check_returncode()
    return result.stdout


try:
    stage("Install uv for managed Python", [sys.executable, "-m", "pip", "install", "uv"])
    stage("Create Python 3.12 research environment", [sys.executable, "-m", "uv", "venv", "--python", "3.12", str(environment)])
    python = str(environment / "bin" / "python")
    stage("Install pinned dependencies", [sys.executable, "-m", "uv", "pip", "install", "--python", python, "-r", str(root / "requirements.lock.txt")])
    report["research_python"] = stage("Research Python version", [python, "--version"]).strip()
    report["gpu"] = stage("GPU availability", ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]).strip()
    report["packages"] = stage("Freeze dependencies", [sys.executable, "-m", "uv", "pip", "freeze", "--python", python]).splitlines()
    stage(
        "Synthetic smoke and metric unit tests, not real-world accuracy",
        [python, "-m", "unittest", "-v", "test_evaluate_progress", "test_progress_baseline", "test_registration", "test_semantic_groups"],
        cwd=root, env={**os.environ, "MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"},
    )
    report["status"] = "passed"
finally:
    report["elapsed_seconds"] = time.monotonic() - started
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "stages"}, indent=2, ensure_ascii=False), flush=True)
