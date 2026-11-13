#!/usr/bin/env bash
# Tạo môi trường ảo .venv ở gốc repo và cài dependencies cục bộ.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
echo "OK: .venv sẵn sàng. Kích hoạt: source .venv/bin/activate"