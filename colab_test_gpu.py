import json, subprocess, datetime, torch

info = {
    "gpu": torch.cuda.get_device_name(0),
    "cuda": torch.cuda.is_available(),
    "torch": torch.__version__,
    "nvidia_smi": subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip(),
    "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "source": "local-script-via-colab-cli",
}
x = torch.randn(4096, 4096, device="cuda")
info["matmul_sum"] = float((x @ x).sum().item())
with open("/content/ket_qua_cli.json", "w") as f:
    json.dump(info, f, indent=2)
print(json.dumps(info, indent=2))