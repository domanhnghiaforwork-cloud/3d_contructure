"""Một lần chạy Colab gộp SfM + MVS (tận dụng tối đa phiên, không idle).

Upload kèm: colmap_reconstruct.py, dense_mvs.py, frames zip, watermark mask.
Env: VCR_ZIP (tên zip), VCR_MVS_SIZE (1200/1600).
"""
import os
import runpy
import sys

os.environ["VCR_SFM_ONLY"] = "1"
runpy.run_path("/content/vcr/colmap_reconstruct.py", run_name="__main__")

best = open("/content/vcr/out/best_model.txt").read().strip()
print(f"\n########## MVS on best model {best} ##########", flush=True)
sys.argv = ["dense_mvs.py", "--models", best]
runpy.run_path("/content/vcr/dense_mvs.py", run_name="__main__")