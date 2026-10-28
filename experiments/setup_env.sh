#!/bin/bash
# Build the Python environment used for the experiments (Python 3.12, CUDA GPU). Run from the repo root.
set -euo pipefail
command -v uv >/dev/null || pip install -q uv
uv venv -q --python 3.12 .venv
. .venv/bin/activate
uv pip install vllm
uv pip install flash-linear-attention datasets pandas pyarrow matplotlib scipy accelerate math-verify
# causal-conv1d needs a matching prebuilt wheel or a compile; try, but do not fail the env on it
uv pip install --no-build-isolation causal-conv1d || echo "causal-conv1d: install failed (fallback to torch conv)"
python - <<'PY'
import importlib, torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for m in ["vllm","transformers","fla","causal_conv1d","datasets","math_verify"]:
    try: print(m, getattr(importlib.import_module(m),"__version__","?"))
    except Exception as e: print(m, "MISSING", type(e).__name__, str(e)[:120])
PY
