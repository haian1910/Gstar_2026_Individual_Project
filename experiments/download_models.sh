#!/bin/bash
# Download the models into $HF_HOME (default ~/.cache/huggingface).
set -euo pipefail
for m in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-9B; do
  hf download "$m"
  echo "downloaded $m"
done
