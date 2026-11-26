#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/rocm/bin/hipcc)}
archs=("$@")
if [ ${#archs[@]} -eq 0 ]; then
  archs=($(python -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])'))
fi
mkdir -p lib
for arch in "${archs[@]}"; do
  "$HIPCC" -O3 --offload-arch="$arch" -shared -fPIC -DMK_MODEL_QWEN3_1_7B \
    -o "lib/libmk_qwen3_1_7b_vllm_${arch}.so" dense_vllm.hip
done
