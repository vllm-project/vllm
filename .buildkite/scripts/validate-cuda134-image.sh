#!/usr/bin/env bash

set -euo pipefail

image_ref="$1"
expected_arch="$2"

docker run --rm --entrypoint python3 "${image_ref}" -c '
import importlib.util
import os
import platform
import sys

import torch
import torchaudio
import torchvision
import vllm

assert platform.machine() == sys.argv[1]
assert torch.__version__ == "2.15.0.dev20260812+cu134"
assert torch.version.cuda == "13.4"
assert torchvision.__version__ == "0.29.0.dev20260813+cu134"
assert torchaudio.__version__ == "2.11.0.dev20260813+cu134"
assert importlib.util.find_spec("vllm._C_stable_libtorch") is not None
assert os.environ["TRITON_PTXAS_BLACKWELL_PATH"] == "/usr/local/cuda/bin/ptxas"
print(vllm.__version__)
print(torch.__version__, torch.version.cuda)
print(torchvision.__version__, torchaudio.__version__)
' "${expected_arch}"

docker run --rm --entrypoint bash "${image_ref}" -c '
set -euo pipefail
tmp_dir=$(mktemp -d)
trap '\''rm -rf "${tmp_dir}"'\'' EXIT
printf "%s\n" "int main(void) { return 0; }" > "${tmp_dir}/smoke.c"
printf "%s\n" "int main() { return 0; }" > "${tmp_dir}/smoke.cc"
printf "%s\n" \
  "extern int cuInit(unsigned int);" \
  "int call_cu_init(void) { return cuInit(0); }" \
  > "${tmp_dir}/cuda-link.c"
gcc "${tmp_dir}/smoke.c" -o "${tmp_dir}/smoke-c"
g++ "${tmp_dir}/smoke.cc" -o "${tmp_dir}/smoke-cxx"
gcc -shared -fPIC -Wl,--no-undefined "${tmp_dir}/cuda-link.c" \
  -L/usr/local/cuda/lib64/stubs -lcuda \
  -o "${tmp_dir}/smoke-cuda-link.so"
readelf -d "${tmp_dir}/smoke-cuda-link.so" | grep -q libcuda.so.1
"${tmp_dir}/smoke-c"
"${tmp_dir}/smoke-cxx"
'

docker run --rm --entrypoint /usr/local/cuda/bin/nvcc "${image_ref}" \
  --version | grep 'release 13.4'
docker run --rm --entrypoint /usr/local/cuda/bin/ptxas "${image_ref}" \
  --version | grep 'release 13.4'
