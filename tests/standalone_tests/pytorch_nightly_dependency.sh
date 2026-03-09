#!/bin/sh
# This script tests if the nightly torch packages are not overridden by the dependencies

set -e
set -x

if command -v rocminfo >/dev/null 2>&1; then
  echo "Skipping test for ROCm platform"
  exit 0
fi

cd /vllm-workspace/

rm -rf .venv

uv venv .venv

source .venv/bin/activate

# check the environment
uv pip freeze

echo ">>> Installing nightly torch packages"
uv pip install --quiet torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu128

echo ">>> Capturing torch-related versions before requirements install"
uv pip freeze | grep -E '^torch|^torchvision|^torchaudio' | sort > before.txt
echo "Before:"
cat before.txt

echo ">>> Installing requirements/nightly_torch_test.txt"
uv pip install --quiet -r requirements/nightly_torch_test.txt

echo ">>> Capturing torch-related versions after requirements install"
uv pip freeze | grep -E '^torch|^torchvision|^torchaudio' | sort > after.txt
echo "After:"
cat after.txt

echo ">>> Comparing versions"
if diff before.txt after.txt; then
  echo "torch version not overridden."
else
  echo "torch version overridden by nightly_torch_test.txt, \
  if the dependency is not triggered by the pytorch nightly test,\
  please add the dependency to the list 'white_list' in tools/pre_commit/generate_nightly_torch_test.py"
  exit 1
fi
