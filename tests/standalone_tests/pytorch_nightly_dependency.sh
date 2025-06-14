#!/bin/sh

set -e
set -x

echo ">>> Current torch-related packages (before uninstall):"
pip freeze | grep -E '^torch|^torchvision|^torchaudio' || echo "None found"

echo ">>> Uninstalling previous PyTorch packages (if any)"
pip uninstall -y torch torchvision torchaudio vllm|| true

echo ">>> Installing nightly torch packages"
pip install --quiet torch torchvision torchaudio --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu128

echo ">>> Capturing torch-related versions before requirements install"
pip freeze | grep -E '^torch|^torchvision|^torchaudio' | sort > before.txt
echo "Before:"
cat before.txt

echo ">>> Installing requirements/nightly_torch_test.txt"
pip install --quiet -r requirements/nightly_torch_test.txt

echo ">>> Capturing torch-related versions after requirements install"
pip freeze | grep -E '^torch|^torchvision|^torchaudio' | sort > after.txt
echo "After:"
cat after.txt

echo ">>> Comparing versions"
if diff before.txt after.txt; then
  echo "✅ torch version not overridden."
else
  echo "torch version overridden by nightly_torch_test.txt"
  exit 1
fi
