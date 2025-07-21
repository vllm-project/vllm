#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e
set -x

cd /vllm-workspace/

# uninstall vllm
pip3 uninstall -y vllm
# restore the original files
mv test_docs/vllm ./vllm

# remove all compilers
apt remove --purge build-essential -y
apt autoremove -y

echo 'import os; os.system("touch /tmp/changed.file")' >> vllm/__init__.py

# TESTING, TO BE REMOVED
VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL=1 VLLM_USE_PRECOMPILED=1 pip3 install -vvv -e . \
  --extra-index-url https://download.pytorch.org/whl/test/cu128

pip3 install nvidia-cudnn-cu12==9.5.1.17 --force-reinstall

# Run the script
python3 -c 'import vllm'

# Check if the clangd log file was created
if [ ! -f /tmp/changed.file ]; then
    echo "changed.file was not created, python only compilation failed"
    exit 1
fi
