#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e
set -x

merge_base_commit=$(git merge-base HEAD origin/main)
echo "Current merge base commit with main: $merge_base_commit"
git show --oneline -s $merge_base_commit

cd /vllm-workspace/

# uninstall vllm
pip3 uninstall -y vllm
# restore the original files
mv src/vllm ./vllm

# remove all compilers
apt remove --purge build-essential -y
apt autoremove -y

echo 'import os; os.system("touch /tmp/changed.file")' >> vllm/__init__.py

VLLM_PRECOMPILED_WHEEL_COMMIT=$merge_base_commit VLLM_USE_PRECOMPILED=1 pip3 install -vvv -e .

# Run the script
python3 -c 'import vllm'

# Check if the clangd log file was created
if [ ! -f /tmp/changed.file ]; then
    echo "changed.file was not created, python only compilation failed"
    exit 1
fi
