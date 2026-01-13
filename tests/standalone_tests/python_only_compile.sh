#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e

merge_base_commit=$(git merge-base HEAD origin/main)
echo "INFO: current merge base commit with main: $merge_base_commit"
git show --oneline -s $merge_base_commit

# test whether the metadata.json url is valid, retry each 3 minutes up to 5 times
# this avoids cumbersome error messages & manual retries in case the precompiled wheel
# for the given commit is still being built in the release pipeline
meta_json_url="https://wheels.vllm.ai/$merge_base_commit/vllm/metadata.json"
echo "INFO: will use metadata.json from $meta_json_url"

for i in {1..5}; do
    echo "Checking metadata.json URL (attempt $i)..."
    if curl --fail "$meta_json_url" > metadata.json; then
        echo "INFO: metadata.json URL is valid."
        # check whether it is valid json by python (printed to stdout)
        if python3 -m json.tool metadata.json; then
            echo "INFO: metadata.json is valid JSON. Proceeding with the check."
            # check whether there is an object in the json matching:
            # "package_name": "vllm", and "platform_tag" matches the current architecture
            # see `determine_wheel_url` in setup.py for more details
            if python3 -c "import platform as p,json as j,sys as s; d = j.load(open('metadata.json')); \
             s.exit(int(not any(o.get('package_name') == 'vllm' and p.machine() in o.get('platform_tag') \
             for o in d)))" 2>/dev/null; then
                echo "INFO: metadata.json contains a pre-compiled wheel for the current architecture."
                break
            else
                echo "WARN: metadata.json does not have a pre-compiled wheel for the current architecture."
            fi
        else
            echo "CRITICAL: metadata.json exists but is not valid JSON, please do report in #sig-ci channel!"
            echo "INFO: metadata.json content:"
            cat metadata.json
            exit 1
        fi
    fi
    # failure handling & retry logic
    if [ $i -eq 5 ]; then
        echo "ERROR: metadata is still not available after 5 attempts."
        echo "ERROR: Please check whether the precompiled wheel for commit $merge_base_commit is available."
        echo " NOTE: If $merge_base_commit is a new commit on main, maybe try again after its release pipeline finishes."
        echo " NOTE: If it fails, please report in #sig-ci channel."
        exit 1
    else
        echo "WARNING: metadata is not available. Retrying after 5 minutes..."
        sleep 300
    fi
done

set -x

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
    echo "ERROR: changed.file was not created, python only compilation failed"
    exit 1
fi
