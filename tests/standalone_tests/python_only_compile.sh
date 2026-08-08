#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e

merge_base_commit=""
rocm_wheel=""
if [[ -n "${ROCM_PATH:-}" || -d /opt/rocm ]]; then
    # Native CI passes the verified wheel artifact explicitly. Legacy ROCm
    # images carry the same-build wheel in /opt/vllm-wheels.
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
        rocm_wheel="${VLLM_PRECOMPILED_WHEEL_LOCATION}"
        if [[ ! -f "${rocm_wheel}" || "$(basename "${rocm_wheel}")" != vllm-*.whl ]]; then
            echo "ERROR: invalid ROCm wheel location: ${rocm_wheel}" >&2
            exit 1
        fi
    else
        shopt -s nullglob
        rocm_wheels=(/opt/vllm-wheels/vllm-*.whl)
        shopt -u nullglob
        if [[ "${#rocm_wheels[@]}" -ne 1 ]]; then
            echo "ERROR: expected exactly one vLLM wheel in /opt/vllm-wheels, found ${#rocm_wheels[@]}." >&2
            exit 1
        fi
        rocm_wheel="${rocm_wheels[0]}"
    fi
    echo "INFO: using same-build ROCm wheel: $rocm_wheel"
else
    # Some CI images do not include .git under /vllm-workspace. Their wrapper
    # passes CI_STANDALONE_MERGE_BASE from the agent checkout.
    if [[ -n "${CI_STANDALONE_MERGE_BASE:-}" ]]; then
        merge_base_commit="${CI_STANDALONE_MERGE_BASE}"
    elif merge_base_commit="$(git -C /vllm-workspace merge-base HEAD origin/main 2>/dev/null)"; then
        :
    elif merge_base_commit="$(git merge-base HEAD origin/main 2>/dev/null)"; then
        :
    else
        echo "ERROR: need a git checkout or CI_STANDALONE_MERGE_BASE to resolve wheels.vllm.ai commit." >&2
        exit 1
    fi

    echo "INFO: current merge base commit with main: $merge_base_commit"
    if git show --oneline -s "$merge_base_commit" 2>/dev/null; then
        :
    else
        echo "INFO: git show unavailable in this environment; using SHA above for precompiled metadata."
    fi

    # Avoid manual retries while a new main-branch wheel is still publishing.
    meta_json_url="https://wheels.vllm.ai/$merge_base_commit/vllm/metadata.json"
    echo "INFO: will use metadata.json from $meta_json_url"

    for i in {1..5}; do
        echo "Checking metadata.json URL (attempt $i)..."
        if curl --fail "$meta_json_url" > metadata.json; then
            echo "INFO: metadata.json URL is valid."
            # check whether it is valid json by python (printed to stdout)
            if python3 -m json.tool metadata.json; then
                echo "INFO: metadata.json is valid JSON. Proceeding with the check."
                # check whether it is for vllm and the current architecture
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
        if [ "$i" -eq 5 ]; then
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
fi

set -x

cd /vllm-workspace/

# uninstall vllm
pip3 uninstall -y vllm
# restore the original files
if [[ -d src/vllm ]]; then
    mv src/vllm ./vllm
elif [[ ! -d vllm ]]; then
    echo "ERROR: expected vllm package at /vllm-workspace/src/vllm or /vllm-workspace/vllm" >&2
    exit 1
fi

# remove all compilers
apt remove --purge build-essential -y
apt autoremove -y

rm -f /tmp/changed.file
echo 'import os; os.system("touch /tmp/changed.file")' >> vllm/__init__.py

# ROCm CI uses setuptools develop for editable installs (see Dockerfile.rocm).
if [[ -n "$rocm_wheel" ]]; then
    VLLM_PRECOMPILED_WHEEL_LOCATION="$rocm_wheel" VLLM_USE_PRECOMPILED=1 python3 setup.py develop
else
    VLLM_PRECOMPILED_WHEEL_COMMIT=$merge_base_commit VLLM_USE_PRECOMPILED=1 pip3 install -vvv -e .
fi
# Run the script
python3 -c 'import vllm'

# Check if the clangd log file was created
if [ ! -f /tmp/changed.file ]; then
    echo "ERROR: changed.file was not created, python only compilation failed"
    exit 1
fi
