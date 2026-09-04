#!/bin/bash
# This script tests if the python only compilation works correctly
# for users who do not have any compilers installed on their system

set -e

merge_base_commit=""
rocm_wheel=""
is_rocm=0
_vllm_target_lower="$(printf '%s' "${VLLM_TARGET_DEVICE:-}" | tr '[:upper:]' '[:lower:]')"
if [[ "${_vllm_target_lower}" == "rocm" || -n "${ROCM_PATH:-}" || -d /opt/rocm ]] \
        || command -v rocminfo >/dev/null 2>&1; then
    is_rocm=1
fi
unset -v _vllm_target_lower

if [[ "${is_rocm}" == "1" ]]; then
    # Native CI passes the verified wheel artifact explicitly. Legacy ROCm
    # images carry the same-build wheel in /opt/vllm-wheels.
    if [[ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]]; then
        rocm_wheel="${VLLM_PRECOMPILED_WHEEL_LOCATION}"
        if [[ ! -f "${rocm_wheel}" || "$(basename "${rocm_wheel}")" != vllm-*.whl ]]; then
            echo "ERROR: invalid ROCm wheel location: ${rocm_wheel}" >&2
            exit 1
        fi
        rocm_wheel="$(realpath -- "${rocm_wheel}")"
    elif [[ -d /opt/vllm-wheels ]]; then
        shopt -s nullglob
        rocm_wheels=(/opt/vllm-wheels/vllm-*.whl)
        shopt -u nullglob
        if [[ "${#rocm_wheels[@]}" -ne 1 ]]; then
            echo "ERROR: expected exactly one vLLM wheel in /opt/vllm-wheels, found ${#rocm_wheels[@]}." >&2
            exit 1
        fi
        rocm_wheel="${rocm_wheels[0]}"
    fi
fi

if [[ -n "${rocm_wheel}" ]]; then
    echo "INFO: using same-build ROCm wheel: ${rocm_wheel}"
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

    # Test whether the metadata.json URL is valid, retry each 5 minutes up to 5 times.
    # This avoids manual retries while a new main-branch wheel is still publishing.
    if [[ "${is_rocm}" == "1" ]]; then
        _rocm_env_variant="$(python3 - <<'PY'
import ctypes
import os
from pathlib import Path


def get_rocm_version() -> str | None:
    rocm_home = os.environ.get("ROCM_HOME") or os.environ.get("ROCM_PATH") or "/opt/rocm"
    try:
        librocm_core = Path(rocm_home) / "lib" / "librocm-core.so"
        if not librocm_core.is_file():
            return None
        librocm = ctypes.CDLL(str(librocm_core))
        get_rocm_core_version = librocm.getROCmVersion
        major = ctypes.c_uint32()
        minor = ctypes.c_uint32()
        patch = ctypes.c_uint32()
        if get_rocm_core_version(
            ctypes.byref(major), ctypes.byref(minor), ctypes.byref(patch)
        ) == 0:
            return f"{major.value}.{minor.value}.{patch.value}"
    except Exception:
        return None
    return None


version = get_rocm_version()
if version:
    print(f"rocm{version.replace('.', '')}", end="")
PY
)"
        _available_variants="$(curl -sf "https://wheels.vllm.ai/rocm/${merge_base_commit}/" \
            | grep -oP 'rocm\d+' | sort -u | tr '\n' ' ' || true)"
        if [[ -n "${VLLM_PRECOMPILED_WHEEL_VARIANT:-}" ]]; then
            _rocm_variant="${VLLM_PRECOMPILED_WHEEL_VARIANT}"
            if [[ -n "${_rocm_env_variant}" && "${_rocm_variant}" != "${_rocm_env_variant}" ]]; then
                echo "ERROR: VLLM_PRECOMPILED_WHEEL_VARIANT=${_rocm_variant} does not match detected environment ROCm variant ${_rocm_env_variant}" >&2
                exit 1
            fi
        else
            _rocm_variant="${_rocm_env_variant}"
        fi
        if [[ -z "${_rocm_variant}" ]]; then
            echo "ERROR: Could not detect ROCm variant from the environment for commit ${merge_base_commit}" >&2
            exit 1
        fi
        if [[ -z "${_available_variants}" ]] \
                || [[ " ${_available_variants} " != *" ${_rocm_variant} "* ]]; then
            echo "ERROR: Environment ROCm variant '${_rocm_variant}' is not published for commit ${merge_base_commit} (available:${_available_variants:-none})" >&2
            exit 1
        fi
        meta_json_url="https://wheels.vllm.ai/rocm/${merge_base_commit}/${_rocm_variant}/vllm/metadata.json"
        unset -v _rocm_env_variant _available_variants _rocm_variant
    else
        meta_json_url="https://wheels.vllm.ai/${merge_base_commit}/vllm/metadata.json"
    fi
    echo "INFO: will use metadata.json from ${meta_json_url}"

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

# ROCm CI uses setuptools develop for editable installs (see Dockerfile.rocm and run-amd-test.sh).
if [[ -n "${rocm_wheel}" ]]; then
    VLLM_PRECOMPILED_WHEEL_LOCATION="${rocm_wheel}" VLLM_USE_PRECOMPILED=1 python3 setup.py develop --no-deps
elif [[ "${is_rocm}" == "1" ]]; then
    VLLM_PRECOMPILED_WHEEL_COMMIT=$merge_base_commit VLLM_USE_PRECOMPILED=1 python3 setup.py develop --no-deps
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
