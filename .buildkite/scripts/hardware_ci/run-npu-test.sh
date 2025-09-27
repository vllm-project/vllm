#!/bin/bash

# This script build the Ascend NPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Base ubuntu image with basic ascend development libraries and python installed
VLLM_ASCEND_REPO="https://github.com/vllm-project/vllm-ascend.git"
CONFIG_FILE_REMOTE_PATH="tests/e2e/vllm_interface/vllm_test.cfg"
TEST_RUN_CONFIG_FILE="vllm_test.cfg"
VLLM_ASCEND_TMP_DIR=
# Get the test run configuration file from the vllm-ascend repository
fetch_vllm_test_cfg() {
    VLLM_ASCEND_TMP_DIR=$(mktemp -d)
    # Ensure that the temporary directory is cleaned up when an exception occurs during configuration file retrieval
    cleanup() {
        rm -rf "${VLLM_ASCEND_TMP_DIR}"
    }
    trap cleanup EXIT

    GIT_TRACE=1 git clone -v --depth 1 "${VLLM_ASCEND_REPO}" "${VLLM_ASCEND_TMP_DIR}"
    if [ ! -f "${VLLM_ASCEND_TMP_DIR}/${CONFIG_FILE_REMOTE_PATH}" ]; then
        echo "Error: file '${CONFIG_FILE_REMOTE_PATH}' does not exist in the warehouse" >&2
        exit 1
    fi

    # If the file already exists locally, just overwrite it
    cp "${VLLM_ASCEND_TMP_DIR}/${CONFIG_FILE_REMOTE_PATH}" "${TEST_RUN_CONFIG_FILE}"
    echo "Copied ${CONFIG_FILE_REMOTE_PATH} to ${TEST_RUN_CONFIG_FILE}"

    # Since the trap will be overwritten later, and when it is executed here, the task of cleaning up resources
    # when the trap is abnormal has been completed, so the temporary resources are manually deleted here.
    rm -rf "${VLLM_ASCEND_TMP_DIR}"
    trap - EXIT
}

# Downloads test run configuration file from a remote URL.
# Loads the configuration into the current script environment.
get_config() {
    if [ ! -f "${TEST_RUN_CONFIG_FILE}" ]; then
        echo "Error: file '${TEST_RUN_CONFIG_FILE}' does not exist in the warehouse" >&2
        exit 1
    fi
    source "${TEST_RUN_CONFIG_FILE}"
    echo "Base docker image name that get from configuration: ${BASE_IMAGE_NAME}"
    return 0
}

# get test running configuration.
fetch_vllm_test_cfg
get_config
# Check if the function call was successful. If not, exit the script.
if [ $? -ne 0 ]; then
  exit 1
fi

image_name="npu/vllm-ci:${BUILDKITE_COMMIT}_${EPOCHSECONDS}"
container_name="npu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards
agent_idx=$(echo "${BUILDKITE_AGENT_NAME}" | awk -F'-' '{print $(NF-1)}')
echo "agent_idx: ${agent_idx}"
builder_name="cachebuilder${agent_idx}"
builder_cache_dir="/mnt/docker-cache${agent_idx}"
mkdir -p ${builder_cache_dir}

# Try building the docker image
cat <<EOF | DOCKER_BUILDKIT=1 docker build \
    --add-host cache-service-vllm.nginx-pypi-cache.svc.cluster.local:${PYPI_CACHE_HOST} \
    --builder ${builder_name} --cache-from type=local,src=${builder_cache_dir} \
                           --cache-to type=local,dest=${builder_cache_dir},mode=max \
    --progress=plain --load -t ${image_name} -f - .
FROM ${BASE_IMAGE_NAME}

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url http://cache-service-vllm.nginx-pypi-cache.svc.cluster.local:${PYPI_CACHE_PORT}/pypi/simple && \
    pip config set global.trusted-host cache-service-vllm.nginx-pypi-cache.svc.cluster.local && \
    apt-get update -y && \
    apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

# Install for pytest to make the docker build cache layer always valid
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pytest>=6.0  modelscope

WORKDIR /workspace/vllm

# Install vLLM dependencies in advance. Effect: As long as common.txt remains unchanged, the docker cache layer will be valid.
COPY requirements/common.txt /workspace/vllm/requirements/common.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements/common.txt

COPY . .

# Install vLLM
RUN --mount=type=cache,target=/root/.cache/pip \
    VLLM_TARGET_DEVICE="empty" python3 -m pip install -v -e /workspace/vllm/ --extra-index https://download.pytorch.org/whl/cpu/ && \
    python3 -m pip uninstall -y triton

# Install vllm-ascend
WORKDIR /workspace
ARG VLLM_ASCEND_REPO=https://github.com/vllm-project/vllm-ascend.git
ARG VLLM_ASCEND_TAG=main
RUN git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/" && \
    git clone --depth 1 \$VLLM_ASCEND_REPO --branch \$VLLM_ASCEND_TAG /workspace/vllm-ascend

# Install vllm dependencies in advance. Effect: As long as common.txt remains unchanged, the docker cache layer will be valid.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /workspace/vllm-ascend/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    python3 -m pip install -v -e /workspace/vllm-ascend/ --extra-index https://download.pytorch.org/whl/cpu/

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_USE_MODELSCOPE=True

WORKDIR /workspace/vllm-ascend

CMD ["/bin/bash"]

EOF

# Setup cleanup
remove_docker_container() {
  docker rm -f "${container_name}" || true;
  docker image rm -f "${image_name}" || true;
  docker system prune -f || true;
}
trap remove_docker_container EXIT

# Generate corresponding --device args based on BUILDKITE_AGENT_NAME
# Ascend NPU BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards, and agent_idx starts from 1.
#   e.g. atlas-a2-001-1-2cards means this is the 1-th agent on atlas-a2-001 host, and it has 2 NPU cards.
#   returns --device /dev/davinci0 --device /dev/davinci1
parse_and_gen_devices() {
    local input="$1"
    local index cards_num
    if [[ "$input" =~ ([0-9]+)-([0-9]+)cards$ ]]; then
        index="${BASH_REMATCH[1]}"
        cards_num="${BASH_REMATCH[2]}"
    else
        echo "parse error" >&2
        return 1
    fi

    local devices=""
    local i=0
    while (( i < cards_num )); do
        local dev_idx=$(((index - 1)*cards_num + i ))
        devices="$devices --device /dev/davinci${dev_idx}"
        ((i++))
    done

    # trim leading space
    devices="${devices#"${devices%%[![:space:]]*}"}"
    # Output devices: assigned to the caller variable
    printf '%s' "$devices"
}

devices=$(parse_and_gen_devices "${BUILDKITE_AGENT_NAME}") || exit 1

# Run the image and execute the Out-Of-Tree (OOT) platform interface test case on Ascend NPU hardware.
# This test checks whether the OOT platform interface is functioning properly in conjunction with
# the hardware plugin vllm-ascend.
model_cache_dir=/mnt/modelscope${agent_idx}
mkdir -p ${model_cache_dir}
docker run \
    ${devices} \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v ${model_cache_dir}:/root/.cache/modelscope \
    --entrypoint="" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    pytest -v -s tests/e2e/vllm_interface/
'
