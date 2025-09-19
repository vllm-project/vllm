#!/bin/bash

# This script build the Ascend NPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

image_name="npu/vllm-ci:${BUILDKITE_COMMIT}_${EPOCHSECONDS}"
container_name="npu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# Try building the docker image
cat <<EOF | DOCKER_BUILDKIT=1 docker build \
    --add-host cache-service-vllm.nginx-pypi-cache.svc.cluster.local:${PYPI_CACHE_HOST} \
    --builder cachebuilder --cache-from type=local,src=/mnt/docker-cache \
                           --cache-to type=local,dest=/mnt/docker-cache,mode=max \
    --progress=plain --load -t ${image_name} -f - .
FROM quay.io/ascend/cann:8.3.rc1.alpha002-910b-ubuntu22.04-py3.11

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
ARG VLLM_REPO=https://github.com/vllm-project/vllm-ascend.git
ARG VLLM_TAG=main
RUN git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/" && \
    git clone --depth 1 \$VLLM_REPO --branch \$VLLM_TAG /workspace/vllm-ascend

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
# Ascend NPU BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards
#   e.g. atlas-a2-001-0-2cards means this is the 0-th agent on atlas-a2-001 host, and it has 2 NPU cards.
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
    -v /root/.cache/modelscope:/root/.cache/modelscope \
    --entrypoint="" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    pytest -v -s tests/e2e/singlecard/test_sampler.py::test_models_topk
'
