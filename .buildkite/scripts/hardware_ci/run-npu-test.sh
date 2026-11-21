#!/bin/bash

# This script build the Ascend NPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Base ubuntu image with basic ascend development libraries and python installed,
# pinned by digest (was quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.12).
# Update via PR only when the check's runtime dependencies require it.
BASE_IMAGE="quay.io/ascend/cann@sha256:cfa76a9bcd84dd81c487d6d26edb8b95e2140525891009ccb6009e5cd2544d8e"

VLLM_ASCEND_REPO="https://github.com/vllm-project/vllm-ascend.git"
GH_PROXY_REWRITE="url.https://gh-proxy.test.osinfra.cn/https://github.com/.insteadOf=https://github.com/"

# Resolve vllm-ascend main to an exact commit so each run checks out one fixed
# revision and the docker layer cache cannot reuse a checkout of another one.
VLLM_ASCEND_SHA=$(git -c "${GH_PROXY_REWRITE}" ls-remote "${VLLM_ASCEND_REPO}" refs/heads/main | cut -f1)
echo "vllm-ascend main @ ${VLLM_ASCEND_SHA}"

image_name="npu/vllm-ci:${BUILDKITE_COMMIT}_${EPOCHSECONDS}"
container_name="npu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards
agent_idx=$(echo "${BUILDKITE_AGENT_NAME}" | awk -F'-' '{print $(NF-1)}')
echo "agent_idx: ${agent_idx}"
builder_name="cachebuilder${agent_idx}"
builder_cache_dir="/mnt/docker-cache${agent_idx}"
mkdir -p "${builder_cache_dir}"

# Try building the docker image
cat <<EOF | DOCKER_BUILDKIT=1 docker build \
    --add-host cache-service-vllm.nginx-pypi-cache.svc.cluster.local:"${PYPI_CACHE_HOST}" \
    --builder "${builder_name}" --cache-from type=local,src="${builder_cache_dir}" \
                           --cache-to type=local,dest="${builder_cache_dir}",mode=max \
    --progress=plain --load -t "${image_name}" -f - .
FROM ${BASE_IMAGE}

# Define environments
ENV DEBIAN_FRONTEND=noninteractive
ENV SOC_VERSION="ascend910b1"

RUN pip config set global.index-url http://cache-service-vllm.nginx-pypi-cache.svc.cluster.local:${PYPI_CACHE_PORT}/pypi/simple && \
    pip config set global.trusted-host cache-service-vllm.nginx-pypi-cache.svc.cluster.local

# Install pytest from an inline hash-locked set; kept before COPY . . so the
# layer stays cacheable. Regenerate the hashes with:
#   uv pip compile --generate-hashes --no-annotate --python-version 3.12 \
#     --python-platform x86_64-manylinux_2_28 <<< 'pytest==<version>'
RUN --mount=type=cache,target=/root/.cache/pip \
    printf '%s\n' \
    'iniconfig==2.3.1 --hash=sha256:67f4b9c50da0dedf52af349e7749a80a9057a5031199791b906c3bb3ae878960 --hash=sha256:9121e2c1fdb355232495be3194c8dfe87ccc2d5dee45947b78e68f499790d7a7' \
    'packaging==26.3 --hash=sha256:94edc256424af38762eb31306eed28beb9f0efc50a8837492c9d6fd6004aed79 --hash=sha256:d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c' \
    'pluggy==1.7.0 --hash=sha256:7dd7b0d8832ba3cb632c306926ded123429211b83641b35dc5c41ad2d34f9bec --hash=sha256:d1eaa46ebb595891b860ab086b4d09c8588af65ebd4361b8e8f4bb8920b90ba8' \
    'pygments==2.21.0 --hash=sha256:2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9 --hash=sha256:610ca751c9bc2492b38eb9a38a7fbc93edbbb2d7182edaf34e66ae493dee5c8c' \
    'pytest==8.4.2 --hash=sha256:86c0d0b93306b961d58d62a4db4879f27fe25513d4b969df351abdddb3c30e01 --hash=sha256:872f880de3fc3a5bdc88a11b39c9710c3497a547cfa9320bc3c5e62fbf272e79' \
    > /tmp/pytest-lock.txt \
    && pip install --require-hashes -r /tmp/pytest-lock.txt

WORKDIR /workspace/vllm

COPY . .

# Install vllm-ascend, detached at the exact commit resolved above
WORKDIR /workspace
RUN git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/" && \
    git clone --depth 1 "${VLLM_ASCEND_REPO}" /workspace/vllm-ascend && \
    git -C /workspace/vllm-ascend checkout --detach "${VLLM_ASCEND_SHA}"

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
#   returns one argument per line: --device, /dev/davinciX, ...
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

    local i=0
    while (( i < cards_num )); do
        local dev_idx=$(((index - 1)*cards_num + i ))
        printf '%s\n' "--device"
        printf '%s\n' "/dev/davinci${dev_idx}"
        ((i++))
    done
}

mapfile -t device_args < <(parse_and_gen_devices "${BUILDKITE_AGENT_NAME}") || exit 1

# Run the image and execute the Out-Of-Tree (OOT) platform interface test case on Ascend NPU hardware.
# This test checks whether the OOT platform interface is functioning properly in conjunction with
# the hardware plugin vllm-ascend.
model_cache_dir=/mnt/modelscope${agent_idx}
mkdir -p "${model_cache_dir}"
docker run \
    "${device_args[@]}" \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v "${model_cache_dir}":/root/.cache/modelscope \
    --entrypoint="" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    pytest --noconftest -v -s tests/e2e/vllm_interface/test_vllm_pr_interface_compatibility.py
'
