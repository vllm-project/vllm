# This script check the correctness of python_only_dev.py for code development use

set -ex
BUILDKITE_COMMIT=$BUILDKITE_COMMIT

# Use docker from building wheel stage
DOCKER_BUILDKIT=1 docker build --build-arg max_jobs=16 --build-arg USE_SCCACHE=1 --tag vllm-ci:dev --target base --progress plain .
docker run -dit --entrypoint /bin/bash --privileged=true --network host --name vllm-dev-test vllm-ci:dev

# Install vllm dev wheel (latest nightly wheel)
docker exec vllm-dev-test bash -c "
    pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${BUILDKITE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl && \
    pip install setuptools-scm"

# Test python_only_dev.py
docker exec vllm-dev-test bash -c '
    git clone https://github.com/vllm-project/vllm.git --no-checkout && \
    cd vllm && \
    git checkout ${BUILDKITE_COMMIT} && \
    python3 python_only_dev.py && \
    cd / && \
    python3 -c "import vllm; print(vllm.__file__)"'

# Test code change
docker exec vllm-dev-test bash -c '
    cd vllm && \
    echo "test_var=123456" > vllm/fake_module.py && \
    cd / && \
    python3 -c "import vllm.fake_module; print(vllm.fake_module.test_var)"'

# Test uninstall/updateing condition
docker exec vllm-dev-test bash -c '
    cd vllm && \
    python3 python_only_dev.py --quit-dev && \
    cd / && \
    pip uninstall vllm && \
    pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${BUILDKITE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl'

docker rm -f vllm-dev-test