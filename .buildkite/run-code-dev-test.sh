# This script check the correctness of python_only_dev.py for code development use

set -ex
BUILDKITE_COMMIT=$BUILDKITE_COMMIT

# Assume this test is in vllm ci test image which already install vllm
# Uninstall first for testing
pip uninstall -y vllm

# Test directory
TEST_DIR=/tmp/vllm_test
mkdir -p ${TEST_DIR}

# Install vllm dev wheel
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${BUILDKITE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl && \
pip install setuptools-scm # Hack to generate __version.py

# Test python_only_dev.py
cd ${TEST_DIR} && \
git clone https://github.com/vllm-project/vllm.git --no-checkout && \
cd vllm && \
git checkout ${BUILDKITE_COMMIT} && \
python3 python_only_dev.py && \
cd / && \
python3 -c "import vllm; print(vllm.__file__)"

# Test code change
cd ${TEST_DIR}/vllm && \
echo "test_var=123456" > vllm/fake_module.py && \
cd / && \
python3 -c "import vllm.fake_module; print(vllm.fake_module.test_var)"

# Test uninstall/updateing condition
cd ${TEST_DIR}/vllm && \
python3 python_only_dev.py --quit-dev && \
cd / && \
pip uninstall -y vllm && \
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${BUILDKITE_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl && \
pip show vllm
