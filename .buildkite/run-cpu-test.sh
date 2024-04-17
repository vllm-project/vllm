# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Try building the docker image
docker build -t cpu-test -f Dockerfile.cpu .

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test || true; }
trap remove_docker_container EXIT
remove_docker_container

# Run the image and launch offline inference
<<<<<<< HEAD
docker run --network host --env VLLM_CPU_KVCACHE_SPACE=1 --name cpu-test cpu-test python3 vllm/examples/offline_inference.py
=======
docker run -itd --network host --env VLLM_CPU_KVCACHE_SPACE=4 --name cpu-test cpu-test

# offline inference
docker exec cpu-test bash -c "python3 examples/offline_inference.py"

# async engine test
#docker exec cpu-test bash -c "cd tests; pytest -v -s async_engine"

# Run basic model test
docker exec cpu-test bash -c "cd tests;
  pip install pytest Pillow
  rm -f __init__.py
  sed -i '/*stablelm-3b-4e1t/d' models/test_models.py
  sed -i '/torch.cuda.empty_cache/d' conftest.py
  sed -i 's/cuda/cpu/g' conftest.py
  bash ../.buildkite/download-images.sh
  pytest -v -s models --ignore=models/test_llava.py --ignore=models/test_mistral.py --ignore=models/test_marlin.py"
>>>>>>> ca0870d3... enable more tests
