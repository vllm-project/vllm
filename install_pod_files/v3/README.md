To deploy a pod with the vllm that we can edit into a pod to run experimetns do:

1- deploy the pod using vm-fm-vllm.yaml
2- you can connect to the pod using
>> oc exec -it pod/llama-3-8b-768cdff9f5-srt77 -- /bin/sh
3- inside the pod run this commands:

# uninstall vllm

pip uninstall -y vllm

# install vllm developer mode, I think that is where the vllm source is inside the image

git clone <git@github.com>:diegocastanibm/vllm.git
VLLM_USE_PRECOMPILED=1 pip install -e vllm

# Install benchmarch packages and run server and benchmark

1- Run in terminal 1:
>> VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.2-1B

or a 8B parameters using:
>> VLLM_LOGGING_LEVEL=DEBUG vllm serve meta-llama/Llama-3.1-8B-Instruct

2- Run in terminal 2:
>> python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10

or a 8B parameters using:
>> python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10

Also, there is a new version of using 
>> vllm bench serve --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10

If you need to download the dataset (shareGPT) do:
wget <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json>

Note:
If you have an error that pandas or dataset packages are missed, we need to run:
pip install vllm[bench]

# Install development environment (in pod, but pre-commit has been also installed in my mac)

1- Install UV:
>> curl -LsSf https://astral.sh/uv/install.sh | sh

2- New venv
>> uv venv --python 3.12 --seed
>> source .venv/bin/activate

3- In CUDA environments:
>> uv pip install -r requirements/common.txt -r requirements/dev.txt --torch-backend=auto

Linting, formatting and static type checking
>> pre-commit install

You can manually run pre-commit with
>> pre-commit run --all-files --show-diff-on-failure

To manually run something from CI that does not run
locally by default, you can run:
>> pre-commit run mypy-3.9 --hook-stage manual --all-files

Unit tests
>> pytest tests/

Run tests for a single test file with detailed output
>> pytest -s -v tests/test_logger.py

4- (In just MAC or local)
>> uv pip install pre-commit
>> pre-commit

5- To run just pre-commit with a hook do:
>> pre-commit run <hook_id> 
Check list in .pre-commit-config.yaml
