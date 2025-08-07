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

2- Run in terminal 2:
>> python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10

If you need to download the dataset (shareGPT) do:
wget <https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json>

Note:
If you have an error that pandas or dataset packages are missed, we need to run:
pip install vllm[bench]
