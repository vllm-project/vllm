1-Run:
vllm serve meta-llama/Llama-3.2-1B --swap-space 16 --disable-log-requests

2-Run:
python3 vllm/benchmarks/benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-1B --endpoint /v1/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 10

If you need to download the dataset (shareGPT) do:
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
