# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:
```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Creating requests from text file (`benchmark_serving.py`)

To create requests from a text file, use arguments `--dataset path/to/dataset --request-from-text`. This will make the benchmark repeatedly send the text file to the server, resulting in best case caching scenario. Example:
```
python benchmarks/benchmark_serving.py  --model huggyllama/llama-7b --dataset benchmarks/data/sonnet.txt --request-rate 2.5 --num-prompts 1000 --backend openai --endpoint /v1/completions --request-from-text
```
