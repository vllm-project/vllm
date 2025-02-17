# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the ShareGPT dataset by running:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Downloading the ShareGPT4V dataset

The json file refers to several image datasets (coco, llava, etc.). The benchmark scripts
will ignore a datapoint if the referred image is missing.

```bash
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json
mkdir coco -p
wget http://images.cocodataset.org/zips/train2017.zip -O coco/train2017.zip
unzip coco/train2017.zip -d coco/
```

## Run the Benchmarking client

```bash
pip install datasets
python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
```

for example:

```bash
python3 benchmark_serving.py --backend vllm --model meta-llama/Llama-3.2-3B-Instruct --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --profile --num-prompts 2
```

## Example Output (as of `v0.7.1`)

```
============ Serving Benchmark Result ============
Successful requests:                     5
Benchmark duration (s):                  17.23
Total input tokens:                      404
Total generated tokens:                  1418
Request throughput (req/s):              0.29
Output token throughput (tok/s):         82.29
Total Token throughput (tok/s):          105.73
---------------Time to First Token----------------
Mean TTFT (ms):                          51.10
Median TTFT (ms):                        56.13
P99 TTFT (ms):                           57.34
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          23.17
Median TPOT (ms):                        23.32
P99 TPOT (ms):                           23.54
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.79
Median ITL (ms):                         22.72
P99 ITL (ms):                            27.37
==================================================
=======
# Downloading the BurstGPT dataset

You can download the BurstGPT v1.1 dataset by running:

```bash
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
```
