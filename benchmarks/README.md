# Benchmarking vLLM

## Downloading the ShareGPT dataset

You can download the dataset by running:

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

# Downloading the BurstGPT dataset

You can download the BurstGPT v1.1 dataset by running:

```bash
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
```

# Downloading and using the LongBench dataset

To download the configuration for system prompts used for each dataset:

```
wget https://raw.githubusercontent.com/THUDM/LongBench/refs/heads/main/LongBench/config/dataset2prompt.json
```

To download the configuration for the max length limited for each dataset:

```
wget https://raw.githubusercontent.com/THUDM/LongBench/refs/heads/main/LongBench/config/dataset2maxlen.json
```

An example command to run benchmark_serving.py with LongBench dataset:

```
python benchmarks/benchmark_serving.py \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name longbench \
    --longbench-dataset-name narrativeqa \
    --dataset-to-prompt benchmarks/dataset2prompt.json \
    --dataset-to-maxlen benchmarks/dataset2maxlen.json \
    --max-input-len 128000 \
    --request-rate 0.2 \
    --num-prompts 200 
```
