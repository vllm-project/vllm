# Benchmarking LLM Performance: Multi-Round QA Use Case

## Overview

This repository contains benchmarking tools for evaluating the performance of language models in various scenarios. The initial focus of this benchmark is on the multi-round QA (Question Answering) use case. The script `multi_round_qa.py` simulates multiple users interacting with a language model concurrently, allowing you to analyze the serving engine's throughput and latency.

### Current Workloads

- **Multi-Round QA Benchmark**: Simulates a realistic multi-user, multi-turn question-answering session to evaluate key metrics such as token throughput, latency, and average response times.


## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Multi-Round QA Benchmark

To run the multi-round QA benchmark, use the following command:

```bash
python3 multi_round_qa.py \
    --num-users 10 \
    --num-rounds 5 \
    --qps 0.5 \
    --shared-system-prompt 1000 \
    --user-history-prompt 2000 \
    --answer-len 100 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --base-url http://localhost:8000/v1
```

Use ctrl-C to terminate the benchmark at any time, and the the script will write each request's detailed stats to `summary.csv`.


*Note:* the above command requires there is a serving engine with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`. Here's an example command to launch the serving engine:

```bash
vllm serve mistralai/Mistral-7B-Instruct-v0.2 --disable-log-requests
```

### Arguments

#### Configuring the workload
- `--num-users <int>`: The maximum number of concurrent users in the system.
- `--num-rounds <int>`: The number of rounds per user.
- `--qps <float>`: The overall queries per second (QPS) rate for the system.
- `--shared-system-prompt <int>`: Length of the system prompt shared across all users (in tokens).
- `--user-history-prompt <int>`: Length of the user-specific context (simulating existing chat history) (in tokens).
- `--answer-len <int>`: Length of the answer expected (in tokens).
- `--init-user-id <int>`: The initial user ID to start the benchmark (default = 0). This is useful when you want to resume the benchmark from a specific user ID or avoid serving engine caching the request from previous runs
- `--request-with-user-id`: If this option is present, the script will include the user ID in the request header.
- `--sharegpt`: If this option is present, the script will use ShareGPT workload instead of dummy context.

*Note:* If you use ShareGPT dataset, the length of the answer expected (in tokens) will be determined by the min value of the dataset response and  `--answer-len`. You also need to follow the instructions in **ShareGPT Datasets** first.

#### Configuring the serving engine connection
- `--model <str>`: The model name (e.g., `mistralai/Mistral-7B-Instruct-v0.2`).
- `--base-url <str>`: The URL endpoint for the language model server.

#### Configuring the experiment (Optional)
- `--output <str>`: The csv file to dump the detailed stats for each query (default = summary.csv)
- `--log-interval <float>`: Time between each performance summary log in seconds (default = 30)
- `--time <float>`: Total time to run the experiment (default = forever)
- `--dry-run`: If this option is present, the script will not send requests to the endpoint (server). This option is useful when quickly verifying whether a script can properly process trace data.

#### Processing previous outputs only (Optional)
- `--process-summary <filename>`: if this option is present, the script will only process the existing output csv and print out the summary without running any experiment.

### Example Use Case

The above command starts a benchmark with 10 users engaging in 5 rounds of interaction, with an expected QPS of 0.5. It assumes there is already a serving engine (vLLM or lmcache\_vllm) with the `mistralai/Mistral-7B-Instruct-v0.2` model served locally at `http://localhost:8000/v1`.

Upon completion, a summary of key performance metrics (e.g., QPS, average response time) is printed to the console and saved as `summary.csv`.

## Understanding the Benchmark Script

The `multi_round_qa.py` script works by:

- Simulating multiple user sessions (`UserSessionManager`) which make requests (`UserSession`) to a specified language model concurrently.
- Tracking key metrics such as token throughput, time to first token (TTFT), and generation times.
- Printing a summary of the performance metrics periodically and writing the results to a CSV file at the end.

## Benchmark Metrics

- **Queries Per Second (QPS)**: The average number of queries processed by the model per second.
- **Average Prompt Throughput**: Tokens generated in the prompt per second.
- **Average Generation Throughput**: Tokens generated as part of the response per second.
- **Average TTFT (Time to First Token)**: Average time taken for the model to generate the first token of a response.

## ShareGPT Datasets

1. Download and prepare the ShareGPT dataset 
    You can easily download the ShareGPT dataset and perform the preparation step to remove invalid traces by running the script below.

    ```bash
    bash prepare_sharegpt_data.sh 1
    ```

    You can specify the proportion of data to process by providing a number between `0` and `1` as an argument to the script. In this example, `1` indicates processing 100% of the dataset. You can adjust this value as needed.

    Once the script runs successfully, `ShareGPT_V3_unfiltered_cleaned_split.json` will be downloaded, and the prepared `ShareGPT.json` will be generated.

    The `prepare_sharegpt_data.sh` script internally executes `data_preprocessing.py`, which provides the following options:

    - `--parse`: proportion of data to process by providing a number between `0` and `1` (default = 1)
    - `--model`: model name for tokenizer (default = `mistralai/Mistral-7B-Instruct-v0.2`)
    - `--trace`: trace file name to process (default = `ShareGPT_V3_unfiltered_cleaned_split.json`)

2. Run the benchmark
    Example:

    ```bash
    python3 multi_round_qa.py \
        --num-users 10 \
        --num-rounds 5 \
        --qps 0.3 \
        --shared-system-prompt 1000 \
        --user-history-prompt 2000 \
        --answer-len 100 \
        --model mistralai/Mistral-7B-Instruct-v0.2 \
        --base-url http://localhost:8000/v1 \
        --sharegpt
    ```

