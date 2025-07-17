# vLLM for Gaudi – Quick Start

This guide explains how to quickly run vLLM on Gaudi using a prebuilt Docker image and Docker Compose, with options for custom parameters and benchmarking.
Supports a wide range of validated models including LLaMa, Mistral, and Qwen families, with flexible configuration via environment variables or YAML files.

## Supported Models

| Model Name | Validated TP Size |
|--|--|
| deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 8 |
| meta-llama/Llama-3.1-70B-Instruct         | 4 |
| meta-llama/Llama-3.1-405B-Instruct        | 8 |
| meta-llama/Llama-3.1-8B-Instruct          | 1 |
| meta-llama/Llama-3.2-1B-Instruct          | 1 |
| meta-llama/Llama-3.2-3B-Instruct          | 1 |
| meta-llama/Llama-3.3-70B-Instruct         | 4 |
| mistralai/Mistral-7B-Instruct-v0.2        | 1 |
| mistralai/Mixtral-8x7B-Instruct-v0.1      | 2 |
| mistralai/Mixtral-8x22B-Instruct-v0.1     | 4 |
| Qwen/Qwen2.5-7B-Instruct                  | 1 |
| Qwen/Qwen2.5-VL-7B-Instruct               | 1 |
| Qwen/Qwen2.5-14B-Instruct                 | 1 |
| Qwen/Qwen2.5-32B-Instruct                 | 1 |
| Qwen/Qwen2.5-72B-Instruct                 | 4 |
| meta-llama/Llama-3.2-11B-Vision-Instruct  | 1 |
| meta-llama/Llama-3.2-90B-Vision-Instruct  | 4 |
| ibm-granite/granite-8b-code-instruct-4k   | 1 |
| ibm-granite/granite-20b-code-instruct-8k  | 1 |

## How to Use

### 0. Clone the Repository

Before proceeding with any of the steps below, make sure to clone the vLLM fork repository and navigate to the `.cd` directory. This ensures you have all necessary files and scripts for running the server or benchmarks.

```bash
git clone https://github.com/HabanaAI/vllm-fork.git
cd vllm-fork/.cd/
```

### 1. Run the server using Docker Compose

   The recommended and easiest way to start the vLLM server is with Docker Compose. At a minimum, set the following environment variables:

   - `MODEL` - Select a model from the table above.
   - `HF_TOKEN` - Your Hugging Face token (generate one at <https://huggingface.co>).
   - `DOCKER_IMAGE` - The vLLM Docker image URL from Gaudi or local repository.

   **Example usage:**

   ```bash
   MODEL="Qwen/Qwen2.5-14B-Instruct" \
   HF_TOKEN="<your huggingface token>" \
   DOCKER_IMAGE="<docker image url>" \
   docker compose up
   ```

### 2. Running the Server with a Benchmark

   To easily initiate benchmark dedicated for a specific model using default parameters, use the `--profile benchmark up` option with Docker Compose:

   ```bash
   MODEL="Qwen/Qwen2.5-14B-Instruct" \
   HF_TOKEN="<your huggingface token>" \
   DOCKER_IMAGE="<docker image url>" \
   docker compose --profile benchmark up
   ```

   This launches the vLLM server and runs the benchmark suite automatically.

### 3. Run the server using Docker Compose with custom parameters

   To override default settings, you can provide additional parameters when starting the server. This is a more advanced approach:

   - `PT_HPU_LAZY_MODE` - Enables lazy execution mode for HPU (Habana Processing Unit), which may improve performance by batching operations.
   - `VLLM_SKIP_WARMUP` - If enabled, skips the model warmup phase, which can reduce startup time but may affect initial performance.
   - `MAX_MODEL_LEN` - Specifies the maximum sequence length the model can handle.
   - `MAX_NUM_SEQS` - Sets the maximum number of sequences that can be processed simultaneously.
   - `TENSOR_PARALLEL_SIZE` - Defines the number of parallel tensor partitions.
   - `VLLM_EXPONENTIAL_BUCKETING` - Controls enabling/disabling of exponential bucketing warmup strategy.
   - `VLLM_DECODE_BLOCK_BUCKET_STEP` - Sets the step size for allocating decode blocks during inference, affecting memory allocation granularity.
   - `VLLM_DECODE_BS_BUCKET_STEP` - Determines the batch size step for decode operations, influencing how batches are grouped and processed.
   - `VLLM_PROMPT_BS_BUCKET_STEP` - Sets the batch size step for prompt processing, impacting how prompt batches are handled.
   - `VLLM_PROMPT_SEQ_BUCKET_STEP` - Controls the step size for prompt sequence allocation, affecting how sequences are bucketed for processing.

   **Example usage:**

   ```bash
   MODEL="Qwen/Qwen2.5-14B-Instruct" \
   HF_TOKEN="<your huggingface token>" \
   DOCKER_IMAGE="<docker image url>" \
   TENSOR_PARALLEL_SIZE=1 \
   MAX_MODEL_LEN=2048 \
   docker compose up
   ```

### 4. Running the Server and Benchmark with Custom Parameters

   You can customize benchmark parameters using:

   - `INPUT_TOK` – Number of input tokens per prompt.
   - `OUTPUT_TOK` – Number of output tokens to generate per prompt.
   - `CON_REQ` – Number of concurrent requests to send during benchmarking.
   - `NUM_PROMPTS` – Total number of prompts to use in the benchmark.

   **Example usage:**

   ```bash
   MODEL="Qwen/Qwen2.5-14B-Instruct" \
   HF_TOKEN="<your huggingface token>" \
   DOCKER_IMAGE="<docker image url>" \
   INPUT_TOK=128 \
   OUTPUT_TOK=128 \
   CON_REQ=16 \
   NUM_PROMPTS=64 \
   docker compose --profile benchmark up
   ```

   This will launch the vLLM server and run the benchmark suite using your specified parameters.

### 5. Running the Server and Benchmark, both with Custom Parameters

   You can launch the vLLM server and benchmark together, specifying any combination of optional parameters for both the server and the benchmark. Set the desired environment variables before running Docker Compose.

   **Example usage:**

   ```bash
   MODEL="Qwen/Qwen2.5-14B-Instruct" \
   HF_TOKEN="<your huggingface token>" \
   DOCKER_IMAGE="<docker image url>" \
   VTENSOR_PARALLEL_SIZE=1 \
   MAX_MODEL_LEN=2048 \
   INPUT_TOK=128 \
   OUTPUT_TOK=128 \
   CON_REQ=16 \
   NUM_PROMPTS=64 \
   docker compose --profile benchmark up
   ```

   This command will start the vLLM server and run the benchmark suite using your specified custom parameters.

### 6. Running the Server and Benchmark Using Configuration Files

   You can also configure the server and benchmark by specifying parameters in configuration files. To do this, set the following environment variables:

   - `VLLM_SERVER_CONFIG_FILE` – Path to the server configuration file inside the Docker container.
   - `VLLM_SERVER_CONFIG_NAME` – Name of the server configuration section.
   - `VLLM_BENCHMARK_CONFIG_FILE` – Path to the benchmark configuration file inside the Docker container.
   - `VLLM_BENCHMARK_CONFIG_NAME` – Name of the benchmark configuration section.

   **Example:**

   ```bash
   HF_TOKEN=<your huggingface token> \
   VLLM_SERVER_CONFIG_FILE=server_configurations/server_text.yaml \
   VLLM_SERVER_CONFIG_NAME=llama31_8b_instruct \
   VLLM_BENCHMARK_CONFIG_FILE=benchmark_configurations/benchmark_text.yaml \
   VLLM_BENCHMARK_CONFIG_NAME=llama31_8b_instruct \
   docker compose --profile benchmark up
   ```

   > [!NOTE]
   > When using configuration files, you do not need to set the `MODEL` environment variable, as the model name is specified within the configuration file. However, you must still provide your `HF_TOKEN`.

### 7. Running the Server Directly with Docker

   For full control, you can run the server using the `docker run` command. This approach allows you to specify any native Docker parameters as needed.

   **Example:**

   ```bash
   docker run -it --rm \
     -e MODEL=$MODEL \
     -e HF_TOKEN=$HF_TOKEN \
     -e http_proxy=$http_proxy \
     -e https_proxy=$https_proxy \
     -e no_proxy=$no_proxy \
     --cap-add=sys_nice \
     --ipc=host \
     --runtime=habana \
     -e HABANA_VISIBLE_DEVICES=all \
     -p 8000:8000 \
     --name vllm-server \
     <docker image name>
   ```

   This method gives you full flexibility over Docker runtime options.
