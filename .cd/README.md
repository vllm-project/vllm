# vLLM for Gaudi – Quick Start

This guide explains how to quickly run vLLM with multi-model support on Gaudi using a prebuilt Docker image.

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
| mistralai/Mixtral-8x22B-Instruct-v0.1     | 4 |
| mistralai/Mixtral-8x7B-Instruct-v0.1      | 2 |
| Qwen/Qwen2.5-14B-Instruct                 | 1 |
| Qwen/Qwen2.5-32B-Instruct                 | 1 |
| Qwen/Qwen2.5-72B-Instruct                 | 4 |
| Qwen/Qwen2.5-7B-Instruct                  | 1 |
| meta-llama/Llama-3.2-11B-Vision-Instruct  | 1 |
| meta-llama/Llama-3.2-90B-Vision-Instruct  | 4 |

## How to Use

1. **Use the prebuilt vLLM container**

   You do **not** need to build the Docker image yourself.  
   Use the ready-to-use image from an image registry:

   ```bash
   docker pull <path to a docker image>
   ```

2. **Set required environment variables**

   - `export MODEL=` (choose from the table above)
   - `export HF_TOKEN=` (your huggingface token, can be generated from https://huggingface.co)

   Tips:
   - Model files can be large. For best performance, use an external disk for the Huggingface cache and set `HF_HOME` accordingly.  
   Example: `-e HF_HOME=/mnt/huggingface -v /mnt/huggingface:/mnt`\
   - For a quick startup and to skip the initial model warmup (useful for development testing), you can add:  
   `-e VLLM_SKIP_WARMUP=true`

3. **Run the vLLM server**

   ```bash
   docker run -it --rm \
     -e MODEL=$MODEL \
     -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
     --cap-add=sys_nice \
     --ipc=host \
     --runtime=habana \
     -e HF_TOKEN=$HF_TOKEN \
     -e HABANA_VISIBLE_DEVICES=all \
     -p 8000:8000 \
     --name vllm-server \
     <docker image name>
   ```

4. **(Optional) Test the server**

   In a separate terminal:

   ```bash
   MODEL= # choose from the table above
   target=localhost
   curl_query="What is DeepLearning?"
   payload="{ \"model\": \"${MODEL}\", \"prompt\": \"${curl_query}\", \"max_tokens\": 128, \"temperature\": 0 }"
   curl -s --noproxy '*' http://${target}:8000/v1/completions -H 'Content-Type: application/json' -d "$payload"
   ```

5. **Customizing server parameters**

   You can override defaults with additional `-e` variables, for example:

   ```bash
   docker run -it --rm \
     -e MODEL=$MODEL \
     -e TENSOR_PARALLEL_SIZE=8 \
     -e MAX_MODEL_LEN=8192 \
     -e HABANA_VISIBLE_DEVICES=all \
     -e HF_TOKEN=$HF_TOKEN \
     -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
     --runtime=habana \
     --cap-add=sys_nice \
     --ipc=host \
     -p 8000:8000 \
     --name vllm-server \
     <docker image name>
   ```

6. **Running multiple instances**

   Each instance should have unique values for `HABANA_VISIBLE_DEVICES`, host port, and container name.  
   See [docs.habana.ai - Multiple Tenants](https://docs.habana.ai/en/latest/Orchestration/Multiple_Tenants_on_HPU/Multiple_Dockers_each_with_Single_Workload.html) for details.

   Example for two instances:

   ```bash
   # Instance 1
   docker run -it --rm \
     -e MODEL=$MODEL \
     -e TENSOR_PARALLEL_SIZE=4 \
     -e HABANA_VISIBLE_DEVICES=0,1,2,3 \
     -e MAX_MODEL_LEN=8192 \
     -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
     --runtime=habana \
     --cap-add=sys_nice \
     --ipc=host \
     -p 8000:8000 \
     --name vllm-server1 \
     <docker image name>

   # Instance 2 (in another terminal)
   docker run -it --rm \
     -e MODEL=$MODEL \
     -e TENSOR_PARALLEL_SIZE=4 \
     -e HABANA_VISIBLE_DEVICES=4,5,6,7 \
     -e MAX_MODEL_LEN=8192 \
     -e http_proxy=$http_proxy -e https_proxy=$https_proxy -e no_proxy=$no_proxy \
     --runtime=habana \
     --cap-add=sys_nice \
     --ipc=host \
     -p 9222:8000 \
     --name vllm-server2 \
     <docker image name>
   ```

7. **Viewing logs**

   ```bash
   docker logs -f vllm-server
   ```
