# Pinning CPU Cores for improving vLLM Performance on GPUs

To improve memory-access coherence and release CPUs to other CPU-only workloads, such as vLLM serving with Llama3 8B, you can pin CPU cores based on different CPU Non-Uniform Memory Access (NUMA) nodes. Intel **Priority Core Turbo (PCT)** also helps GPU inference by ensuring the most latency-sensitive CPU threads (scheduler, tokenization, GPU feeding) run on **High-Priority (HP)** cores that can sustain **higher turbo frequencies**.
Using the automatically generated `docker-compose.override.yml` file can make sure vLLM uses higher frequencies cores and release idle CPUs to other CPU-only workloads.
The following procedure explains the process.

The Xeon processors currently validated for this setup are: Intel Xeon 6776P.

## 0. (Optional) Enable Priority Core Turbo

Follow [Enabling Priority Core Turbo](priority_core_turbo/README.md) to enable PCT on Intel® Xeon® 6 platforms.
After PCT is enabled, a CPU list file will be generated at:priority_core_turbo/results/clos0_cpulist.txt

## 1. Set up Environment

    Install the required python libraries.

    ```bash
    pip install -r requirements_cpu_binding.txt
    ```

## 2. Pin CPU cores for vLLM GPU Service

    Pin CPU cores using the docker-compose.override.yml file.

    ```bash
    export MODEL="meta-llama/Llama-3.1-405B-Instruct"
    export HF_TOKEN="<your huggingface token>"
    python3 generate_cpu_binding_from_csv.py --settings cpu_binding_gnr.csv --output ./docker-compose.override.yml
    docker compose up
    ```

## 3. Use Idle CPU Cores for Other CPU Workloads

    Specify the service name in `docker-compose.override.yml` to bind idle CPUs to another service, such as `vllm-cpu-service`, as in the following example:

    ```bash
    export MODEL="meta-llama/Llama-3.1-405B-Instruct"
    export HF_TOKEN="<your huggingface token>"
    python3 generate_cpu_binding_from_csv.py --settings cpu_binding_gnr.csv --output ./docker-compose.override.yml --cpuservice vllm-cpu-service
    docker compose -f docker-compose.yml -f docker-compose.vllm-cpu-service.yml -f docker-compose.override.yml up
    ```
