
# Pinning CPU Cores for improving vLLM Performance on GPUs

To improve memory-access coherence and release CPUs to other CPU-only workloads, such as vLLM serving with Llama3 8B, you can pin CPU cores based on different CPU Non-Uniform Memory Access (NUMA) nodes using the automatically generated `docker-compose.override.yml` file. The following procedure explains the process.

The Xeon processors currently validated for this setup are: Intel Xeon 6960P and Intel Xeon PLATINUM 8568Y+.

1. Install the required python libraries.

    ```bash
    pip install -r requirements_cpu_binding.txt
    ```

2. Pin CPU cores using the `docker-compose.override.yml` file.

    ```bash
    export MODEL="meta-llama/Llama-3.1-405B-Instruct"
    export HF_TOKEN="<your huggingface token>"
    python3 generate_cpu_binding_from_csv.py --settings cpu_binding_gnr.csv --output ./docker-compose.override.yml
    docker compose up
    ```

3. Specify the service name in `docker-compose.override.yml` to bind idle CPUs to another service, such as `vllm-cpu-service`, as in the following example:

    ```bash
    export MODEL="meta-llama/Llama-3.1-405B-Instruct"
    export HF_TOKEN="<your huggingface token>"
    python3 generate_cpu_binding_from_csv.py --settings cpu_binding_gnr.csv --output ./docker-compose.override.yml --cpuservice vllm-cpu-service
    docker compose -f docker-compose.yml -f docker-compose.vllm-cpu-service.yml -f docker-compose.override.yml up
    ```
