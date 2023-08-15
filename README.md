# pytorch_bash_container

A Docker container for PyTorch development with a Bash shell.

## Usage

1. Build the container:

    ```bash
    docker build -t pytorch_bash_container .
    ```

2. Run the container in detached mode:

    ```bash
    docker run -it -d -p 5678:5678 --name pytorch_bash_container -v "$(pwd)":/workspace pytorch_bash_container
    ```

3. Enter into it with Bash:

    ```bash
    docker exec -it pytorch_bash_container /bin/bash
    ```

4. Install packages:

    ```bash
    pip install -e .
    ```

5. Launch Server

    ```bash
    python -m vllm.entrypoints.api_server --model JosephusCheung/Guanaco --swap-space 16 --disable-log-requests --host 127.0.0.1 --port 8080
    ```
6. Trigger Benchmarking

    ```bash
    python benchmarks/benchmark_serving.py --backend vllm --tokenizer JosephusCheung/Guanaco --dataset ShareGPT_V3_unfiltered_cleaned_split.json --request-rate 5 --num-prompts 100 --host 127 0.0.1 --port 8080
    ```