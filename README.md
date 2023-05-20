# CacheFlow

## Build from source

```bash
pip install -r requirements.txt
pip install -e .  # This may take several minutes.
```

## Test simple server

```bash
# Single-GPU inference.
python examples/simple_server.py # --model <your_model>

# Multi-GPU inference (e.g., 2 GPUs).
ray start --head
python examples/simple_server.py -tp 2 # --model <your_model>
```

The detailed arguments for `simple_server.py` can be found by:
```bash
python examples/simple_server.py --help
```

## FastAPI server

To start the server:
```bash
ray start --head
python -m cacheflow.entrypoints.fastapi_server # --model <your_model>
```

To test the server:
```bash
python test_cli_client.py
```

## Gradio web server

Install the following additional dependencies:
```bash
pip install gradio
```

Start the server:
```bash
python -m cacheflow.http_frontend.fastapi_frontend
# At another terminal
python -m cacheflow.http_frontend.gradio_webserver
```

## Load LLaMA weights

Since LLaMA weight is not fully public, we cannot directly download the LLaMA weights from huggingface. Therefore, you need to follow the following process to load the LLaMA weights.

1. Converting LLaMA weights to huggingface format with [this script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).
    ```bash
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path/llama-7b
    ```
2. For all the commands above, specify the model with `--model /output/path/llama-7b` to load the model. For example:
    ```bash
    python simple_server.py --model /output/path/llama-7b
    python -m cacheflow.http_frontend.fastapi_frontend --model /output/path/llama-7b
    ```
