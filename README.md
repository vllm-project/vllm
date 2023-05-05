# CacheFlow

## Installation

```bash
pip install ninja psutil numpy sentencepiece ray torch transformers xformers
pip install -e .
```

## Test simple server

```bash
ray start --head
python simple_server.py
```

The detailed arguments for `simple_server.py` can be found by:
```bash
python simple_server.py --help
```

## FastAPI server

Install the following additional dependencies:
```bash
pip install fastapi uvicorn
```

To start the server:
```bash
ray start --head
python -m cacheflow.http_frontend.fastapi_frontend
```

To test the server:
```bash
python -m cacheflow.http_frontend.test_cli_client
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
    Please make sure that `llama` is included in the output directory name.
2. For all the commands above, specify the model with `--model /output/path/llama-7b` to load the model. For example:
    ```bash
    python simple_server.py --model /output/path/llama-7b
    python -m cacheflow.http_frontend.fastapi_frontend --model /output/path/llama-7b
    ```
