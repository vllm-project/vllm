# CacheFlow

## Installation

```bash
pip install psutil numpy ray torch
pip install git+https://github.com/huggingface/transformers  # Required for LLaMA.
pip install sentencepiece  # Required for LlamaTokenizer.
pip install flash-attn  # This may take up to 20 mins.
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
