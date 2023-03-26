# CacheFlow

## Installation

```bash
pip install psutil numpy ray torch
pip install git+https://github.com/huggingface/transformers  # Required for LLaMA.
pip install sentencepiece  # Required for LlamaTokenizer.
pip install flash-attn  # This may take up to 20 mins.
pip install -e .
```

## Run

```bash
ray start --head
python server.py [--tensor-parallel-size <N>]
```
