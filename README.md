# CacheFlow

## Installation

```bash
pip install psutil numpy torch transformers
pip install flash-attn # This may take up to 10 mins.
pip install -e .
```

## Run

```bash
ray start --head
python server.py [--tensor-parallel-size <N>]
```
