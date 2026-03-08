# Quick Start Cheat Sheet

## Installation

```bash
pip install vllm
```

## Basic Usage

### Offline Inference

```python
from vllm import LLM
llm = LLM("meta-llama/Llama-2-7b")
output = llm.generate("Hello")
```

### Online Serving

```bash
vllm serve meta-llama/Llama-2-7b
```

Signed-off-by: goingforstudying-ctrl <goingforstudying-ctrl@users.noreply.github.com>
