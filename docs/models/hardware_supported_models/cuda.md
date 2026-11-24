# CUDA - NVIDIA GPUs

## Validated Hardware

| Hardware |
| -------- |
| NVIDIA GeForce RTX 5060 Ti (16 GB, SM 120) |

## Recommended Models

The models below were validated on a single RTX 5060 Ti with 16 GB of VRAM,
`gpu_memory_utilization=0.9`, and `max_model_len=4096`.

### Text-only Language Models

| Model | Architecture | Dtype |
| ----- | ----- | ----- |
| Qwen/Qwen3-0.6B | Qwen3ForCausalLM | BF16 |
| Qwen/Qwen3-1.7B | Qwen3ForCausalLM | BF16 |
| Qwen/Qwen3-4B | Qwen3ForCausalLM | BF16 |
| Qwen/Qwen3-4B-AWQ | Qwen3ForCausalLM | AWQ |
| Qwen/Qwen3-8B-AWQ | Qwen3ForCausalLM | AWQ |
