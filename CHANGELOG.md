# Changelog

## [Unreleased]
- Restore `vllm_config.model_config.max_model_len` after running `estimate_max_model_len` so global configuration isn't left mutated following KV-cache sizing (2025-11-16).
- Increase `tests/v1/kv_offload/test_cpu_offloading.py`'s `gpu_memory_utilization` to 0.9 so the Llama-3.2 CPU-offload coverage passes on 12 GB GPUs that only provide ~2.3 GB of KV cache at the default (2025-11-16).
