# MUSA Benchmarks

MUSA benchmarks compare the existing torch path with optional native
`musa_aiter` acceleration for the in-process vLLM MUSA connector.

Run on a Moore Threads MUSA host:

```bash
python -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.device_count())"
pytest -q tests/v1/test_musa_native.py tests/v1/test_musa_connector.py -rs
python benchmarks/musa/bench_inprocess_transfer.py --memory-device musa --iters 20 --warmup-iters 5 --min-speedup 1.2
```

The benchmark passes when native mode is at least `1.2x` faster than the torch
fallback for the default shape. Use `--memory-device musa` when measuring native
acceleration because the optional `musa_aiter` path consumes contiguous MUSA
tensors directly. For review, attach the full benchmark command, MUSA driver
version, `torch_musa` version, model-like shape, and speedup.
