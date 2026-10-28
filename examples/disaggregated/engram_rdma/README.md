# Experimental remote Engram adapter

This example illustrates the vLLM-side row adapter used in the accompanying
research measurements. It is **not a standalone RDMA implementation**. The
adapter currently imports an external `ReaderRuntime` and checkpoint inspector
backed by a legacy NIXL/GPUNETIO GPU-GET shim. Neither is supplied by current
public NIXL; [NIXL #2544](https://github.com/ai-dynamo/nixl/pull/2544) tracks
the proposed supported interface. Do not use this example as a production
provider or assume it works with an ordinary NIXL wheel.

The self-contained GPU test replaces that external reader with a fake and checks
row layout, masking, int32 hash-ID conversion, strided output, and retention on
failed retirement. With PyTorch, Triton and a CUDA GPU installed:

```bash
.venv/bin/python examples/disaggregated/engram_rdma/test_provider_gpu.py
```

This test does not perform RDMA. A separate, explicitly unsupported
[legacy research snapshot](https://github.com/foraxe/nixl/tree/b5322e46f6e686ff16bdb3afb8a1a940a64e1163/research/vllm_engram_repro)
contains the matching owner, reader and GPU GET source. It builds with DOCA
3.1 and a fresh build passed synthetic cross-node READ and a small real vLLM
1P1D correctness/performance study. It is not evidence of current-SDK or
production support.
