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
[legacy research snapshot](https://github.com/foraxe/nixl/tree/dd70c9828b52b0d1bc19c30ad8966600a596cfc5/research/vllm_engram_repro)
contains the matching owner, reader and GPU GET source. It builds with DOCA
3.1 and has passed a synthetic cross-node READ; a full vLLM 1P1D run using
that freshly built snapshot remains to be verified. It is not evidence of
current-SDK support.
