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

This test does not perform RDMA. End-to-end reproduction additionally needs a
public compatible GPU GET provider, table owner, and matching setup; these are
not yet available. The measured research stack used a legacy DOCA 3.1 runtime
and is not evidence of current-SDK support.
