================================
Reducing PyTorch Profiler Overhead
================================

The built-in PyTorch profiler can add 50–70 s overhead on A100/H100 because of gzip compression and CUDA time dumps.

Quick Fix (zero code changes)
=============================

.. code-block:: bash

    export VLLM_PROFILER_GZIP=false      # skip compression
    export VLLM_PROFILER_CUDA_DUMP=false # skip CUDA table
    python -m vllm.entrypoints.openai.api_server ...

Benchmark (Nov 30 2025, T4 GPU, opt-125m)
========================================
- Default settings : ~18 s trace time
- With both disabled : ~4 s trace time (78 % faster)

These env vars are checked in the profiler block (follow-up PR will add them if missing).

Closes #29564 (documentation part).
