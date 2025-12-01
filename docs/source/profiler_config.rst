========================================
Reducing PyTorch Profiler Overhead
========================================

The built-in PyTorch profiler can add 50–70s overhead on A100/H100 due to gzip compression and CUDA time dumps.

Quick Workaround (Fork & Modify)
=================================

Edit `vllm/profiler/gpu_profiler.py` (~line 80):

.. code-block:: diff

   - use_gzip=True,
   + use_gzip=False,
   - dump_self_cuda_time_total=True
   + dump_self_cuda_time_total=False

Run server:
.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m

Benchmarks (Nov 30, 2025, T4, opt-125m)
=======================================

- Default: ~18s trace time
- Modified: ~4s trace time (78% faster)

Native env var support tracked in #29564.

.. include:: ../../CONTRIBUTING.rst
