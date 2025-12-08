================================
Reducing PyTorch Profiler Overhead
================================

The built-in PyTorch profiler can add 50–70 s overhead on A100/H100 due to gzip compression and CUDA time dumps.

Quick Workaround (Fork & Modify)
================================

Edit ``vllm/profiler/gpu_profiler.py`` (~line 80):

.. code-block:: diff

   - use_gzip=True,
   + use_gzip=False,
   - dump_self_cuda_time_total=True
   + dump_self_cuda_time_total=False

Run the server:

.. code-block:: bash

    python -m vllm.entrypoints.openai.api_server --model facebook/opt-125m

Benchmarks (Dec 1, 2025, T4 GPU, opt-125m)
==========================================

- Default: ~18 s trace time
- Modified: ~4 s trace time (78 % faster)

Native env-var support is tracked in #29564.

.. include:: ../../CONTRIBUTING.rst
