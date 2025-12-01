================================
Reducing PyTorch Profiler Overhead
================================

The built-in PyTorch profiler can add 50–70 s overhead on A100/H100 because of gzip compression and CUDA time dumps.

Quick Workaround (no code changes needed in main repo)
======================================================

Fork the repo and edit ``vllm/profiler/gpu_profiler.py`` (~line 80):

.. code-block:: diff

   - use_gzip=True,
   + use_gzip=False,
   - dump_self_cuda_time_total=True
   + dump_self_cuda_time_total=False

Benchmark (Nov 30 2025, T4, opt-125m)
====================================
- Default: ~18 s trace time
- Modified: ~4 s trace time (78 % faster)

Future native support
=====================
Env-var support is being tracked in #29564.

.. include:: ../../CONTRIBUTING.rst
