# Source and License

This skill was copied from NVIDIA's TensorRT-LLM repository:

- Source: `https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-triton-writing`
- Snapshot commit: `395985c025c8d1cf5aa842bc752b337ba88721b6`
- Copyright: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES.
  All rights reserved.
- License: Apache License 2.0

The NVIDIA copyright and Apache-2.0 SPDX notices are preserved in the copied
reference files. The vLLM copy adds explicit source comments, removes
unsupported skill metadata, replaces unsafe cache-removal examples, and keeps
upstream Markdown tables/code blocks with targeted lint suppressions.

The upstream standalone verification and benchmark scripts are omitted. vLLM
uses its existing parametrized kernel pytest suites for correctness and the
`kernel-microbenchmark` skill and `benchmarks/kernels/` for performance work.
The associated fixed-name export contract and workflow sections are adapted to
those vLLM conventions.
