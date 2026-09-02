# Source and License

This skill was copied from NVIDIA's TensorRT-LLM repository:

- Source: `https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-triton-writing`
- Snapshot commit: `395985c025c8d1cf5aa842bc752b337ba88721b6`
- Copyright: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES.
  All rights reserved.
- License: Apache License 2.0

The NVIDIA copyright and Apache-2.0 SPDX notices are preserved in the adapted
reference files. The vLLM copy adds explicit source comments and removes
unsupported skill metadata.

The upstream standalone verification and benchmark scripts are omitted. vLLM
uses its existing parametrized kernel pytest suites for correctness and the
`kernel-microbenchmark` skill and `benchmarks/kernels/` for performance work.
The associated fixed-name export contract and workflow sections are adapted to
those vLLM conventions. Copied API catalogs, fixed tuning recipes, performance
claims, and incomplete kernel examples were removed after review because they
duplicated versioned Triton documentation or were not generally supportable.
The remaining guidance directs contributors to current official Triton
documentation and device-specific measurement.
