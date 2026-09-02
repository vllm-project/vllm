# Source and License

This skill was copied from NVIDIA's TensorRT-LLM repository:

- Source: `https://github.com/NVIDIA/TensorRT-LLM/tree/main/.claude/skills/kernel-triton-writing`
- Snapshot commit: `395985c025c8d1cf5aa842bc752b337ba88721b6`
- Copyright: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES.
  All rights reserved.
- License: Apache License 2.0

The NVIDIA copyright and Apache-2.0 SPDX notices are preserved in the copied
reference and script files. The vLLM copy adds explicit source comments,
removes unsupported skill metadata, replaces unsafe cache-removal examples,
keeps upstream Markdown tables/code blocks with targeted lint suppressions,
and adapts paths and Python commands to vLLM's `.venv/bin/python` and `uv`
workflow. The benchmark helper uses vLLM's accelerator-neutral synchronization
API.
