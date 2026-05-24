# SPDX-License-Identifier: Apache-2.0
"""Genesis patches — thin bridges to upstream vLLM code.

These modules monkey-patch or runtime-inject our Genesis kernels into
upstream vLLM code paths. They are the "glue" between our professional
drop-in replacements (in `vllm._genesis.kernels`) and the actual vLLM
runtime.

Migration path:
  v5.14.1 → text-replacement patches in patch_genesis_unified.py (monolith)
  v6.0    → per-patch monkey-patch bridges in this package (transition)
  v7.0    → kernels submitted upstream → most patches retire (target)

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
