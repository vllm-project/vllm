# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 decode attention on Hopper datacenter GPUs (H100, H200, SM90).

The stock SM90 path decodes with FlashMLA's FP8 sparse kernel, which takes 64
or 128 query heads. Under tensor parallelism a rank holds 8 to 32 heads, so
the queries are zero-padded to 64 and most of the kernel's work goes to
padding; the call also plans its KV splits with a metadata kernel and merges
them with a combine kernel. Here decode tokens run the CuTe DSL sparse MLA of
the SM80 package (`ampere/sparse_mla.py`) instead, which takes the rank's
heads as they are and reads the 584-byte V4 records FlashMLA reads. On decode
steps the indexer's two-level selection bounds its candidate work by each
row's length: the candidate source ranks only the blocks below the row's
length (the specialized decode's cooperative selection) instead of the whole
`max_model_len` width, and the candidate consumers score only their
candidate keys, with the SM80 package's CuTe DSL logits and top-k on SM90's
FP8 index cache, instead of scoring the whole context and masking it. The
caches, DeepGEMM's logits of the other indexer calls, prefill (FlashMLA's
sparse prefill) and the FP8 output projection, which applies the inverse
RoPE, stay stock.

| Module | Contents |
| --- | --- |
| `attention.py` | The attention layer and its SWA metadata backend |
| `indexer.py` | The indexer layer with length-bounded candidate selection |

This module holds only the platform predicate, so that the model's attention
selection can ask for this path without importing CuTe DSL.
"""

from vllm.platforms import current_platform

# Compute capability of the GPUs this package targets (SM90).
HOPPER_CAPABILITY = 90


def is_hopper() -> bool:
    """Whether the current device is an SM90 GPU."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.to_int() == HOPPER_CAPABILITY
