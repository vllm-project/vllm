# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4.1 on Ampere datacenter GPUs (A100, SM80).

The stock DeepSeek-V4.1 path needs kernels that A100 lacks: FlashMLA and
DeepGEMM build for SM90 and SM100 only, and Triton cannot convert to or from
FP8 E4M3 below SM89. This package replaces each of them with a CuTe DSL kernel
that runs on SM80, keeping the stock model, scheduler and cache layout around
them:

* the KV caches keep DeepSeek's V4.1 records -- the 528-byte MXFP8
  sliding-window record and the 288-byte NVFP4 compressed record -- and the
  68-byte MXFP4 index keys, as on SM100;
* every FP8 and FP4 value is decoded with integer operations into BF16, where
  it is exact, and multiplied on BF16 tensor cores (`mma.sync`) with FP32
  accumulation; quantizers round to nearest even exactly as the hardware
  conversions of SM100 would.

The modules:

| Module | Contents |
| --- | --- |
| `attention.py` | The attention layer, its compressor and its metadata backends |
| `indexer.py` | The indexer: MXFP4 queries and keys, its op (logits, top-k) |
| `indexer_metadata.py` | The indexer metadata builder (no DeepGEMM schedule) |
| `sparse_mla.py` | The sparse MLA kernels |
| `mqa_logits.py` | The indexer logits kernel |
| `topk.py` | The indexer's decode top-k and candidate-block selection |
| `quantize.py` | The FP4 quantizers (queries, index keys, compressed records) |
| `mhc.py` | The mHC layer boundary: post-mix, collapse + RMSNorm, coefficients |
| `all_reduce.py` | The boundaries' two-shot Lamport all-reduce |
| `all_gather.py` | The Lamport all-gather (indexer rows; logits and Engram columns) |
| `columns.py` | The logits and Engram gathers along the last dimension |
| `codecs.py` | Device helpers: FP8/FP4 decoding and encoding, raw loads/stores |

Each kernel is specified by a PyTorch function of
`tests/models/deepseek_v41/nvidia/specialized_decode/ampere/reference.py`,
which the docstrings call `reference.<function>`.

This module holds only the platform predicate, the index K cache choice and
a table helper, so that shared DeepSeek-V4.1 code can ask for the Ampere path
without importing CuTe DSL.
"""

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.platforms import current_platform

# Compute capability of the GPUs this package targets: A100 and A800 (SM80).
# Their 164 KB of shared memory per SM sizes the attention kernel's tiles;
# other SM8x parts have less.
AMPERE_CAPABILITY = 80


def is_ampere() -> bool:
    """Whether the current device runs DeepSeek-V4.1 through this package."""
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return capability is not None and capability.to_int() == AMPERE_CAPABILITY


def indexer_uses_fp4(vllm_config: VllmConfig) -> bool:
    """The index K cache on SM80: always MXFP4, the only one its kernels
    write and score. An explicit FP8 request is rejected."""
    kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("mxfp4")
    if kv_dtype != "mxfp4":
        raise ValueError(
            "DeepSeek-V4.1 on SM80 scores the MXFP4 indexer cache only; "
            f"got indexer_kv_dtype={kv_dtype!r}."
        )
    return True


def rope_table_f32(rotary_emb: nn.Module) -> torch.Tensor:
    """The RoPE (cos, sin) table `[positions, 64]` in FP32, which the
    kernels read; a table of another dtype is converted once."""
    table = rotary_emb.cos_sin_cache
    assert isinstance(table, torch.Tensor)
    if table.dtype == torch.float32:
        return table
    converted = getattr(rotary_emb, "_ampere_cos_sin_f32", None)
    if converted is None:
        converted = table.float()
        rotary_emb._ampere_cos_sin_f32 = converted
    return converted
