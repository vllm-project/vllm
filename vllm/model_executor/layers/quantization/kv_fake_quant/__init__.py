# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
KV-cache fake-quantization for accuracy studies.

Methods: BF16 / FP16 baselines, FP8 (per-token, group=128 default),
per-token int{2,4} (KIVI-style), SmoothKV (per-step rescale + int4),
SmoothKV-fused (load-time weight fold + plain pertoken int4).

The kernels run quant -> dequant in the same step. KV cache *storage* is
unchanged (still BF16); only the *values* are constrained to the chosen
grid. So this is an accuracy study, not a memory/speedup study.

Wired into vLLM via three direct calls. None are runtime monkey-patches:

    USER                                     OUR CODE                       UPSTREAM EDIT
    ────                                     ────────                       ─────────────
    LLM(kv_cache_quant_config=...)
      └─ stored on VllmConfig
                                                                            (none)
    Attention.__init__:
      └─ attach_kv_quant_to_layer ───────────> layer_hooks.attach_*         attention.py:380
                                                  │
                                                  ├─ reads VllmConfig
                                                  ├─ attaches LayerKVQuantState
                                                  └─ (smoothkv) loads calib

    Worker.load_model:
      └─ maybe_run_post_load_fusion ─────────> fusion.maybe_run_*           gpu_worker.py:323
                                                  │
                                                  └─ (smoothkv_fused only) folds
                                                     s_K/s_V into qkv_proj/o_proj
                                                     weights in place

    Attention.forward:
      └─ apply_kv_quant ─────────────────────> layer_hooks.apply_*          attention.py:472
                                                  │
                                                  └─ dispatches to kernels.* ops
                                                     based on layer.kv_quant_state.method

Public API (re-exported below):

    attach_kv_quant_to_layer(layer, prefix)         called from Attention.__init__
    apply_kv_quant(layer, key, value) -> (K, V)     called from Attention.forward
    maybe_run_post_load_fusion(model)               called from Worker.load_model

    LayerKVQuantState                               per-layer state (nn.Module)
    fake_quantize_fp8 / fake_quantize_pertoken      direct test/utility access
"""

from .kernels import fake_quantize_fp8, fake_quantize_pertoken
from .layer_hooks import (
    LayerKVQuantState,
    apply_kv_quant,
    attach_kv_quant_to_layer,
)
from .fusion import (
    fuse_smoothkv_into_model,
    maybe_run_post_load_fusion,
)

__all__ = [
    "attach_kv_quant_to_layer",
    "apply_kv_quant",
    "maybe_run_post_load_fusion",
    "fuse_smoothkv_into_model",
    "LayerKVQuantState",
    "fake_quantize_fp8",
    "fake_quantize_pertoken",
]
