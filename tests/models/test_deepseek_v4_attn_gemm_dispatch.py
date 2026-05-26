# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Regression guard for DSv4 attn_gemm_parallel_execute quant-aware dispatch.

Companion to ``test_deepseek_v4_quant_config_passthrough.py``: the
sibling file pins the constructor contract for the compressor's
``fused_wkv_wgate`` and the indexer's ``weights_proj`` / ``wq_b``;
this file pins the corresponding runtime invariant.

Once ``fused_wkv_wgate`` is built with a non-None ``quant_config`` (the
constructor fix in PR #43515 / its supersede), a later edit that
unconditionally calls ``torch.mm(hidden_states, fused.weight.T, ...)``
in ``DeepseekV4MultiHeadLatentAttentionWrapper.attn_gemm_parallel_execute``
would silently corrupt FP8 / W4A16 weights at runtime (packed bytes
multiplied as if they were a plain dense matrix).

The fix routes through ``module.forward`` when the linear's
``quant_method`` is not ``UnquantizedLinearMethod``, and only keeps the
fused FP32-accumulator ``torch.mm`` on the genuinely unquantized path
(so ``compressor.forward`` still receives a real FP32 result instead of
a BF16 accumulator widened after the fact).

This test is a static guard on the method's source: it pins that both
compressor closures (``compressor_kv_score`` for ``self.compressor``
and ``indexer_compressor_kv_score`` for ``indexer.compressor``) still
contain the ``UnquantizedLinearMethod`` check. A purely dynamic test
would require instantiating the full wrapper with all of its DSv4
plumbing (rotary, cache, MLA attention, ...); the static check catches
the regression vector that actually matters (the guard being silently
deleted by a later refactor) without that overhead.
"""

import inspect


def test_attn_gemm_parallel_execute_guards_quant_method_dispatch():
    from vllm.models.deepseek_v4.nvidia.ops.attention import (
        DeepseekV4MultiHeadLatentAttentionWrapper,
    )

    src = inspect.getsource(
        DeepseekV4MultiHeadLatentAttentionWrapper.attn_gemm_parallel_execute
    )

    # The conditional dispatch must check UnquantizedLinearMethod before
    # falling through to raw torch.mm on .weight.T. Without this guard,
    # quantized DSv4 checkpoints load OK (after PR #43515's constructor
    # fix) but produce garbage at runtime when the FP8 / W4A16 .weight
    # is multiplied as raw bytes.
    assert "UnquantizedLinearMethod" in src, (
        "Conditional quant dispatch missing in attn_gemm_parallel_execute. "
        "Without an UnquantizedLinearMethod guard, raw torch.mm on "
        "fused_wkv_wgate.weight.T silently corrupts FP8 / W4A16 weights. "
        "See the supersede PR description for the full story."
    )

    # Both compressor closures (self.compressor and indexer.compressor)
    # duplicate the conditional, so the guard should appear >= 2 times.
    occurrences = src.count("UnquantizedLinearMethod")
    assert occurrences >= 2, (
        "Both compressor_kv_score and indexer_compressor_kv_score must "
        "guard their torch.mm with an UnquantizedLinearMethod check, "
        f"got {occurrences} occurrence(s)."
    )
