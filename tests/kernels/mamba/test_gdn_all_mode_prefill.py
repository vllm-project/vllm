# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-level tests for GDN all-mode prefill (cached-block load + per-block
SSM checkpoint scatter) through the REAL ``_forward_core``.

All-mode changes only WHERE states are cached, never WHAT is computed, so:

* an all-mode prefill must produce bit-identical core-attention output to the
  align-mode prefill on the same inputs, while additionally writing one SSM
  checkpoint per scheduled block (interior blocks from FLA's per-chunk state
  exports -- the +1-chunk start-of-chunk shift -- and the final block from the
  final recurrent state);
* continuing a sequence from a cached block boundary (warm) must match the
  suffix of the single-shot (cold) prefill: outputs and written states.

Metadata is built by the real ``GDNAttentionMetadataBuilder`` with
``mamba_cache_mode="all"`` so the builder's block-index computation is part of
what is being tested. The Triton/FLA prefill backend is forced: it is the only
GDN chunk backend that exports per-chunk states.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN all-mode prefill tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (  # noqa: E402
    MambaStateShapeCalculator,
)
from vllm.v1.attention.backends.gdn_attn import (  # noqa: E402
    GDNAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MambaSpec  # noqa: E402

H = 4  # num key heads
HV = 8  # num value heads
K = 128  # head_k_dim
V = 128  # head_v_dim
CONV_KERNEL = 4
KEY_DIM = H * K
VALUE_DIM = HV * V
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
PREFIX = "model.layers.0.linear_attn"


def _make_vllm_config(mamba_block_size: int, mamba_cache_mode: str):
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=mamba_block_size,
    )
    # The Triton/FLA chunk kernel is the only backend with per-chunk state
    # export (all-mode); force it for both modes so outputs are comparable.
    cfg.additional_config = {"gdn_prefill_backend": "triton"}
    cfg.cache_config.mamba_cache_mode = mamba_cache_mode
    cfg.cache_config.mamba_block_size = mamba_block_size
    return cfg


def _build_layer(vllm_config, mamba_block_size, conv_state, ssm_state, weights):
    """A minimal object that runs the real ``_forward_core`` bound to it."""
    A_log, dt_bias, conv_weight, conv_bias = weights
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.enable_packed_recurrent_decode = False
    layer.tp_size = 1
    layer.num_k_heads = H
    layer.num_v_heads = HV
    layer.head_k_dim = K
    layer.head_v_dim = V
    layer.key_dim = KEY_DIM
    layer.value_dim = VALUE_DIM
    layer.activation = "silu"
    layer.A_log = A_log
    layer.dt_bias = dt_bias
    layer.conv1d = types.SimpleNamespace(weight=conv_weight, bias=conv_bias)
    layer.kv_cache = (conv_state, ssm_state)
    layer.cache_config = types.SimpleNamespace(mamba_block_size=mamba_block_size)
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    assert layer.chunk_gated_delta_rule.gdn_prefill_backend == "triton"
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        setattr(
            layer,
            name,
            types.MethodType(getattr(QwenGatedDeltaNetAttention, name), layer),
        )
    return layer


def _build_metadata(vllm_config, batch, mamba_block_size, device):
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=mamba_block_size,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )
    common = create_common_attn_metadata(
        batch, mamba_block_size, device, arange_block_indices=True
    )
    # Block id 0 is the reserved NULL block (NULL_BLOCK_ID): the conv kernel
    # skips any sequence whose gathered state index is 0. Real block tables
    # never contain it; shift the synthetic arange table to match.
    common.block_table_tensor.add_(1)
    with set_current_vllm_config(vllm_config):
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
    return meta, common


def _run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens):
    core_attn_out = torch.zeros(
        num_tokens, HV, V, dtype=mixed_qkv.dtype, device=mixed_qkv.device
    )
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(
            mixed_qkv=mixed_qkv.clone(),
            b=b.clone(),
            a=a.clone(),
            core_attn_out=core_attn_out,
        )
    return core_attn_out


def _make_weights(device):
    A_log = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device) * 0.1
    conv_weight = (
        torch.randn(CONV_DIM, 1, CONV_KERNEL, dtype=torch.bfloat16, device=device) * 0.1
    )
    conv_bias = torch.randn(CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    return A_log, dt_bias, conv_weight, conv_bias


def _make_pools(pool_size, state_dtype, device):
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=0
    )
    conv_state = torch.zeros(
        pool_size, *conv_shape, dtype=torch.bfloat16, device=device
    )
    ssm_state = torch.zeros(pool_size, *ssm_shape, dtype=state_dtype, device=device)
    return conv_state, ssm_state


def _make_inputs(num_tokens, device):
    mixed_qkv = (
        torch.randn(num_tokens, CONV_DIM, dtype=torch.bfloat16, device=device) * 0.1
    )
    a = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(num_tokens, HV, dtype=torch.bfloat16, device=device) * 0.1
    return mixed_qkv, b, a


def test_all_mode_prefill_matches_align_and_writes_interior_checkpoints():
    """All-mode cold prefill: identical outputs to align mode; interior
    per-block checkpoints written (align leaves them untouched); final block
    state identical across modes."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    mamba_block_size = 64
    # Two fresh prefills; 320 = 5 blocks, 192 = 3 blocks.
    batch = BatchSpec(seq_lens=[320, 192], query_lens=[320, 192])
    num_tokens = 512
    weights = _make_weights(device)
    mixed_qkv, b, a = _make_inputs(num_tokens, device)

    outs, pools, metas = {}, {}, {}
    for mode in ("align", "all"):
        cfg = _make_vllm_config(mamba_block_size, mode)
        meta, common = _build_metadata(cfg, batch, mamba_block_size, device)
        pool_size = int(common.block_table_tensor.max().item()) + 1
        conv_state, ssm_state = _make_pools(pool_size, torch.float32, device)
        layer = _build_layer(cfg, mamba_block_size, conv_state, ssm_state, weights)
        outs[mode] = _run_forward_core(layer, meta, mixed_qkv, b, a, num_tokens)
        pools[mode] = (conv_state, ssm_state)
        metas[mode] = (meta, common)

    meta_all, common_all = metas["all"]
    assert meta_all.all_state_indices_tensor is not None
    # Same computation, different cache placement: outputs bit-identical.
    torch.testing.assert_close(outs["all"], outs["align"], atol=0, rtol=0)

    block_table = common_all.block_table_tensor
    ssm_align, ssm_all = pools["align"][1], pools["all"][1]
    # Final blocks (seq0: idx 4, seq1: idx 2) hold the final state in BOTH
    # modes (align writes it via its sliced table, all via the scatter).
    for seq, last_idx in ((0, 4), (1, 2)):
        blk = int(block_table[seq, last_idx].item())
        torch.testing.assert_close(ssm_all[blk], ssm_align[blk], atol=0, rtol=0)
    # Interior blocks: written only by all-mode.
    for seq, interior in ((0, (0, 1, 2, 3)), (1, (0, 1))):
        for j in interior:
            blk = int(block_table[seq, j].item())
            assert ssm_align[blk].abs().max().item() == 0.0
            assert ssm_all[blk].abs().max().item() > 0.0


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mamba_block_size", [64, 128])
def test_all_mode_cold_warm_parity(state_dtype, mamba_block_size):
    """The 8a acceptance: resuming from a cached block boundary reproduces the
    cold run's suffix outputs and block states (interior mapping exact for
    chunk-stride 1 and 2)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    total = 5 * mamba_block_size
    cached = 3 * mamba_block_size
    suffix = total - cached
    weights = _make_weights(device)
    mixed_qkv, b, a = _make_inputs(total, device)
    cfg = _make_vllm_config(mamba_block_size, "all")

    # Cold: single-shot full prefill.
    cold_batch = BatchSpec(seq_lens=[total], query_lens=[total])
    meta_cold, common_cold = _build_metadata(cfg, cold_batch, mamba_block_size, device)
    pool_size = int(common_cold.block_table_tensor.max().item()) + 1
    conv_state, ssm_state = _make_pools(pool_size, state_dtype, device)
    layer = _build_layer(cfg, mamba_block_size, conv_state, ssm_state, weights)
    out_cold = _run_forward_core(layer, meta_cold, mixed_qkv, b, a, total)
    ssm_cold = ssm_state.clone()

    # Warm: continue the last `suffix` tokens on the same pools, as if the
    # first `cached` tokens hit the prefix cache.
    warm_batch = BatchSpec(seq_lens=[total], query_lens=[suffix])
    meta_warm, common_warm = _build_metadata(cfg, warm_batch, mamba_block_size, device)
    torch.testing.assert_close(
        common_warm.block_table_tensor, common_cold.block_table_tensor
    )
    assert meta_warm.num_computed_tokens.tolist() == [cached]
    out_warm = _run_forward_core(
        layer, meta_warm, mixed_qkv[cached:], b[cached:], a[cached:], suffix
    )

    # Chunk-aligned resume follows the same accumulation order as the cold
    # suffix; the only divergence is the checkpoint's round-trip through the
    # pool dtype.
    if state_dtype == torch.float32:
        atol = rtol = 2e-3
    else:
        atol = rtol = 4e-2
    torch.testing.assert_close(out_warm, out_cold[cached:], atol=atol, rtol=rtol)

    # Re-written blocks (the ones at/after the resume point) match the cold
    # run's checkpoints.
    block_table = common_cold.block_table_tensor
    for j in range(cached // mamba_block_size, 5):
        blk = int(block_table[0, j].item())
        torch.testing.assert_close(ssm_state[blk], ssm_cold[blk], atol=atol, rtol=rtol)


def test_all_mode_interior_checkpoint_equals_shorter_prefill_final_state():
    """The +1-chunk shift, independently verified: the block-j interior
    checkpoint of a long prefill equals the FINAL state of a fresh prefill
    truncated at that block boundary."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    mamba_block_size = 64
    long_len = 5 * mamba_block_size
    short_len = 4 * mamba_block_size  # ends exactly at block 3's boundary
    weights = _make_weights(device)
    mixed_qkv, b, a = _make_inputs(long_len, device)
    cfg = _make_vllm_config(mamba_block_size, "all")

    def run(seq_len):
        batch = BatchSpec(seq_lens=[seq_len], query_lens=[seq_len])
        meta, common = _build_metadata(cfg, batch, mamba_block_size, device)
        pool_size = int(common.block_table_tensor.max().item()) + 1
        conv_state, ssm_state = _make_pools(pool_size, torch.float32, device)
        layer = _build_layer(cfg, mamba_block_size, conv_state, ssm_state, weights)
        _run_forward_core(
            layer, meta, mixed_qkv[:seq_len], b[:seq_len], a[:seq_len], seq_len
        )
        return ssm_state, common.block_table_tensor

    ssm_long, table_long = run(long_len)
    ssm_short, table_short = run(short_len)

    # Long run's interior checkpoint for block 3 (boundary at 256 tokens)
    # vs short run's final state (written from last_recurrent_state). The two
    # runs use different chunk grids (5 vs 4 chunks), so tiny accumulation
    # noise is expected; an off-by-one-chunk shift would differ by a full
    # chunk of state updates (orders of magnitude above this tolerance).
    blk_long = int(table_long[0, 3].item())
    blk_short = int(table_short[0, 3].item())
    torch.testing.assert_close(
        ssm_long[blk_long], ssm_short[blk_short], atol=5e-4, rtol=5e-3
    )


def test_chunk_wrapper_rejects_intermediate_states_on_non_triton():
    """FlashInfer/CuteDSL chunk backends cannot export per-chunk states; the
    wrapper must fail fast rather than silently skip checkpoints."""
    for fwd in (
        ChunkGatedDeltaRule.forward_cuda,
        ChunkGatedDeltaRule.forward_cutedsl,
    ):
        with pytest.raises(AssertionError):
            fwd(
                None,  # self: unused before the assert
                q=None,
                k=None,
                v=None,
                g=None,
                beta=None,
                initial_state=None,
                output_final_state=True,
                return_intermediate_states=True,
            )
