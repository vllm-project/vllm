# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-row prefill checkpoints on the Triton/FLA and AITER FlyDSL GDN backends.

Checkpoint slots must match a forward that stops at the checkpoint, and outputs
and final states must match the same batch without checkpoints.
"""

import types
from unittest.mock import patch

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike() or (
    current_platform.is_cuda() and current_platform.is_device_capability_family(100)
):
    pytest.skip(
        reason="The Triton/FLA GDN chunk kernel needs ROCm or CUDA below SM10x.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import (  # noqa: E402
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: E402
from vllm.model_executor.layers.mamba.gdn.prefill_checkpoint import (  # noqa: E402
    GDN_SPLIT_CHECKPOINT_ALIGNMENT,
)
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
CONV_DIM = 2 * H * K + HV * V
BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"

# (context, query) per prefill. The first two start at a block boundary and
# span more than two blocks, so they get checkpoints at tokens 112 and 48 (the
# second with no initial state); the last starts mid-block and gets none.
PREFILLS = [(32, 87), (0, 50), (37, 20)]
HEAD_LENS = [80, 48]  # query tokens before each checkpoint


def _make_vllm_config(backend: str):
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=BLOCK_SIZE,
        hf_config_override={"linear_key_head_dim": K},
    )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    cfg.cache_config.mamba_cache_mode = "align"
    cfg.cache_config.hash_block_size = BLOCK_SIZE
    cfg.cache_config.cache_hit_alignment_tokens = BLOCK_SIZE
    return cfg


def _make_builder(vllm_config, device, num_prefill_checkpoint_blocks: int):
    return GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            mamba_cache_mode="align",
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
            prefill_checkpoint_alignment=GDN_SPLIT_CHECKPOINT_ALIGNMENT,
        ),
        layer_names=[PREFIX],
        vllm_config=vllm_config,
        device=device,
    )


def _build(builder, vllm_config, seq_lens, query_lens, device):
    common = create_common_attn_metadata(
        BatchSpec(seq_lens=seq_lens, query_lens=query_lens),
        BLOCK_SIZE,
        device,
        arange_block_indices=True,
    )
    with set_current_vllm_config(vllm_config):
        meta = builder.build(common_prefix_len=0, common_attn_metadata=common)
    return meta, int(common.block_table_tensor.max()) + 1


def _run_forward_core(vllm_config, weights, meta, pools, mixed_qkv, b, a):
    conv_state, ssm_state = pools
    layer = types.SimpleNamespace(
        prefix=PREFIX,
        enable_packed_recurrent_decode=False,
        tp_size=1,
        num_k_heads=H,
        num_v_heads=HV,
        head_k_dim=K,
        head_v_dim=V,
        key_dim=H * K,
        value_dim=HV * V,
        activation="silu",
        conv_kernel_size=CONV_KERNEL,
        kv_cache=(conv_state, ssm_state),
        **weights,
    )
    with set_current_vllm_config(vllm_config):
        layer.chunk_gated_delta_rule = ChunkGatedDeltaRule()
    for name in ("rearrange_mixed_qkv", "_forward_core"):
        method = getattr(QwenGatedDeltaNetAttention, name)
        setattr(layer, name, types.MethodType(method, layer))

    out = torch.zeros(
        mixed_qkv.shape[0], HV, V, dtype=mixed_qkv.dtype, device=mixed_qkv.device
    )
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core(mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=out)
    return out


def _random_pools(pool_size, state_dtype, device):
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, H, HV, K, V, CONV_KERNEL, num_spec=0
    )
    conv = torch.randn(pool_size, *conv_shape, device=device).bfloat16() * 0.05
    ssm = torch.randn(pool_size, *ssm_shape, device=device).to(state_dtype)
    return conv, ssm


@pytest.mark.parametrize("backend", ["triton", "aiter_flydsl"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_decodes", [0, 1])
def test_split_prefill_checkpoint(
    backend: str, state_dtype: torch.dtype, num_decodes: int
) -> None:
    if (
        backend == "aiter_flydsl"
        and not rocm_aiter_ops.is_gdn_flydsl_prefill_available()
    ):
        pytest.skip("AITER FlyDSL GDN prefill is not available")
    torch.manual_seed(0)
    device = torch.device("cuda")
    vllm_config = _make_vllm_config(backend)

    seq_lens = [64] * num_decodes + [ctx + query for ctx, query in PREFILLS]
    query_lens = [1] * num_decodes + [query for _, query in PREFILLS]
    ckpt_builder = _make_builder(vllm_config, device, 1)
    plain_builder = _make_builder(vllm_config, device, 0)
    assert ckpt_builder.gdn_prefill_backend == backend
    meta, pool_size = _build(ckpt_builder, vllm_config, seq_lens, query_lens, device)
    plain_meta, _ = _build(plain_builder, vllm_config, seq_lens, query_lens, device)
    assert meta.checkpoint is not None and meta.checkpoint_split is not None
    assert meta.checkpoint.offsets == [0] * num_decodes + HEAD_LENS + [0]
    assert plain_meta.checkpoint is None

    conv_weight = torch.randn(CONV_DIM, 1, CONV_KERNEL, device=device).bfloat16()
    weights = dict(
        # Slow decay, so the tail visibly depends on the state it starts from.
        A_log=torch.randn(HV, device=device) * 0.1 - 5,
        dt_bias=torch.randn(HV, device=device) * 0.1,
        conv1d=types.SimpleNamespace(
            weight=conv_weight * 0.1,
            bias=torch.randn(CONV_DIM, device=device).bfloat16() * 0.1,
        ),
    )
    num_tokens = sum(query_lens)
    mixed_qkv = torch.randn(num_tokens, CONV_DIM, device=device).bfloat16() * 0.1
    a = torch.randn(num_tokens, HV, device=device).bfloat16() * 0.1
    b = torch.randn(num_tokens, HV, device=device).bfloat16() * 0.1
    pools = _random_pools(pool_size, state_dtype, device)

    ckpt_pools = tuple(p.clone() for p in pools)
    out = _run_forward_core(vllm_config, weights, meta, ckpt_pools, mixed_qkv, b, a)
    plain_pools = tuple(p.clone() for p in pools)
    plain_out = _run_forward_core(
        vllm_config, weights, plain_meta, plain_pools, mixed_qkv, b, a
    )

    atol = rtol = 2e-2 if state_dtype == torch.float32 else 6e-2
    final_slots = meta.non_spec_state_indices_tensor.long()
    torch.testing.assert_close(out, plain_out, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        ckpt_pools[0][final_slots], plain_pools[0][final_slots], atol=0, rtol=0
    )
    torch.testing.assert_close(
        ckpt_pools[1][final_slots], plain_pools[1][final_slots], atol=atol, rtol=rtol
    )

    # Reference: the checkpointed prefills stopping at their checkpoints, from
    # the same initial states, write the expected checkpoints to their slots.
    rows = slice(num_decodes, num_decodes + len(HEAD_LENS))
    starts = [num_decodes, num_decodes + PREFILLS[0][1]]
    head = torch.cat(
        [torch.arange(s, s + n, device=device) for s, n in zip(starts, HEAD_LENS)]
    )
    ref_meta, ref_pool_size = _build(
        plain_builder,
        vllm_config,
        [ctx + n for (ctx, _), n in zip(PREFILLS, HEAD_LENS)],
        HEAD_LENS,
        device,
    )
    ref_slots = ref_meta.non_spec_state_indices_tensor.long()
    ref_pools = _random_pools(ref_pool_size, state_dtype, device)
    for ref_pool, pool in zip(ref_pools, pools):
        ref_pool[ref_slots] = pool[final_slots[rows]]
    _run_forward_core(
        vllm_config, weights, ref_meta, ref_pools, mixed_qkv[head], b[head], a[head]
    )

    ckpt_slots = meta.checkpoint.state_indices[rows].long()
    torch.testing.assert_close(
        ckpt_pools[0][ckpt_slots], ref_pools[0][ref_slots], atol=0, rtol=0
    )
    torch.testing.assert_close(
        ckpt_pools[1][ckpt_slots], ref_pools[1][ref_slots], atol=atol, rtol=rtol
    )
