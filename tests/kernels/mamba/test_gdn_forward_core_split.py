# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed decode/prefill state updates through real Qwen and OLMo GDN layers.

The decode-first split must match a single chunked recurrence over identical
projected inputs. Convolution state is exact; recurrent math uses the existing
chunk-versus-update tolerances.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.gdn.olmo_gdn_linear_attn import (
    OlmoHybridGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GDN layer kernels require CUDA"
)

BLOCK_SIZE = 16
PREFIX = "model.layers.0.linear_attn"


def _make_vllm_config(model, backend):
    cfg = create_vllm_config(
        model_name=(
            "Qwen/Qwen3.5-0.8B" if model == "qwen" else "allenai/Olmo-Hybrid-7B"
        ),
        block_size=BLOCK_SIZE,
        dtype="bfloat16",
    )
    cfg.cache_config.enable_prefix_caching = False
    cfg.model_config.hf_text_config.update(
        {
            "hidden_size": 64,
            "linear_num_key_heads": 4 if model == "qwen" else 2,
            "linear_num_value_heads": 8 if model == "qwen" else 2,
            "linear_key_head_dim": 128 if model == "qwen" else 96,
            "linear_value_head_dim": 128 if model == "qwen" else 192,
            "linear_conv_kernel_dim": 4,
        }
    )
    if backend is None:
        backend = (
            "cutedsl"
            if model == "qwen" and current_platform.is_device_capability_family(100)
            else "triton"
        )
    cfg.additional_config = {"gdn_prefill_backend": backend}
    return cfg


def _build_layer(cfg, model, conv_state, ssm_state, a_log, dt_bias, conv_weight):
    prefix = (
        f"model.layers.{len(cfg.compilation_config.static_forward_context)}.linear_attn"
    )
    layer_type = (
        QwenGatedDeltaNetAttention
        if model == "qwen"
        else OlmoHybridGatedDeltaNetAttention
    )
    with (
        set_current_vllm_config(cfg),
        torch.device("cuda"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        layer = layer_type(cfg.model_config.hf_text_config, cfg, prefix=prefix)
    layer.load_state_dict(
        {
            "A_log": a_log,
            "dt_bias": dt_bias,
            "conv1d.weight": conv_weight,
        },
        strict=False,
    )
    layer.kv_cache = (conv_state, ssm_state)
    return layer


def _run_forward_core(layer, cfg, meta, mixed_qkv, b, a):
    out = torch.zeros(
        mixed_qkv.shape[0],
        layer.num_v_heads,
        layer.head_v_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    with set_current_vllm_config(cfg), set_forward_context({layer.prefix: meta}, cfg):
        layer._forward_core(mixed_qkv.clone(), b.clone(), a.clone(), out)
    return out


@pytest.mark.parametrize(
    "model,backend",
    [
        ("qwen", None),
        ("olmo", None),
        pytest.param(
            "qwen",
            "flashinfer",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda()
                or not (
                    current_platform.is_device_capability(90)
                    or (
                        current_platform.get_cuda_runtime_major() >= 13
                        and (
                            current_platform.is_device_capability_family(100)
                            or current_platform.is_device_capability_family(120)
                        )
                    )
                ),
                reason="FlashInfer GDN needs SM90, or SM100/SM120 with CUDA 13.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_decodes,prefill_lens", [(3, [512, 300]), (4, [64, 5])])
@pytest.mark.parametrize("fresh_prefill", [False, True])
@torch.inference_mode()
def test_forward_core_split_matches_unified(
    dist_init,
    model,
    backend,
    state_dtype,
    num_decodes,
    prefill_lens,
    fresh_prefill,
):
    torch.manual_seed(0)
    device = torch.device("cuda")
    cfg = _make_vllm_config(model, backend)
    config = cfg.model_config.hf_text_config
    h, hv = config.linear_num_key_heads, config.linear_num_value_heads
    k, v = config.linear_key_head_dim, config.linear_value_head_dim
    conv_dim = 2 * h * k + hv * v
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, h, hv, k, v, 4, 0
    )
    cfg.cache_config.mamba_ssm_cache_dtype = (
        "float32" if state_dtype == torch.float32 else "bfloat16"
    )
    batch = BatchSpec(
        seq_lens=[64] * num_decodes
        + [
            length if fresh_prefill and i == 0 else length + 37
            for i, length in enumerate(prefill_lens)
        ],
        query_lens=[1] * num_decodes + list(prefill_lens),
    )
    builder = GDNAttentionMetadataBuilder(
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=(conv_shape, ssm_shape),
            dtypes=(torch.bfloat16, state_dtype),
        ),
        [PREFIX],
        cfg,
        device,
    )
    common = create_common_attn_metadata(
        batch, BLOCK_SIZE, device, arange_block_indices=True
    )
    # Block zero is the null slot, not a live request's recurrent state.
    common = common.replace(block_table_tensor=common.block_table_tensor + 1)
    with set_current_vllm_config(cfg):
        meta_split = builder.build(0, common)
    if backend == "flashinfer":
        assert builder.gdn_prefill_backend == "flashinfer"
    assert meta_split.num_decodes == num_decodes
    assert meta_split.num_prefills == len(prefill_lens)
    assert meta_split.spec_sequence_masks is None
    cu_full = meta_split.non_spec_query_start_loc
    assert cu_full is not None
    if builder.gdn_prefill_backend == "cutedsl":
        from vllm.model_executor.layers.mamba.ops.gdn_chunk_cutedsl import (
            prepare_metadata_cutedsl,
        )

        chunk_indices, chunk_offsets = prepare_metadata_cutedsl(
            cu_full, int(cu_full[-1].item()), FLA_CHUNK_SIZE
        )
    else:
        chunk_indices = prepare_chunk_indices(cu_full.cpu(), FLA_CHUNK_SIZE).to(device)
        chunk_offsets = prepare_chunk_offsets(cu_full.cpu(), FLA_CHUNK_SIZE).to(device)
    meta_unified = dataclasses.replace(
        meta_split,
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=meta_split.num_decodes + meta_split.num_prefills,
        num_prefill_tokens=batch.compute_num_tokens(),
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        prefill_query_start_loc=cu_full,
        prefill_state_indices=meta_split.non_spec_state_indices_tensor,
        prefill_has_initial_state=meta_split.has_initial_state,
    )
    assert meta_split.non_spec_state_indices_tensor is not None
    pool_size = int(meta_split.non_spec_state_indices_tensor.max().item()) + 1
    conv_state = (
        torch.randn(pool_size, *conv_shape, dtype=torch.bfloat16, device=device) * 0.05
    )
    ssm_state = (
        torch.randn(pool_size, *ssm_shape, dtype=state_dtype, device=device) * 0.05
    )
    a_log = torch.randn(hv, dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn_like(a_log) * 0.1
    conv_weight = torch.randn(conv_dim, 1, 4, dtype=torch.bfloat16, device=device) * 0.1
    mixed_qkv = (
        torch.randn(
            batch.compute_num_tokens(), conv_dim, dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    a = (
        torch.randn(batch.compute_num_tokens(), hv, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    b = torch.randn_like(a) * 0.1
    split_layer = _build_layer(
        cfg, model, conv_state.clone(), ssm_state.clone(), a_log, dt_bias, conv_weight
    )
    unified_layer = _build_layer(
        cfg, model, conv_state.clone(), ssm_state.clone(), a_log, dt_bias, conv_weight
    )
    out_split = _run_forward_core(split_layer, cfg, meta_split, mixed_qkv, b, a)
    out_unified = _run_forward_core(unified_layer, cfg, meta_unified, mixed_qkv, b, a)
    torch.testing.assert_close(
        split_layer.kv_cache[0], unified_layer.kv_cache[0], atol=0, rtol=0
    )
    atol = rtol = 2e-2 if state_dtype == torch.float32 else 6e-2
    torch.testing.assert_close(out_split, out_unified, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        split_layer.kv_cache[1], unified_layer.kv_cache[1], atol=atol, rtol=rtol
    )
