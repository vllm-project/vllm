# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import types
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.kv_cache_interface import MambaSpec

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA GDN state-restoration test"
)


@pytest.mark.parametrize("accepted", [1, 3])
@pytest.mark.parametrize("mode", ["decode", "packed_decode", "mixed"])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_forward_core_restores_spec_state(accepted, mode, state_dtype):
    torch.manual_seed(7)
    device = torch.device("cuda")
    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=16,
        hf_config_override={"linear_key_head_dim": 128},
    )
    cfg.additional_config = {"gdn_prefill_backend": "triton"}
    cfg.cache_config.mamba_cache_mode = "none"
    cfg.speculative_config = SpeculativeConfig(method="ngram", num_speculative_tokens=2)
    mixed = mode == "mixed"
    query_lens = [1, 1, 3] if mixed else [1, 1]
    counts = [accepted, 1, 2] if mixed else [accepted, 1]
    drafts = [-1, -1, 2] if mixed else None
    batch = BatchSpec(seq_lens=[65] * len(query_lens), query_lens=query_lens)
    common = create_common_attn_metadata(batch, 16, device, arange_block_indices=True)
    common.block_table_tensor.add_(1)
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=16, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=["gdn"],
        vllm_config=cfg,
        device=device,
    )
    with set_current_vllm_config(cfg):
        meta = builder.build(
            common_prefix_len=0,
            common_attn_metadata=common,
            num_accepted_tokens=torch.tensor(counts, dtype=torch.int32, device=device),
            num_decode_draft_tokens_cpu=(
                torch.tensor(drafts, dtype=torch.int32) if drafts is not None else None
            ),
        )
    h, hv, k, v = 4, 8, 128, 128
    conv_dim = 2 * h * k + hv * v
    conv_shape, ssm_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, h, hv, k, v, 4, num_spec=2
    )
    pool_size = int(common.block_table_tensor.max().item()) + 1
    conv = (
        torch.randn(pool_size, *conv_shape, device=device, dtype=torch.bfloat16) * 0.05
    )
    ssm = torch.randn(pool_size, *ssm_shape, device=device, dtype=state_dtype) * 0.05
    conv_ref, ssm_ref = conv.clone(), ssm.clone()
    destination = meta.non_spec_state_indices_tensor[:2]
    source = (
        common.block_table_tensor[:2, :]
        .gather(1, torch.tensor([[accepted - 1], [0]], device=device))
        .squeeze(1)
    )
    ssm_ref[destination] = ssm[source]
    conv_view = conv if is_conv_state_dim_first() else conv.transpose(-1, -2)
    ref_view = conv_ref if is_conv_state_dim_first() else conv_ref.transpose(-1, -2)
    for i, count in enumerate([accepted, 1]):
        ref_view[destination[i], :, :3] = conv_view[
            destination[i], :, count - 1 : count + 2
        ]
    meta_ref = dataclasses.replace(
        meta,
        spec_decode_src_indices=None,
        non_spec_num_accepted=torch.ones(2, device=device, dtype=torch.int32)
        if mixed
        else None,
        num_accepted_tokens=meta.num_accepted_tokens if mixed else None,
    )
    weight = torch.randn(conv_dim, 1, 4, device=device, dtype=torch.bfloat16) * 0.1
    bias = torch.randn(conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    a_log = torch.randn(hv, device=device) * 0.1
    dt_bias = torch.randn(hv, device=device) * 0.1
    tokens = sum(query_lens)
    qkv = torch.randn(tokens, conv_dim, device=device, dtype=torch.bfloat16) * 0.1
    a = torch.randn(tokens, hv, device=device, dtype=torch.bfloat16) * 0.1
    b = torch.randn_like(a)
    outputs = []
    for conv_pool, ssm_pool, metadata in [
        (conv, ssm, meta),
        (conv_ref, ssm_ref, meta_ref),
    ]:
        layer = types.SimpleNamespace(
            prefix="gdn",
            enable_packed_recurrent_decode=mode == "packed_decode",
            tp_size=1,
            num_k_heads=h,
            num_v_heads=hv,
            head_k_dim=k,
            head_v_dim=v,
            key_dim=h * k,
            value_dim=hv * v,
            activation="silu",
            A_log=a_log,
            dt_bias=dt_bias,
            conv1d=types.SimpleNamespace(weight=weight, bias=bias),
            kv_cache=(conv_pool, ssm_pool),
        )
        with set_current_vllm_config(cfg):
            layer.chunk_gated_delta_rule = gdn.ChunkGatedDeltaRule()
        for name in (
            "rearrange_mixed_qkv",
            "_forward_core",
            "_forward_core_decode_non_spec",
        ):
            setattr(
                layer,
                name,
                types.MethodType(getattr(gdn.QwenGatedDeltaNetAttention, name), layer),
            )
        out = torch.zeros(tokens, hv, v, device=device, dtype=torch.bfloat16)
        ctx = types.SimpleNamespace(attn_metadata={"gdn": metadata})
        with patch.object(gdn, "get_forward_context", return_value=ctx):
            layer._forward_core(qkv.clone(), b.clone(), a.clone(), out)
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(ssm, ssm_ref, rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(
        conv_view[destination, :, :3], ref_view[destination, :, :3], rtol=0, atol=0
    )
