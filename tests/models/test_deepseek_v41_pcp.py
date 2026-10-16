# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41 import attention as attention_module
from vllm.models.deepseek_v41.decoder_replay_layers import (
    DecoderReplayLayers,
    ReplayBatch,
)
from vllm.models.deepseek_v41.nvidia import model as model_module


@pytest.mark.parametrize("pcp_size", [1, 2])
def test_replay_kv_publishes_all_pcp_owners(pcp_size, monkeypatch):
    """The KV-only source publishes the same rank-major payload as attention."""
    kv, latent = torch.arange(6).view(2, 3).float(), torch.ones(2, 3)
    positions = torch.tensor([10, 11])
    cache_kv = torch.cat([kv + 10 * rank for rank in range(pcp_size)])
    cache_latent = torch.cat([latent + rank for rank in range(pcp_size)])
    cache_positions = torch.cat([positions + 10 * rank for rank in range(pcp_size)])
    slots = torch.arange(2 * pcp_size)
    slots[1] = -1  # APC replay rows must keep their PAD slots.
    metadata = SimpleNamespace(
        slot_mapping=slots[:2],
        cache_slot_mapping=slots,
        cache_positions=cache_positions,
        block_size=64,
    )
    gather = Mock(side_effect=[cache_kv, cache_latent])
    monkeypatch.setattr(
        attention_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    insert = Mock()
    monkeypatch.setattr(
        torch.ops._C, "fused_deepseek_v4_kv_rope_insert", insert, raising=False
    )
    compressor, indexer = Mock(return_value=latent), Mock(owns_k=True)
    rotary = SimpleNamespace(cos_sin_cache=torch.empty(0))
    layer = SimpleNamespace(
        use_pcp=pcp_size > 1,
        _run_parallel_input_projections=lambda hidden: (hidden, None, None),
        _split_qkv_and_norm=lambda projected: (None, None, kv),
        swa_cache_layer=SimpleNamespace(
            prefix="swa", kv_cache=torch.empty(0), block_size=64
        ),
        compressor=compressor,
        indexer=indexer,
        rotary_emb=rotary,
        indexer_rotary_emb=rotary,
        kv_mxfp8=False,
    )
    context = ForwardContext(
        no_compile_layers={}, attn_metadata={"swa": metadata}, slot_mapping={}
    )
    with override_forward_context(context):
        attention_module.DeepseekV4Attention.forward_kv(layer, positions, kv)
    torch.testing.assert_close(insert.call_args.args[0], cache_kv)
    torch.testing.assert_close(insert.call_args.args[2], slots)
    torch.testing.assert_close(insert.call_args.args[3], cache_positions)
    for publish in (compressor.insert_cache, indexer._produce_k):
        torch.testing.assert_close(publish.call_args.args[0], cache_latent)
        torch.testing.assert_close(publish.call_args.args[1], cache_positions)
    assert gather.call_count == (2 if pcp_size > 1 else 0)


@pytest.mark.parametrize(
    "pcp_size,use_ep", [(1, False), (2, True), (2, False), (4, False)]
)
@pytest.mark.parametrize("use_sp", [False, True])
def test_pcp_gathers_routing_ids_once_after_local_input_preparation(
    pcp_size, use_ep, use_sp, monkeypatch
):
    """Embedding keeps local IDs; all decoder layers receive aligned routing IDs."""
    gather = Mock(side_effect=lambda ids, dim: torch.cat([ids] * pcp_size, dim=dim))
    monkeypatch.setattr(
        model_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(model_module, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(model_module, "sp_shard", lambda tensor: tensor[::2])
    monkeypatch.setattr(
        model_module, "sp_all_gather", lambda tensor: tensor.repeat_interleave(2, dim=0)
    )
    monkeypatch.setattr(model_module, "hc_collapse_triton", lambda hidden, mix: hidden)
    embed = Mock(side_effect=lambda ids: ids[:, None].float())
    run_layers = Mock(
        side_effect=lambda layer_ids, hidden, *args: (hidden, None, None, None, hidden)
    )
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size, enable_expert_parallel=use_ep
        ),
        use_native_mega_moe=False,
        use_sequence_parallel=use_sp,
        engram_hash=None,
        embed_input_ids=embed,
        start_layer=0,
        decoder_replay_start=2,
        decoder_replay_layers=None,
        aux_hidden_state_layers=(),
        _mtp_hidden_buffer=None,
        _run_layers=run_layers,
        _collapse=lambda hidden, *args: hidden,
        norm=lambda hidden: hidden,
    )
    for offset in (0, 10):
        local_ids = torch.tensor([3, 7, 11, 15]) + offset
        model_module.DeepseekV4Model.forward(model, local_ids, torch.arange(4), None)
        assert embed.call_args.args[0] is local_ids
        routing_ids = local_ids[::2] if use_sp else local_ids
        needs_gather = pcp_size > 1 and not use_ep
        if needs_gather:
            torch.testing.assert_close(gather.call_args.args[0], routing_ids)
            routing_ids = routing_ids.repeat(pcp_size)
        torch.testing.assert_close(run_layers.call_args.args[3], routing_ids)
    assert gather.call_count == 2 * int(needs_gather)

@pytest.mark.parametrize("method,adaptive", [("mtp", False), ("dspark", True)])
def test_pcp_rejects_speculation_other_than_fixed_length_dspark(method, adaptive):
    config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        quant_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=SimpleNamespace(
            use_dspark=lambda: method == "dspark",
            enable_adaptive_verification=adaptive,
        ),
    )
    with pytest.raises(NotImplementedError, match="fixed-length DSpark"):
        model_module.DeepseekV4Model(vllm_config=config)
