# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.config import KVTransferConfig
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
    if pcp_size == 2 and not use_ep and not use_sp:
        replay = Mock(return_value=(embed(local_ids), embed(local_ids)))
        replay.replay_batch = object()
        model.decoder_replay_layers = replay
        model_module.DeepseekV4Model.forward(model, local_ids, torch.arange(4), None)
        torch.testing.assert_close(replay.call_args.args[2], local_ids)


@pytest.mark.parametrize("rows", [[], [0, 3]])
def test_pcp_replay_pads_inputs_and_scatters_only_real_rows(rows):
    """Empty peers still run padded layers without overwriting real outputs."""
    hidden = torch.arange(4, dtype=torch.float32)[:, None]

    def run_layers(values):
        assert values.shape == (4, 1)
        torch.testing.assert_close(values[: len(rows)], hidden[rows])
        assert not values[len(rows) :].count_nonzero()
        return (values + 1,)

    layers = DecoderReplayLayers(2, run_layers, lambda *args: None, set(), "swa")
    layers.replay_batch = ReplayBatch(
        torch.tensor(rows, dtype=torch.int64),
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}),
        num_padded=4,
    )
    (output,) = layers(hidden)
    expected = torch.zeros_like(hidden)
    expected[rows] = hidden[rows] + 1
    torch.testing.assert_close(output, expected)


def test_pcp_replay_gathers_compacted_routing_ids(monkeypatch):
    gather = Mock(side_effect=lambda ids, dim: ids.repeat(2))
    monkeypatch.setattr(
        model_module, "get_pcp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    run_layers = Mock(side_effect=lambda _, hidden, *args: (hidden,) * 5)
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=2, enable_expert_parallel=False
        ),
        decoder_replay_layers=SimpleNamespace(replay_batch=object()),
        decoder_replay_start=1,
        end_layer=2,
        aux_hidden_state_layers=(),
        _run_layers=run_layers,
        _collapse=lambda hidden, *args: hidden,
    )
    values = torch.tensor([3, 7])
    model_module.DeepseekV4Model._run_replay_layers(model, *([values] * 7))
    torch.testing.assert_close(run_layers.call_args.args[3], values.repeat(2))


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


def test_pcp_nixl_requires_pull_producer(default_vllm_config):
    config = default_vllm_config
    config.parallel_config.prefill_context_parallel_size = 2
    config.cache_config.enable_prefix_caching = False
    config.scheduler_config.async_scheduling = False
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="NixlConnector", kv_role="kv_producer", kv_buffer_device="cuda"
    )
    model_module._validate_pcp_config(config)

    config.kv_transfer_config.kv_role = "kv_consumer"
    with pytest.raises(NotImplementedError, match="consumers require PCP=1"):
        model_module._validate_pcp_config(config)
