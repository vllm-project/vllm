# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v41.nvidia import model as model_module


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
