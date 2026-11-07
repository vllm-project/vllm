# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_entry_injects_engram_before_mhc_pre(monkeypatch):
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")

    class StopAfterMhcPre(Exception):
        pass

    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    hashes = torch.arange(6).reshape(2, 3)
    mask = torch.tensor([True, False])
    pre_mix = torch.randn(2, 4)
    engram_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]] = []
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    class Engram:
        layer_hash_index = 1

        def __call__(self, residual, layer_hashes, engram_mask):
            engram_calls.append((residual, layer_hashes, engram_mask))
            return residual + 10

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        raise StopAfterMhcPre

    layer = SimpleNamespace(
        mhc_stream=None,
        engram=Engram(),
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
    )
    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)

    with pytest.raises(StopAfterMhcPre):
        model_module.DeepseekV4DecoderLayer.forward(
            layer,
            hidden,
            torch.arange(2),
            None,
            pre_mix=pre_mix,
            engram_hashes=hashes,
            engram_mask=mask,
        )

    assert len(engram_calls) == len(mhc_pre_calls) == 1
    engram_residual, layer_hashes, engram_mask = engram_calls[0]
    assert engram_residual is hidden
    assert engram_mask is mask
    torch.testing.assert_close(layer_hashes, hashes[:, 1])
    mhc_residual, mhc_pre_mix = mhc_pre_calls[0]
    torch.testing.assert_close(mhc_residual, hidden + 10)
    assert mhc_pre_mix is pre_mix


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_starting_at_replay_layers_runs_only_them(monkeypatch):
    """The default 20/20 PP2 split starts the last stage at the last KV source:
    it has no early layers, and the replay layers take the stage's input."""
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")
    monkeypatch.setattr(
        model_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=True),
    )
    monkeypatch.setattr(
        model_module, "hc_collapse_triton", lambda hidden, mix: hidden[:, 0]
    )
    hidden = torch.randn(3, 4, 5)
    pre_mix = torch.randn(3, 4)
    replay = MagicMock(return_value=(hidden + 1, pre_mix + 1))
    replay.replay_batch = None
    model = SimpleNamespace(
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1),
        use_native_mega_moe=False,
        use_sequence_parallel=False,
        engram_hash=None,
        start_layer=20,
        end_layer=40,
        decoder_replay_start=20,
        decoder_replay_layers=replay,
        aux_hidden_state_layers=(),
        layers=[],
        _mtp_hidden_buffer=None,
        _mega_gate_metadata=lambda input_ids: None,
        norm=lambda hidden: hidden,
    )
    model._run_layers = lambda *args: model_module.DeepseekV4Model._run_layers(
        model, *args
    )
    input_ids = torch.arange(3)
    positions = torch.arange(3)

    output = model_module.DeepseekV4Model.forward(
        model,
        input_ids,
        positions,
        IntermediateTensors({"hidden_states": hidden, "pre_mix": pre_mix}),
    )

    replay.assert_called_once()
    stage_input, *states = replay.call_args.args
    assert stage_input is hidden
    assert states[0] is positions and states[1] is input_ids
    assert states[2] is pre_mix
    assert states[3:] == [None, None, None]
    torch.testing.assert_close(output, hidden[:, 0] + 1)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")
@pytest.mark.skip_global_cleanup
def test_stage_entry_write_kv_collapses_stage_input(monkeypatch):
    """At a stage start the first replay layer has no residual to post: it writes
    the KV of the received hc streams collapsed with the carried pre-mix."""
    model_module = import_module("vllm.models.deepseek_v41.nvidia.model")
    hidden = torch.randn(2, 4, 3)
    pre_mix = torch.randn(2, 4)
    collapsed = torch.randn(2, 3)
    mhc_pre_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def mhc_pre(residual, *args, pre_mix=None, **kwargs):
        mhc_pre_calls.append((residual, pre_mix))
        return None, None, collapsed, None

    def mhc_shifted_post_pre(*args, **kwargs):
        raise AssertionError("a stage start has no previous layer to post")

    monkeypatch.setattr(model_module, "mhc_pre_delayed_tilelang", mhc_pre)
    monkeypatch.setattr(model_module, "mhc_shifted_post_pre", mhc_shifted_post_pre)
    forward_kv = MagicMock()
    layer = SimpleNamespace(
        engram=None,
        use_sequence_parallel=False,
        hc_attn_fn=None,
        hc_attn_scale=None,
        hc_attn_base=None,
        rms_norm_eps=1e-6,
        hc_eps=1e-6,
        hc_post_alpha=1.0,
        hc_sinkhorn_iters=1,
        attn_norm=SimpleNamespace(weight=None, variance_epsilon=1e-6),
        fuse_mhc_all_reduce=False,
        attn=SimpleNamespace(forward_kv=forward_kv),
    )
    positions = torch.arange(2)

    model_module.DeepseekV4DecoderLayer.write_kv(
        layer, hidden, positions, None, pre_mix, None, None, None
    )

    assert len(mhc_pre_calls) == 1
    assert mhc_pre_calls[0][0] is hidden and mhc_pre_calls[0][1] is pre_mix
    forward_kv.assert_called_once()
    assert forward_kv.call_args.args[0] is positions
    assert forward_kv.call_args.args[1] is collapsed
