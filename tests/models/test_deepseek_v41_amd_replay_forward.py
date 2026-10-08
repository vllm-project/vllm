# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the AMD replay forward does with a prefill longer than the window.

The cut in this fixture is layer 1, so replay starts at layer 2. That is the
same boundary as production layers 20/21, on a 4-layer stand-in. The final
``mhc_post`` and the aux capture at that cut both run on the window rows.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.forward_context import ForwardContext
from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

WINDOW = 128
FULL = 300
# Trailing window of a 300-token prefill.
ROWS = list(range(FULL - WINDOW, FULL))
HC, HIDDEN = 4, 8
DEVICE = torch.device("cuda")


class _Layer(nn.Module):
    def __init__(self, idx: int):
        super().__init__()
        self.idx = idx
        self.seen: list[torch.Tensor] = []
        self.attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix=f"layers.{idx}.swa"),
            compressed_cache_prefix=None,
            indexer=None,
        )

    def forward(
        self,
        hidden_states,
        positions,
        input_ids,
        pre_mix,
        post_mix,
        res_mix,
        residual,
        engram_hashes,
        engram_mask,
    ):
        self.seen.append(hidden_states[:, 0, 0].detach().clone())
        if residual is None:
            residual = hidden_states
        if pre_mix is None:
            pre_mix = hidden_states[:, :, 0]
        if post_mix is None:
            post_mix = pre_mix
        if res_mix is None:
            res_mix = pre_mix
        return hidden_states, residual, post_mix, res_mix, pre_mix


def _model(monkeypatch):
    import vllm.models.deepseek_v41.amd.model as amd

    monkeypatch.setattr(
        amd,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=False, is_last_rank=False),
    )
    monkeypatch.setattr(amd.EngramLayout, "from_config", lambda config: None)
    monkeypatch.setattr(amd, "MHCPostOp", lambda: SimpleNamespace())

    layers = [_Layer(i) for i in range(4)]
    monkeypatch.setattr(amd, "make_layers", lambda n, factory, prefix: (0, n, layers))

    config = SimpleNamespace(
        vocab_size=32,
        hc_eps=1e-6,
        hc_mult=HC,
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
        index_topk=4,
        kv_source_layer_ids=[1],
        num_hidden_layers=4,
        sliding_window=WINDOW,
        engram_layer_ids=(),
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=torch.bfloat16),
        quant_config=None,
        cache_config=SimpleNamespace(swa_bounded_replay=True),
        use_v2_model_runner=True,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1,
            use_ubatching=False,
            enable_expert_parallel=False,
        ),
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=FULL),
    )
    model = amd.DeepseekV4Model(vllm_config=vllm_config)
    # Production requests aux at the first replay layer. Here that id is 2.
    model.aux_hidden_state_layers = (model.decoder_replay_start,)
    model.topk_indices_buffer = model.topk_indices_buffer.to(DEVICE)
    model.decoder_replay_layers.row_buffers = [model.topk_indices_buffer]

    calls: list[torch.Tensor] = []

    def mhc_post(hidden, residual, post_mix, res_mix):
        calls.append(hidden[:, 0, 0].detach().clone())
        return hidden

    model.mhc_post = mhc_post
    saved_aux: list[list[torch.Tensor]] = []
    run_layers = model._run_layers

    def wrapped(*args, **kwargs):
        result = run_layers(*args, **kwargs)
        # Snapshot: forward() extends the early list with the scattered aux.
        saved_aux.append(list(result[5]))
        return result

    model._run_layers = wrapped
    scattered: list[tuple[torch.Tensor, ...]] = []
    replay = model.decoder_replay_layers
    run = replay._run

    def wrapped_run(*args, **kwargs):
        result = run(*args, **kwargs)
        scattered.append(result)
        return result

    replay._run = wrapped_run
    return model, layers, calls, saved_aux, scattered


def _forward(model):
    token_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)
    hidden = token_ids[:, None, None].expand(FULL, HC, HIDDEN).contiguous()
    pre_mix = torch.ones(FULL, HC, device=DEVICE)
    rows = torch.tensor(ROWS, device=DEVICE)
    model.decoder_replay_layers.replay_batch = ReplayBatch(
        rows,
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}),
    )
    positions = torch.arange(FULL, device=DEVICE)
    out = model(
        input_ids=positions.to(torch.int32),
        positions=positions,
        intermediate_tensors=IntermediateTensors(
            {"hidden_states": hidden, "pre_mix": pre_mix}
        ),
    )
    return out


def test_replay_layers_see_only_the_trailing_window(monkeypatch):
    model, layers, calls, _saved_aux, _scattered = _model(monkeypatch)
    assert model.decoder_replay_layers.replay_batch is None
    out = _forward(model)
    replay = model.decoder_replay_layers.replay_batch
    assert replay is not None and replay.rows.tolist() == ROWS

    early = [layer.seen[0] for layer in layers[: model.decoder_replay_start]]
    late = [layer.seen[0] for layer in layers[model.decoder_replay_start :]]
    full_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)
    window_ids = full_ids[ROWS]
    assert [t.tolist() for t in early] == [full_ids.tolist(), full_ids.tolist()]
    assert [t.tolist() for t in late] == [window_ids.tolist(), window_ids.tolist()]

    # Scatter writes the replay rows back and leaves the trimmed rows at zero.
    hidden = out["hidden_states"][:, 0, 0]
    assert torch.equal(hidden[ROWS], window_ids)
    assert hidden[: FULL - WINDOW].eq(0).all()
    assert calls  # the seams below read these


def test_final_mhc_post_runs_on_the_replay_window(monkeypatch):
    model, _layers, calls, _saved_aux, _scattered = _model(monkeypatch)
    _forward(model)
    window_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)[ROWS]
    # calls[0] is the cut-aux capture. calls[1] is the final mhc_post.
    # Both run inside the replay batch, before the scatter back to FULL.
    assert len(calls) == 2
    assert [t.shape[0] for t in calls] == [WINDOW, WINDOW]
    assert torch.equal(calls[0], window_ids)
    assert torch.equal(calls[1], window_ids)


def test_aux_at_the_cut_is_captured_on_the_replay_rows(monkeypatch):
    model, _layers, _calls, saved_aux, scattered = _model(monkeypatch)
    _forward(model)
    # The early loop skips this id. The replay loop does not see it either.
    assert saved_aux[0] == []
    assert saved_aux[1] == []
    _hidden, _pre_mix, aux = scattered[0]
    window_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)[ROWS]
    assert aux.shape[0] == FULL
    assert torch.equal(aux[ROWS, 0], window_ids)
    assert aux[: FULL - WINDOW].eq(0).all()
