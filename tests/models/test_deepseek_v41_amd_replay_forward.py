# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What the AMD replay forward does with a prefill longer than the window.

The cut in this fixture is layer 1, so replay starts at layer 1. That is the
same boundary as production layer 20, on a 4-layer stand-in. The final
``mhc_post`` and the aux capture at that cut both run on the window rows.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm.forward_context import ForwardContext, override_forward_context
from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
from vllm.sequence import IntermediateTensors

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

WINDOW = 128
FULL = 300
# Trailing window of a 300-token prefill.
ROWS = list(range(FULL - WINDOW, FULL))
HC, HIDDEN = 4, 8
DEVICE = torch.device("cuda")
# gfx942 writes the V4 record (584 B). Family-100 writes the V4.1 record (528 B).
# The SWA layer uses block size 32, which is the size the KV-only insert checks.
_BLOCK = 32
_HEAD_DIM = 512


class _RocmKvInsert:
    """SWA cache from ``forward_kv``, checked against the KV-only insert."""

    def __init__(self, num_tokens: int):
        from vllm.models.deepseek_v41.attention import _use_v41_mxfp8_kv_record

        if not hasattr(torch.ops._C, "fused_deepseek_v4_kv_rope_insert"):
            pytest.skip("fused DeepseekV4 KV insert op is not built")

        self.num_tokens = num_tokens
        self.kv_mxfp8 = _use_v41_mxfp8_kv_record()
        bytes_per = 528 if self.kv_mxfp8 else 584
        num_blocks = (num_tokens + _BLOCK - 1) // _BLOCK + 1
        cache = torch.full(
            (num_blocks, _BLOCK, bytes_per),
            0xA5,
            dtype=torch.uint8,
            device=DEVICE,
        )
        self.poison = cache.clone()
        self.kv = torch.randn(
            num_tokens, _HEAD_DIM, dtype=torch.bfloat16, device=DEVICE
        )
        prefix = "replay.swa"
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
        self.metadata = {
            prefix: SimpleNamespace(slot_mapping=slot_mapping, block_size=_BLOCK)
        }
        self.context = ForwardContext({}, self.metadata, {})
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=DEVICE) / 64)
        )
        freqs = torch.outer(
            torch.arange(num_tokens + 8, dtype=torch.float32, device=DEVICE), inv_freq
        )
        self.attn = SimpleNamespace(
            kv_mxfp8=self.kv_mxfp8,
            compressor=None,
            indexer=None,
            rotary_emb=SimpleNamespace(
                cos_sin_cache=torch.cat((freqs.cos(), freqs.sin()), dim=-1)
            ),
            swa_cache_layer=SimpleNamespace(
                prefix=prefix, kv_cache=cache, block_size=_BLOCK
            ),
            _run_parallel_input_projections=lambda hidden: (hidden, None, None),
            _split_qkv_and_norm=lambda qr_kv: (None, None, self.kv[: qr_kv.shape[0]]),
        )
        self.last_rows: int | None = None
        self.last_positions: torch.Tensor | None = None

    def write(
        self,
        x,
        positions,
        input_ids,
        pre_mix,
        post_mix,
        res_mix,
        residual,
    ):
        from vllm.models.deepseek_v41.amd.rocm import (
            DeepseekV41ROCMAiterMLAAttention,
        )
        from vllm.models.deepseek_v41.attention import DeepseekV4Attention

        assert (
            DeepseekV41ROCMAiterMLAAttention.forward_kv
            is DeepseekV4Attention.forward_kv
        )
        self.last_rows = x.shape[0]
        self.last_positions = positions
        DeepseekV41ROCMAiterMLAAttention.forward_kv(self.attn, positions, x)

    def assert_matches_forward(self):
        assert self.last_rows == self.num_tokens
        assert self.last_positions is not None
        written = self.attn.swa_cache_layer.kv_cache
        assert not torch.equal(written, self.poison)
        ref = self.poison.clone()
        meta = self.metadata[self.attn.swa_cache_layer.prefix]
        torch.ops._C.fused_deepseek_v4_kv_rope_insert(
            self.kv,
            ref,
            meta.slot_mapping,
            self.last_positions,
            self.attn.rotary_emb.cos_sin_cache,
            _BLOCK,
            None,
            self.kv_mxfp8,
        )
        assert torch.equal(written, ref)
        assert torch.equal(written[-1], self.poison[-1])


class _Layer(nn.Module):
    def __init__(
        self,
        idx: int,
        compressed: str | None = None,
        indexer: str | None = None,
    ):
        super().__init__()
        self.idx = idx
        self.seen: list[torch.Tensor] = []
        self.attn = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix=f"layers.{idx}.swa"),
            compressed_cache_prefix=compressed,
            indexer=(
                None
                if indexer is None
                else SimpleNamespace(k_cache=SimpleNamespace(prefix=indexer))
            ),
            compress_ratio=1,
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

    # Layer 0 sits before the cut, so its keys are not collected. Layer 2
    # shares layer 1's compressed cache. Layer 3 has neither.
    layers = [
        _Layer(0, "layers.0.compressed", "layers.0.indexer"),
        _Layer(1, "layers.1.compressed", "layers.1.indexer"),
        _Layer(2, "layers.1.compressed", "layers.2.indexer"),
        _Layer(3),
    ]
    # Layer 1 is the KV source, so the replay callback is this layer's write.
    insert = _RocmKvInsert(FULL)
    layers[1].write_kv = insert.write
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
    # Production requests aux at the first replay layer. Here that id is the cut.
    model.aux_hidden_state_layers = (model.decoder_replay_start,)
    model.topk_indices_buffer = model.topk_indices_buffer.to(DEVICE)

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
    return model, layers, calls, saved_aux, scattered, insert


def _forward(model, insert: _RocmKvInsert):
    token_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)
    hidden = token_ids[:, None, None].expand(FULL, HC, HIDDEN).contiguous()
    pre_mix = torch.ones(FULL, HC, device=DEVICE)
    rows = torch.tensor(ROWS, device=DEVICE)
    model.decoder_replay_layers.replay_batch = ReplayBatch(
        rows,
        ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={}),
    )
    positions = torch.arange(FULL, device=DEVICE)
    with override_forward_context(insert.context):
        out = model(
            input_ids=positions.to(torch.int32),
            positions=positions,
            intermediate_tensors=IntermediateTensors(
                {"hidden_states": hidden, "pre_mix": pre_mix}
            ),
        )
    return out


def test_amd_replay_forward_trims_to_the_window(monkeypatch):
    model, layers, calls, saved_aux, scattered, insert = _model(monkeypatch)
    replay_layers = model.decoder_replay_layers
    assert model.decoder_replay_start == 1
    assert replay_layers.first_swa_prefix == "layers.1.swa"
    assert replay_layers.metadata_prefixes == {
        "layers.1.swa",
        "layers.1.compressed",
        "layers.1.indexer",
        "layers.2.swa",
        "layers.2.indexer",
        "layers.3.swa",
    }
    assert replay_layers.replay_batch is None
    out = _forward(model, insert)
    insert.assert_matches_forward()
    replay = model.decoder_replay_layers.replay_batch
    assert replay is not None and replay.rows.tolist() == ROWS

    early = [layer.seen[0] for layer in layers[: model.decoder_replay_start]]
    late = [layer.seen[0] for layer in layers[model.decoder_replay_start :]]
    full_ids = torch.arange(FULL, device=DEVICE, dtype=torch.float32)
    window_ids = full_ids[ROWS]
    assert [t.tolist() for t in early] == [full_ids.tolist()]
    assert [t.tolist() for t in late] == [
        window_ids.tolist(),
        window_ids.tolist(),
        window_ids.tolist(),
    ]

    # Scatter writes the replay rows back and leaves the trimmed rows at zero.
    hidden = out["hidden_states"][:, 0, 0]
    assert torch.equal(hidden[ROWS], window_ids)
    assert hidden[: FULL - WINDOW].eq(0).all()

    # calls[0] is the cut-aux capture. calls[1] is the final mhc_post.
    # Both run inside the replay batch, before the scatter back to FULL.
    assert len(calls) == 2
    assert [t.shape[0] for t in calls] == [WINDOW, WINDOW]
    assert torch.equal(calls[0], window_ids)
    assert torch.equal(calls[1], window_ids)

    # The early loop skips this id. The replay loop does not see it either.
    assert saved_aux[0] == []
    assert saved_aux[1] == []
    _hidden, _pre_mix, aux = scattered[0]
    assert aux.shape[0] == FULL
    assert torch.equal(aux[ROWS, 0], window_ids)
    assert aux[: FULL - WINDOW].eq(0).all()
