# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma must prefill image tokens bidirectionally (#62705)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import vllm.forward_context as forward_context_mod
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import ForwardContext
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
    DiffusionGemmaModelState,
)
from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.attention.backends import flash_attn as flash_attn_mod
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

LAYER = "model.layers.{}.self_attn.attn"


class _RecordingGroup:
    layer_names = [LAYER.format(0)]
    kv_cache_spec = object()

    def __init__(self):
        self.seen = []

    def get_metadata_builder(self, ubatch_idx=0):
        return SimpleNamespace(supports_update_block_table=False)

    def build_metadata(self, common_attn_metadata, ubatch_idx=0, **kwargs):
        self.seen.append(common_attn_metadata)
        return common_attn_metadata


def test_prepare_attn_passes_image_ranges():
    # Request 0 prefills an image wider than the 1024-token window (1120 soft
    # tokens); request 1 is denoising. The batch buffers are padded past
    # num_reqs, as in the runner.
    state = object.__new__(DiffusionGemmaModelState)
    state.supports_mm_inputs = True
    state.model = object.__new__(DiffusionGemmaForConditionalGeneration)
    state.model_config = SimpleNamespace(
        max_model_len=4096,
        is_mm_prefix_lm=True,
        get_sliding_window=lambda: 1024,
        hf_text_config=SimpleNamespace(),
    )
    image = SimpleNamespace(
        modality="image", mm_position=PlaceholderRange(offset=4, length=1120)
    )
    state.encoder_cache = SimpleNamespace(mm_features={"a": [image], "b": []})
    state.diffusion_states = SimpleNamespace(
        is_encoder_phase=torch.tensor([1, 0], dtype=torch.int32)
    )
    state._causal_buf = torch.zeros(4, dtype=torch.int32)

    qsl = np.array([0, 1130, 1162, 1162, 1162], dtype=np.int32)
    input_batch = SimpleNamespace(
        req_ids=["a", "b"],
        num_reqs=2,
        num_tokens=1162,
        idx_mapping=torch.tensor([0, 1]),
        query_start_loc_np=qsl,
        query_start_loc=torch.from_numpy(qsl.copy()),
        num_scheduled_tokens=np.array([1130, 32], dtype=np.int32),
        seq_lens=torch.tensor([1130, 288, 0, 0], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1130, 288, 0, 0], dtype=torch.int32),
        is_prefilling_np=np.array([True, False]),
    )
    group = _RecordingGroup()

    state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=[torch.zeros((2, 4), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, 1162), dtype=torch.int64),
        attn_groups=[[group]],
        kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
    )

    (common,) = group.seen
    assert common.mm_req_doc_ranges == {0: [(4, 1123)], 1: []}
    assert common.seq_lens_cpu_upper_bound.tolist() == [1130, 288]
    assert common.query_start_loc_cpu.tolist() == [0, 1130, 1162]
    assert common.query_start_loc.tolist() == [0, 1130, 1162]
    assert common.is_prefilling.tolist() == [True, False]
    assert common.causal.tolist() == [1, 0]


def test_forward_clears_image_ranges_on_full_attention_layers(monkeypatch):
    model = object.__new__(DiffusionGemmaForConditionalGeneration)
    nn.Module.__init__(model)
    model._full_attn_layer_idxs = frozenset({1, 3})

    metadata = {
        LAYER.format(i): SimpleNamespace(
            mm_prefix_range={0: [(4, 283)]},
            mm_prefix_range_tensor=torch.tensor([[[4, 283]]]),
            mm_prefix_query_range_tensor=torch.tensor([[4, 283]]),
        )
        for i in range(4)
    }
    seen = {}

    class Backbone(nn.Module):
        def forward(self, **kwargs):
            for i in range(4):
                md = metadata[LAYER.format(i)]
                seen[i] = (
                    md.mm_prefix_range,
                    md.mm_prefix_range_tensor,
                    md.mm_prefix_query_range_tensor,
                )
            return torch.zeros(2, 4)

    model.model = Backbone()
    monkeypatch.setattr(
        forward_context_mod,
        "_forward_context",
        ForwardContext(no_compile_layers={}, attn_metadata=metadata, slot_mapping={}),
    )

    model.forward(input_ids=torch.zeros(2, dtype=torch.long), positions=torch.arange(2))

    for i in (0, 2):
        assert all(x is not None for x in seen[i])
    for i in (1, 3):
        assert seen[i] == (None, None, None)


@pytest.mark.parametrize("has_image", [True, False])
def test_fa4_keeps_image_mask_with_per_request_causal(monkeypatch, has_image):
    # The FA4 call is stubbed out: this checks the arguments forward() picks
    # when attn_metadata.causal is a per-request tensor. The kernel itself
    # needs SM90/SM100.
    impl = object.__new__(FlashAttentionImpl)
    impl.vllm_flash_attn_version = 4
    impl.attn_type = flash_attn_mod.AttentionType.DECODER
    impl.head_size = 256
    impl.num_heads = 8
    impl.num_kv_heads = 4
    impl.scale = 256**-0.5
    impl.sliding_window = (1023, 0)
    impl.alibi_slopes = None
    impl.logits_soft_cap = 0
    impl.kv_cache_dtype = "auto"
    impl.supports_quant_query_input = False
    impl.fa4_hd256 = False
    impl.sinks = None
    impl.dcp_world_size = 1

    mask_mod = object()
    mask_mod_args = []
    calls = []

    def make_mask_mod(**kwargs):
        mask_mod_args.append(kwargs)
        return mask_mod

    monkeypatch.setattr(flash_attn_mod, "_make_mm_prefix_mask_mod", make_mask_mod)
    monkeypatch.setattr(
        flash_attn_mod, "_FA4_DENSE_ATTENTION_KERNEL", lambda **kw: calls.append(kw)
    )

    qsl = torch.tensor([0, 4, 6], dtype=torch.int32)
    causal = torch.tensor([1, 0], dtype=torch.int32)
    ranges = (
        torch.tensor([[1, 3]] * 3 + [[-1, -1]] * 3, dtype=torch.int32)
        if has_image
        else None
    )
    md = FlashAttentionMetadata(
        num_actual_tokens=6,
        max_query_len=4,
        query_start_loc=qsl,
        max_seq_len=16,
        seq_lens=torch.tensor([4, 10], dtype=torch.int32),
        block_table=torch.zeros((2, 2), dtype=torch.int32),
        slot_mapping=torch.arange(6),
        causal=causal,
        mm_prefix_query_range_tensor=ranges,
    )
    one = torch.tensor(1.0)
    layer = SimpleNamespace(
        _q_scale=one,
        _q_scale_float=1.0,
        _k_scale=one,
        _k_scale_float=1.0,
        _v_scale=one,
        _v_scale_float=1.0,
        mm_prefix_clamp_sliding_window=True,
    )
    q = torch.zeros(6, 8, 256)
    kv = torch.zeros(6, 4, 256)

    impl.forward(layer, q, kv, kv, torch.zeros(4, 4, 16, 512), md, torch.zeros_like(q))

    (call,) = calls
    if has_image:
        assert mask_mod_args == [
            dict(sliding_window=1024, sliding_window_left=1024, use_dynamic_causal=True)
        ]
        assert call["mask_mod"] is mask_mod
        assert call["aux_tensors"] == [ranges, qsl, causal]
        assert call["causal"] is False
        assert call["window_size"] is None
        assert call["dynamic_causal"] is None
    else:
        # Unchanged path: FA4's own per-request causal and symmetric window.
        assert mask_mod_args == []
        assert call["mask_mod"] is None
        assert call["dynamic_causal"] is causal
        assert call["window_size"] == [1023, 1023]
