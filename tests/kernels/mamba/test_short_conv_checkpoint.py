# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointMetadata
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_state_layout,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.short_conv import ShortConv
from vllm.platforms import current_platform
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

if not current_platform.is_cuda_alike():
    pytest.skip("ShortConv checkpoints need the Triton path", allow_module_level=True)

DIM, L_CACHE, PREFIX = 64, 3, "short_conv"


@pytest.fixture(autouse=True)
def mock_dist():
    with (
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.parallel_state.model_parallel_is_initialized",
            return_value=True,
        ),
        patch(
            "vllm.distributed.parallel_state.get_tp_group",
            return_value=MagicMock(rank_in_group=0),
        ),
    ):
        yield


def _forward(layer, vllm_config, hidden, num_decodes, prefill_lens, slots, ckpt):
    """Run one step: `num_decodes` decode tokens, then the prefills."""
    device = hidden.device
    qsl_p = torch.tensor([0, *prefill_lens], dtype=torch.int32).cumsum(0)
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(
        qsl_p.to(torch.int32), device=device
    )
    num_prefills = len(prefill_lens)
    metadata = ShortConvAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=sum(prefill_lens),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes + num_prefills,
        has_initial_states_p=torch.zeros(num_prefills, dtype=torch.bool, device=device),
        query_start_loc_p=qsl_p.to(torch.int32).to(device),
        state_indices_tensor_p=torch.tensor(
            slots[num_decodes:], dtype=torch.int32, device=device
        ),
        state_indices_tensor_d=torch.tensor(
            slots[:num_decodes], dtype=torch.int32, device=device
        ).view(-1, 1),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        seq_lens=torch.tensor([1] * num_decodes + list(prefill_lens), device=device),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offset_ptr,
        checkpoint=ckpt,
    )
    output = torch.empty_like(hidden)
    with set_forward_context({PREFIX: metadata}, vllm_config):
        layer.forward_cuda(hidden, output)


@pytest.fixture(params=["SD", "DS"])
def conv_layout(request, monkeypatch):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", request.param)
    get_conv_state_layout.cache_clear()
    yield
    get_conv_state_layout.cache_clear()


@pytest.mark.usefixtures("conv_layout")
def test_prefill_exports_conv_window_at_checkpoint():
    """A checkpoint holds the conv state of the prefill stopped at its offset.

    A decode leads the batch, so the prefills' inputs start after its token.
    Nothing else in the cache changes.
    """
    device = torch.device(current_platform.device_type)
    vllm_config = VllmConfig(compilation_config=CompilationConfig())
    config = SimpleNamespace(conv_L_cache=L_CACHE, conv_bias=False)
    with set_current_vllm_config(vllm_config):
        layer = ShortConv(config=config, dim=DIM, layer_idx=0, prefix=PREFIX)
    layer.to(device)
    torch.manual_seed(0)
    for p in layer.parameters():
        torch.nn.init.normal_(p)

    dim_first = is_conv_state_dim_first()
    shape = (8, DIM, L_CACHE - 1) if dim_first else (8, L_CACHE - 1, DIM)
    initial = torch.randn(shape, device=device)
    layer.kv_cache = (initial.clone(),)

    def run(hidden, num_decodes, prefill_lens, slots, ckpt=None):
        layer.kv_cache[0].copy_(initial)
        _forward(layer, vllm_config, hidden, num_decodes, prefill_lens, slots, ckpt)
        state = layer.kv_cache[0].clone()
        return state if dim_first else state.transpose(-1, -2)

    prefill_lens, offsets, ckpt_slots = [200, 130], [64, 128], [5, 6]
    hidden = torch.randn(1 + sum(prefill_lens), DIM, device=device)
    ckpt = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        state_indices=torch.tensor(ckpt_slots, dtype=torch.int32, device=device),
        offsets=offsets,
    )
    without_checkpoint = run(hidden, 1, prefill_lens, [1, 2, 3])
    with_checkpoint = run(hidden, 1, prefill_lens, [1, 2, 3], ckpt)

    starts = [1, 1 + prefill_lens[0]]
    for start, offset, slot in zip(starts, offsets, ckpt_slots):
        stopped = run(hidden[start : start + offset], 0, [offset], [7])
        torch.testing.assert_close(with_checkpoint[slot], stopped[7])
        with_checkpoint[slot] = without_checkpoint[slot]
    torch.testing.assert_close(with_checkpoint, without_checkpoint, rtol=0, atol=0)
