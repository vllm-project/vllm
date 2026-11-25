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
from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
from vllm.platforms import current_platform
from vllm.v1.attention.backends.short_conv_attn import ShortConvAttentionMetadata
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

if not (current_platform.is_cuda_alike() or current_platform.is_cpu()):
    pytest.skip(
        "ShortConv checkpoints need CPU or CUDA-like ops", allow_module_level=True
    )

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


def _forward(
    layer,
    vllm_config,
    hidden,
    num_decodes,
    prefill_lens,
    slots,
    ckpt,
    context_lens=None,
):
    """Run one step: `num_decodes` decode tokens, then the prefills."""
    device = hidden.device
    qsl_p = torch.tensor([0, *prefill_lens], dtype=torch.int32).cumsum(0)
    nums_dict, batch_ptr, offset_ptr = compute_causal_conv1d_metadata(
        qsl_p.to(torch.int32), device=device
    )
    num_prefills = len(prefill_lens)
    if context_lens is None:
        context_lens = [0] * num_prefills
    metadata = ShortConvAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=sum(prefill_lens),
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_reqs=num_decodes + num_prefills,
        has_initial_states_p=torch.tensor(
            [length > 0 for length in context_lens], dtype=torch.bool, device=device
        ),
        query_start_loc_p=qsl_p.to(torch.int32).to(device),
        state_indices_tensor_p=torch.tensor(
            slots[num_decodes:], dtype=torch.int32, device=device
        ),
        state_indices_tensor_d=torch.tensor(
            slots[:num_decodes], dtype=torch.int32, device=device
        ).view(-1, 1),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        seq_lens=torch.tensor(
            [1] * num_decodes
            + [c + p for c, p in zip(context_lens, prefill_lens, strict=True)],
            device=device,
        ),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=offset_ptr,
        checkpoint=ckpt,
    )
    output = torch.empty_like(hidden)
    with set_forward_context({PREFIX: metadata}, vllm_config):
        layer(hidden, output)
    return output


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
    # Full and truncated prefills can select different GEMM algorithms. Small
    # integers make the input projection and the B*x cache window exact.
    with torch.no_grad():
        layer.in_proj.weight.round_().clamp_(-2, 2)
    if current_platform.is_cpu():
        dispatch_cpu_unquantized_gemm(layer.in_proj, remove_weight=False)
        dispatch_cpu_unquantized_gemm(layer.out_proj, remove_weight=False)
    assert layer.get_kv_cache_checkpoint_alignment(vllm_config) == 1

    dim_first = is_conv_state_dim_first()
    shape = (8, DIM, L_CACHE - 1) if dim_first else (8, L_CACHE - 1, DIM)
    initial = torch.randn(shape, device=device)
    layer.kv_cache = (initial.clone(),)

    def run(hidden, num_decodes, prefill_lens, slots, ckpt=None, context_lens=None):
        layer.kv_cache[0].copy_(initial)
        _forward(
            layer,
            vllm_config,
            hidden,
            num_decodes,
            prefill_lens,
            slots,
            ckpt,
            context_lens,
        )
        state = layer.kv_cache[0].clone()
        return state if dim_first else state.transpose(-1, -2)

    prefill_lens, offsets, ckpt_slots = [200, 130, 8], [64, 128, 0], [5, 6, 0]
    context_lens = [0, 0, 0]
    if current_platform.is_cpu():
        # At offset 1 the two-token convolution window still contains one
        # token from the initialized cache. CUDA requires a full new window.
        prefill_lens[1], offsets[1], context_lens[1] = 8, 1, 127
    hidden = torch.randint(-2, 3, (1 + sum(prefill_lens), DIM), device=device).to(
        layer.in_proj.weight.dtype
    )
    ckpt = MambaPrefillCheckpointMetadata(
        checkpoint_offsets=torch.tensor(offsets, dtype=torch.int32, device=device),
        state_indices=torch.tensor(ckpt_slots, dtype=torch.int32, device=device),
        offsets=offsets,
    )
    slots = [1, 2, 3, 4]
    without_checkpoint = run(hidden, 1, prefill_lens, slots, context_lens=context_lens)
    with_checkpoint = run(hidden, 1, prefill_lens, slots, ckpt, context_lens)

    starts = [1, 1 + prefill_lens[0]]
    next_hidden = torch.randn(1, DIM, device=device)
    for row, start in enumerate(starts):
        offset, slot, source = offsets[row], ckpt_slots[row], slots[row + 1]
        stopped = run(
            hidden[start : start + offset],
            0,
            [offset],
            [source],
            context_lens=[context_lens[row]],
        )
        if offset < L_CACHE - 1:
            old = initial[source] if dim_first else initial[source].T
            torch.testing.assert_close(
                with_checkpoint[slot, :, :-offset], old[:, offset:], rtol=0, atol=0
            )
        torch.testing.assert_close(
            with_checkpoint[slot], stopped[source], rtol=0, atol=0
        )
        stored = layer.kv_cache[0] if dim_first else layer.kv_cache[0].transpose(-1, -2)
        stored[slot].copy_(with_checkpoint[slot])
        # Resume both the exported state and the independently stopped state.
        decoded = _forward(
            layer, vllm_config, next_hidden.expand(2, -1), 2, [], [slot, source], None
        )
        torch.testing.assert_close(decoded[0], decoded[1])
        with_checkpoint[slot] = without_checkpoint[slot]
    torch.testing.assert_close(with_checkpoint, without_checkpoint, rtol=0, atol=0)
