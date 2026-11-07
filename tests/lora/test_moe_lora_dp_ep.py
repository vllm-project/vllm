# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoE-LoRA with DP+EP: the batch the experts see is not this rank's batch.

``MoEPrepareAndFinalizeNaiveDPEP.prepare()`` all-gathers every DP rank's
tokens together with their token->slot mapping and hands the result to the
experts as ``MoELoRAContext.local_token_lora_mapping``. The punica kernel
metadata, the no-LoRA flags and the active-slot list all describe this rank's
own batch only, so none of them may gate the LoRA work for such a batch.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.lora.punica_wrapper.punica_gpu as punica_gpu
from vllm.lora.ops.triton_ops import LoRAKernelMeta
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmMxfp4LoRAExperts,
)
from vllm.platforms import current_platform

MAX_LORAS = 4


@pytest.fixture(autouse=True)
def cleanup_fixture():
    """Override conftest's cleanup_fixture - not needed for these tests."""
    yield


def test_prepare_tensors_refreshes_mapping_for_batch_without_lora():
    """A batch without LoRA must clear the shared token->slot buffer.

    The EP dispatch slices this buffer and ships it to the other DP ranks, so
    leaving the previous batch's slots in it applies that batch's adapters to
    these tokens on the other ranks.
    """
    max_num_tokens = 8
    meta = LoRAKernelMeta.make(MAX_LORAS, max_num_tokens, device="cpu")

    with_lora = torch.tensor([1, 1, 2, 2], dtype=torch.int32)
    meta.prepare_tensors(with_lora)
    assert not bool(meta.no_lora_flag_cpu[0].item())
    assert torch.equal(meta.token_lora_mapping[: with_lora.size(0)], with_lora)

    without_lora = torch.full((4,), -1, dtype=torch.int32)
    meta.prepare_tensors(without_lora)
    assert bool(meta.no_lora_flag_cpu[0].item())
    # Past the batch too: the buffer is sized for max_num_batched_tokens and
    # the previous batch may have been longer.
    assert torch.equal(
        meta.token_lora_mapping, torch.full((max_num_tokens,), -1, dtype=torch.int32)
    )


def test_batch_has_no_lora_is_false_for_a_dispatched_batch():
    """The local no-LoRA flag cannot clear a batch that came from the dispatch."""
    meta = SimpleNamespace(no_lora_flag_cpu=torch.tensor([True], dtype=torch.bool))
    wrapper = SimpleNamespace(token_mapping_meta=meta)

    local_batch = SimpleNamespace(punica_wrapper=wrapper, local_token_lora_mapping=None)
    assert TrtLlmMxfp4LoRAExperts._batch_has_no_lora(local_batch)

    dispatched_batch = SimpleNamespace(
        punica_wrapper=wrapper,
        local_token_lora_mapping=torch.full((4,), -1, dtype=torch.int32),
    )
    assert not TrtLlmMxfp4LoRAExperts._batch_has_no_lora(dispatched_batch)


def _make_wrapper(device: str, num_tokens: int) -> punica_gpu.PunicaWrapperGPU:
    lora_config = SimpleNamespace(max_loras=MAX_LORAS, specialize_active_lora=True)
    wrapper = punica_gpu.PunicaWrapperGPU(
        max_num_batched_tokens=num_tokens,
        max_batches=num_tokens,
        device=device,
        lora_config=lora_config,
    )
    # This rank's own batch holds no LoRA token.
    wrapper.token_mapping_meta.prepare_tensors(
        torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
    )
    assert bool(wrapper.token_mapping_meta.no_lora_flag_cpu[0].item())
    return wrapper


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
def test_add_lora_fused_moe_covers_all_slots_for_a_dispatched_mapping(monkeypatch):
    device = f"{current_platform.device_type}:0"
    num_tokens, num_experts, top_k, max_lora_rank = 16, 8, 2, 16
    K, N = 256, 128
    wrapper = _make_wrapper(device, num_tokens)

    # The dispatch gathered tokens that do select an adapter (slot 2).
    dispatched = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
    dispatched[:4] = 2

    captured: dict[str, torch.Tensor] = {}

    def fake_fused_moe_lora(
        output,
        x,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mapping,
        max_lora_rank,
        top_k_num,
        lora_ids,
        num_active_loras,
        *args,
        **kwargs,
    ):
        captured["token_lora_mapping"] = token_lora_mapping
        captured["lora_ids"] = lora_ids
        captured["num_active_loras"] = num_active_loras

    monkeypatch.setattr(punica_gpu, "fused_moe_lora", fake_fused_moe_lora)

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }
    wrapper.add_lora_fused_moe(
        y=torch.rand(num_tokens, top_k, N, dtype=torch.bfloat16, device=device),
        x=torch.rand(num_tokens, K, dtype=torch.bfloat16, device=device),
        lora_a_stacked=(
            torch.rand(
                MAX_LORAS,
                num_experts,
                max_lora_rank,
                K,
                dtype=torch.bfloat16,
                device=device,
            ),
        ),
        lora_b_stacked=(
            torch.rand(
                MAX_LORAS,
                num_experts,
                N,
                max_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ),
        ),
        topk_weights=torch.ones(num_tokens, top_k, dtype=torch.float32, device=device),
        sorted_token_ids=torch.zeros(
            num_tokens * top_k, dtype=torch.int32, device=device
        ),
        expert_ids=torch.zeros(num_tokens * top_k, dtype=torch.int32, device=device),
        num_tokens_post_padded=torch.tensor(
            [num_tokens * top_k], dtype=torch.int32, device=device
        ),
        max_lora_rank=max_lora_rank,
        top_k_num=top_k,
        shrink_config=config,
        expand_config=config,
        adapter_enabled=torch.ones(MAX_LORAS + 1, dtype=torch.int32, device=device),
        token_lora_mapping=dispatched,
    )

    assert captured, "took the local no-LoRA early exit for a dispatched batch"
    assert torch.equal(captured["token_lora_mapping"], dispatched)
    # Every slot, plus -1 for the tokens without an adapter.
    assert sorted(captured["lora_ids"].tolist()) == [-1, *range(MAX_LORAS)]
    assert int(captured["num_active_loras"].item()) == MAX_LORAS + 1


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like device"
)
def test_moe_lora_align_block_size_covers_all_slots_for_a_dispatched_mapping(
    monkeypatch,
):
    device = f"{current_platform.device_type}:0"
    num_tokens, num_experts, top_k = 16, 8, 2
    wrapper = _make_wrapper(device, num_tokens)

    dispatched = torch.full((num_tokens,), -1, dtype=torch.int32, device=device)
    dispatched[:4] = 2

    captured: dict[str, torch.Tensor] = {}

    def fake_align(*args, **kwargs):
        captured["lora_ids"] = args[11]

    monkeypatch.setattr(punica_gpu.ops, "moe_lora_align_block_size", fake_align)

    wrapper.moe_lora_align_block_size(
        topk_ids=torch.zeros(num_tokens, top_k, dtype=torch.int32, device=device),
        num_tokens=num_tokens,
        block_size=16,
        num_experts=num_experts,
        max_loras=MAX_LORAS,
        adapter_enabled=torch.ones(MAX_LORAS + 1, dtype=torch.int32, device=device),
        token_lora_mapping=dispatched,
    )

    assert sorted(captured["lora_ids"].tolist()) == [-1, *range(MAX_LORAS)]
