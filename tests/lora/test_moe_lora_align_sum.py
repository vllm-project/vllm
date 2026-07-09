# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

DEVICE_TYPE = current_platform.device_type


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def sample_data(num_experts, max_loras, num_tokens, topk_num):
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.zeros((num_tokens,), dtype=torch.int32)

    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        token_lora_mapping[i] = random.randint(0, max_loras - 1)

    return topk_ids.to(DEVICE_TYPE), token_lora_mapping.to(DEVICE_TYPE)


@pytest.mark.parametrize("num_tokens", [100, 200, 1024, 4096])  # 81920
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 128, 256, 512])
@pytest.mark.parametrize("max_loras", [2, 32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(
    num_tokens, topk_num, num_experts, max_loras, block_size
):
    # sample data
    random.seed(1)
    topk_ids, token_lora_mapping = sample_data(
        num_experts, max_loras, num_tokens, topk_num
    )

    # compute paddings
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        num_experts,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    num_tokens_post_pad = torch.zeros(
        (max_loras,), dtype=torch.int32, device=DEVICE_TYPE
    )
    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=DEVICE_TYPE
    )
    lora_ids = torch.arange(max_loras + 2, dtype=torch.int32, device=DEVICE_TYPE)

    # call kernel
    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    # verify values
    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1, block_size)

    for lora_idx in range(max_loras):
        for token_idx in range(sorted_token_ids.size(1)):
            block = sorted_token_ids[lora_idx][token_idx]
            indices = block[block != topk_ids.numel()]
            if indices.numel() > 0:
                expert_id = expert_ids[lora_idx][token_idx]
                assert torch.all(topk_ids.view(-1)[indices] == expert_id)


# Sentinel values for the regression tests below. Distinctive out-of-domain
# ints so that "kernel never wrote this slot" is directly observable: the
# kernel only ever writes a real expert id in [0, num_experts) or -1
# (expert_ids), a token index or the `numel` padding value (sorted_token_ids),
# and a block-aligned cumsum count (num_tokens_post_pad).
SENTINEL_EXPERT = -2
SENTINEL_TOKEN = -7
SENTINEL_NPAD = -13


def _build_and_run_align(
    *,
    num_lora_tokens,
    num_base_tokens,
    max_loras,
    num_experts=64,
    topk_num=6,
    block_size=16,
    lora_ids_override=None,
    disabled_slots=(),
    seed=1,
):
    """Build inputs the way ``LoRAKernelMeta.prepare_tensors`` does, run
    ``moe_lora_align_block_size``, and return a dict of result tensors plus
    derived sizes. Output buffers are pre-filled with ``SENTINEL_*`` so
    callers can assert which slots the kernel did / did not touch.

    Tokens are assigned to LoRA slot 0 (first ``num_lora_tokens``) then -1
    (remaining ``num_base_tokens``), matching the "mixed base + 1 LoRA"
    shape used to repro vllm-project/vllm#32235.

    ``lora_ids_override``: optional 1-D int tensor of length ``max_loras+1``
    used verbatim. Default mirrors ``prepare_tensors`` (sorted-unique into
    the head, -1 tail).
    ``disabled_slots``: iterable of slot indices to clear in ``adapter_enabled``.
    """
    random.seed(seed)
    num_tokens = num_lora_tokens + num_base_tokens
    assert num_tokens > 0, "test requires at least one token"

    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    token_lora_mapping = torch.empty((num_tokens,), dtype=torch.int32)
    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]
        token_lora_mapping[i] = 0 if i < num_lora_tokens else -1
    topk_ids = topk_ids.to(DEVICE_TYPE)
    token_lora_mapping = token_lora_mapping.to(DEVICE_TYPE)

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    if lora_ids_override is None:
        lora_ids = torch.full(
            (max_loras + 1,), -1, dtype=torch.int32, device=DEVICE_TYPE
        )
        unique_ids = torch.unique(token_lora_mapping, sorted=True)
        lora_ids[: unique_ids.numel()] = unique_ids.to(torch.int32)
    else:
        assert lora_ids_override.numel() == max_loras + 1
        lora_ids = lora_ids_override.to(dtype=torch.int32, device=DEVICE_TYPE)

    adapter_enabled = torch.ones(
        (max_loras + 1,), dtype=torch.int32, device=DEVICE_TYPE
    )
    for slot in disabled_slots:
        adapter_enabled[slot] = 0

    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        SENTINEL_TOKEN,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,),
        SENTINEL_EXPERT,
        dtype=torch.int32,
        device=DEVICE_TYPE,
    )
    num_tokens_post_pad = torch.full(
        (max_loras,), SENTINEL_NPAD, dtype=torch.int32, device=DEVICE_TYPE
    )

    ops.moe_lora_align_block_size(
        topk_ids,
        token_lora_mapping,
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
    )

    return {
        "lora_ids": lora_ids,
        "sorted_token_ids": sorted_token_ids,
        "expert_ids": expert_ids,
        "num_tokens_post_pad": num_tokens_post_pad,
        "max_num_tokens_padded": max_num_tokens_padded,
        "block_size": block_size,
        "max_loras": max_loras,
    }


@pytest.mark.parametrize(
    "max_loras",
    [
        1,
        2,
    ],
)
def test_moe_lora_align_block_size_mixed_base_and_lora(max_loras):
    """Regression test for issue #32235: real LoRA slot must not be skipped
    when ``active_lora_ids`` has -1 at position 0."""
    out = _build_and_run_align(
        num_lora_tokens=8, num_base_tokens=8, max_loras=max_loras
    )

    # Sanity check on the layout being tested.
    assert out["lora_ids"][0].item() == -1, (
        "prepare_tensors layout mismatch: -1 expected at position 0 for mixed batch"
    )

    real_slot = 0
    post_pad = out["num_tokens_post_pad"][real_slot].item()
    assert post_pad != SENTINEL_NPAD, (
        f"num_tokens_post_pad[{real_slot}] was never written by the kernel; "
        "the align kernel skipped the real LoRA slot."
    )
    assert (
        0 < post_pad <= out["max_num_tokens_padded"]
        and post_pad % out["block_size"] == 0
    ), f"num_tokens_post_pad[{real_slot}]={post_pad} is not a valid block-aligned count"

    expert_row = out["expert_ids"].view(max_loras, -1)[real_slot]
    assert (expert_row != SENTINEL_EXPERT).all(), (
        f"expert_ids row for slot {real_slot} has unwritten sentinel entries; "
        "the align kernel skipped the real LoRA slot."
    )

    sorted_row = out["sorted_token_ids"].view(max_loras, -1)[real_slot]
    assert (sorted_row != SENTINEL_TOKEN).all(), (
        f"sorted_token_ids row for slot {real_slot} has unwritten sentinel "
        "entries; the align kernel skipped the real LoRA slot."
    )


def test_moe_lora_align_block_size_disabled_adapter_untouched():
    """Disabled-adapter slot rows must remain untouched by all three align
    kernels. Pins the invariant protected by the ``adapter_enabled`` guard
    in ``lora_count_and_sort_expert_tokens_kernel``: without it the sort
    kernel reads uninitialized ``token_mask`` values for disabled slots and
    pollutes ``sorted_token_ids`` / ``cumsum_buffer``."""
    max_loras = 1
    out = _build_and_run_align(
        num_lora_tokens=16,
        num_base_tokens=0,
        max_loras=max_loras,
        disabled_slots=(0,),
    )
    # Sanity: slot 0 IS present in active_lora_ids (otherwise we would only
    # exercise the lora_id == -1 / >= max_loras guards).
    assert (out["lora_ids"] == 0).any().item()

    assert out["num_tokens_post_pad"][0].item() == SENTINEL_NPAD, (
        "num_tokens_post_pad[0] was modified for a disabled adapter slot."
    )
    expert_row = out["expert_ids"].view(max_loras, -1)[0]
    assert (expert_row == SENTINEL_EXPERT).all(), (
        "expert_ids row for disabled slot 0 was partially written."
    )
    # Row specifically protected by the sort-kernel adapter_enabled guard.
    sorted_row = out["sorted_token_ids"].view(max_loras, -1)[0]
    assert (sorted_row == SENTINEL_TOKEN).all(), (
        "sorted_token_ids row for disabled slot 0 was polluted by the sort "
        "kernel; lora_count_and_sort_expert_tokens_kernel must skip "
        "adapter_enabled == 0 slots."
    )


def test_moe_lora_align_block_size_lora_id_oob_guard():
    """Regression test for the ``lora_id >= max_loras`` guard.

    Production ``LoRAKernelMeta.prepare_tensors`` pre-fills the tail of
    ``active_lora_ids`` with -1, so the existing ``lora_id == -1`` check
    covers the extra slot. This test bypasses that invariant and injects
    an out-of-range value (5 with max_loras=1) at the tail to verify the
    explicit guard prevents OOB reads against ``adapter_enabled`` and
    OOB writes against the max_loras-sized output buffers. Without the
    guard, an illegal-memory-access would surface on the next CUDA sync.
    """
    max_loras = 1
    lora_ids_override = torch.tensor([0, 5], dtype=torch.int32)
    out = _build_and_run_align(
        num_lora_tokens=16,
        num_base_tokens=0,
        max_loras=max_loras,
        lora_ids_override=lora_ids_override,
    )
    # The .item() call below syncs and would surface any async
    # illegal-memory-access from the OOB iteration.
    assert out["num_tokens_post_pad"][0].item() != SENTINEL_NPAD, (
        "real LoRA slot 0 was skipped by the align kernel"
    )


if __name__ == "__main__":
    pytest.main([__file__])
