# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.common.mtp import fused_mtp_input
from vllm.models.kimi_k3.nvidia.ops import attn_res
from vllm.platforms import current_platform

HIDDEN_SIZE = 7168
MAX_BLOCKS = 8
EPS = 1e-5


def _randn_with_row_padding(*shape: int, padding: int = 0) -> torch.Tensor:
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return storage[..., : shape[-1]]


def _reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if delta is not None:
        prefix = prefix + delta
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (HIDDEN_SIZE,), norm_weight, EPS)
    probs = (keys @ qk_weight).softmax(dim=-1)
    output = torch.matmul(probs.unsqueeze(1), values).squeeze(1)
    if output_norm_weight is not None:
        output = F.rms_norm(output, (HIDDEN_SIZE,), output_norm_weight, EPS)
    return output, prefix


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "row_padding",
        "write_block",
        "has_delta",
        "backend",
    ),
    [
        pytest.param(1, 0, 0, True, False, "triton", id="triton-empty"),
        pytest.param(1, 0, 0, True, True, "triton", id="triton-empty-add"),
        pytest.param(17, 5, 7, True, False, "triton", id="triton-write"),
        pytest.param(17, 5, 7, True, True, "triton", id="triton-write-add"),
        pytest.param(3, 8, 0, False, False, "triton", id="triton-full"),
        pytest.param(3, 8, 0, False, True, "triton", id="triton-full-add"),
        pytest.param(320, 1, 0, False, True, "nvidia", id="nvidia-1"),
        pytest.param(320, 4, 0, False, True, "nvidia", id="nvidia-4"),
        pytest.param(320, 8, 0, False, True, "nvidia", id="nvidia-8"),
    ],
)
def test_attn_res(
    num_tokens: int,
    num_blocks: int,
    row_padding: int,
    write_block: bool,
    has_delta: bool,
    backend: str,
):
    if backend == "nvidia" and not current_platform.is_device_capability_family(100):
        pytest.skip("NVIDIA AttnRes requires the SM100 family")

    prefix = _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
    delta = (
        _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=row_padding)
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens, MAX_BLOCKS, HIDDEN_SIZE, padding=row_padding
    )
    norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = 1 + 0.1 * torch.randn(
        HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    original_blocks = blocks.clone()
    expected, expected_prefix = _reference(
        prefix.clone(),
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )
    block_write_idx = num_blocks if write_block else -1

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    if write_block:
        original_blocks[:, block_write_idx].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize("num_blocks", range(MAX_BLOCKS + 1))
def test_attn_res_block_counts(num_blocks: int):
    prefix = torch.randn(1, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    blocks = torch.randn(
        1, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.ones(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    output_norm_weight = torch.ones_like(norm_weight)
    expected, _ = _reference(
        prefix.clone(),
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
    )

    actual = attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        -1,
        EPS,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


def test_attn_res_without_output_norm():
    prefix = torch.randn(7, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(prefix)
    blocks = torch.randn(
        7, MAX_BLOCKS, HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    qk_weight = (
        torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16) / HIDDEN_SIZE**0.5
    )
    expected, _ = _reference(
        prefix.clone(), delta, blocks, norm_weight, qk_weight, None, MAX_BLOCKS
    )

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        None,
        MAX_BLOCKS,
        -1,
        EPS,
        0.0,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)


@pytest.mark.parametrize("num_tokens", [0, 1, 17])
def test_fused_mtp_input(num_tokens: int):
    positions = torch.arange(num_tokens, device="cuda")
    inputs_embeds = _randn_with_row_padding(num_tokens, HIDDEN_SIZE, padding=7)
    previous_hidden_states = _randn_with_row_padding(
        num_tokens, HIDDEN_SIZE, padding=11
    )
    enorm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)
    hnorm_weight = torch.randn(HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16)

    masked_inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
    expected = torch.cat(
        (
            F.rms_norm(masked_inputs_embeds, (HIDDEN_SIZE,), enorm_weight, EPS),
            F.rms_norm(previous_hidden_states, (HIDDEN_SIZE,), hnorm_weight, EPS),
        ),
        dim=-1,
    )
    actual = fused_mtp_input(
        positions,
        inputs_embeds,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        EPS,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    assert actual.shape == (num_tokens, 2 * HIDDEN_SIZE)
    assert actual.is_contiguous()
