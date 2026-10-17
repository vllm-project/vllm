# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.models.kimi_k3.amd.ops.attn_res import attn_res
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(),
    reason="AMD AttnRes requires ROCm",
)


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
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    num_blocks: int,
    eps: float,
) -> torch.Tensor:
    hidden_size = prefix.shape[-1]
    values = torch.cat((blocks[:, :num_blocks], prefix.unsqueeze(1)), dim=1)
    keys = F.rms_norm(values, (hidden_size,), norm_weight, eps)
    probs = (keys @ qk_weight).softmax(dim=-1)
    return torch.matmul(probs.unsqueeze(1), values).squeeze(1)


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "block_capacity",
        "hidden_size",
        "row_padding",
    ),
    [
        pytest.param(0, 3, 5, 128, 0, id="empty"),
        pytest.param(1, 1, 2, 128, 0, id="decode-single"),
        pytest.param(17, 4, 6, 1024, 7, id="decode-padded"),
        pytest.param(255, 4, 6, 1024, 0, id="decode-limit"),
        pytest.param(256, 4, 6, 1024, 0, id="prefill-limit"),
        pytest.param(320, 8, 10, 7168, 0, id="prefill-full"),
    ],
)
def test_amd_attn_res_matches_reference(
    num_tokens: int,
    num_blocks: int,
    block_capacity: int,
    hidden_size: int,
    row_padding: int,
) -> None:
    eps = 1e-5
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=row_padding)
    blocks = _randn_with_row_padding(
        num_tokens,
        block_capacity,
        hidden_size,
        padding=row_padding,
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    )
    expected = _reference(
        prefix,
        blocks,
        norm_weight,
        qk_weight,
        num_blocks,
        eps,
    )
    original_prefix = prefix.clone()
    original_blocks = blocks.clone()

    actual = attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        None,
        num_blocks,
        -1,
        eps,
        0.0,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, original_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.shape == prefix.shape
    assert actual.is_contiguous()


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "hidden_size",
        "has_delta",
        "write_block",
        "apply_output_norm",
        "capture_prefix",
    ),
    [
        pytest.param(1, 0, 128, False, True, True, True, id="empty-write-norm"),
        pytest.param(7, 1, 1024, True, False, True, False, id="single-add-norm"),
        pytest.param(17, 5, 7168, True, True, True, True, id="padded-write-add"),
        pytest.param(3, 8, 7168, True, False, True, False, id="full-add-norm"),
        pytest.param(320, 4, 7168, True, False, False, True, id="prefill-add"),
        pytest.param(0, 3, 1024, True, False, True, True, id="no-token-add"),
        pytest.param(0, 3, 1024, False, False, True, True, id="no-token-copy"),
    ],
)
def test_amd_attn_res_fused_contract(
    num_tokens: int,
    num_blocks: int,
    hidden_size: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
    capture_prefix: bool,
) -> None:
    torch.manual_seed(42)
    eps = 1e-5
    output_eps = 2e-5
    block_capacity = 9
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=7)
    delta = (
        _randn_with_row_padding(num_tokens, hidden_size, padding=11)
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens, block_capacity, hidden_size, padding=13
    )
    norm_weight = 1 + 0.1 * torch.randn(
        hidden_size, device="cuda", dtype=torch.bfloat16
    )
    qk_weight = (
        torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) / hidden_size**0.5
    )
    output_norm_weight = (
        1 + 0.1 * torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        if apply_output_norm
        else None
    )
    expected_prefix = prefix.clone()
    if delta is not None:
        expected_prefix = expected_prefix + delta
    values = torch.cat(
        (blocks[:, :num_blocks].clone(), expected_prefix.unsqueeze(1)), dim=1
    )
    keys = F.rms_norm(values.float(), (hidden_size,), norm_weight.float(), eps)
    probs = (keys @ qk_weight.float()).softmax(dim=-1)
    expected = torch.matmul(probs.unsqueeze(1), values.float()).squeeze(1)
    if output_norm_weight is not None:
        expected = F.rms_norm(
            expected, (hidden_size,), output_norm_weight.float(), output_eps
        )
    expected = expected.to(prefix.dtype)
    original_blocks = blocks.clone()
    block_write_idx = num_blocks if write_block else -1
    # The auxiliary buffer is allocated contiguous, so its row pitch differs
    # from the padded prefix pitch the same launch reads.
    prefix_snapshot = torch.empty_like(prefix) if capture_prefix else None

    actual = attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        eps,
        output_eps,
        prefix_snapshot=prefix_snapshot,
    )

    torch.testing.assert_close(actual, expected, atol=8e-2, rtol=3e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    if prefix_snapshot is not None:
        torch.testing.assert_close(prefix_snapshot, expected_prefix, atol=0, rtol=0)
    if write_block:
        original_blocks[:, block_write_idx].copy_(expected_prefix)
    torch.testing.assert_close(blocks, original_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


@pytest.mark.parametrize(
    "num_tokens,num_blocks,has_delta,write_block,capture_prefix",
    [
        (0, 0, False, False, True),
        (1, 0, True, True, False),
        (17, 4, True, True, True),
        (320, 8, False, False, True),
    ],
)
def test_amd_attn_res_fp8_preserves_prefix_and_quantized_output(
    num_tokens,
    num_blocks,
    has_delta,
    write_block,
    capture_prefix,
    default_vllm_config,
):
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    torch.manual_seed(41)
    hidden_size = 7168
    prefix = _randn_with_row_padding(num_tokens, hidden_size, padding=7)
    delta = _randn_with_row_padding(num_tokens, hidden_size) if has_delta else None
    blocks = _randn_with_row_padding(num_tokens, 9, hidden_size, padding=13)
    norm = 1 + torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
    score = torch.randn_like(norm) / hidden_size**0.5
    out_norm = 1 + torch.randn_like(norm) * 0.1
    ref_prefix, ref_blocks = prefix.clone(), blocks.clone()
    kwargs = dict(
        num_blocks=num_blocks,
        block_write_idx=num_blocks if write_block else -1,
        eps=1e-5,
        output_norm_eps=1e-5,
    )
    reference = attn_res(ref_prefix, delta, ref_blocks, norm, score, out_norm, **kwargs)
    prefix_snapshot = torch.empty_like(prefix) if capture_prefix else None
    output, scale = attn_res(
        prefix,
        delta,
        blocks,
        norm,
        score,
        out_norm,
        quant_dtype=current_platform.fp8_dtype(),
        prefix_snapshot=prefix_snapshot,
        **kwargs,
    )
    assert output.dtype == current_platform.fp8_dtype()
    assert scale.shape == (num_tokens, 1)
    if num_tokens:
        # Keep reference rounding independent of Inductor's fused quantization.
        quant = QuantFP8(
            static=False, group_shape=GroupShape.PER_TOKEN, compile_native=False
        )
        ref_output, ref_scale = quant(reference)
        torch.testing.assert_close(scale, ref_scale, atol=1e-7, rtol=1e-6)

        # Floating-point fusion can move values at FP8 rounding midpoints.
        # Bound each change to the adjacent code and bound how often it occurs.
        # Codes are sign-magnitude, so order them by value to keep a step
        # across zero adjacent.
        def ordinal(x: torch.Tensor) -> torch.Tensor:
            code = x.view(torch.uint8).to(torch.int16)
            return torch.where(code >= 0x80, 0x80 - code, code)

        code_delta = (ordinal(output) - ordinal(ref_output)).abs()
        assert code_delta.max().item() <= 1
        assert (code_delta != 0).float().mean().item() < 1e-4
    torch.testing.assert_close(prefix, ref_prefix, atol=0, rtol=0)
    if prefix_snapshot is not None:
        torch.testing.assert_close(prefix_snapshot, ref_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, ref_blocks, atol=0, rtol=0)


@pytest.mark.parametrize(
    "matching,requires_unquantized", [(True, False), (False, False), (True, True)]
)
def test_mla_input_quant_requires_all_consumers(matching, requires_unquantized):
    from types import SimpleNamespace

    from torch import nn

    from vllm.model_executor.layers.fusion.quant_activation import (
        expose_input_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
    )
    from vllm.models.kimi_k3.amd.mla import KimiK3MultiHeadLatentAttentionWrapper

    attention = object.__new__(KimiK3MultiHeadLatentAttentionWrapper)
    nn.Module.__init__(attention)
    attention.indexer = None
    attention.q_lora_rank = 1536
    attention.fused_qkv_a_proj = nn.Identity()
    attention.g_proj = nn.Identity()
    kernel = SimpleNamespace(input_quant_key=lambda: kFp8DynamicTokenSym)
    expose_input_quant_key(attention.fused_qkv_a_proj, kernel)
    if matching:
        expose_input_quant_key(attention.g_proj, kernel)
    attention.g_proj.requires_unquantized_input = requires_unquantized
    actual = attention.get_input_quant_key()
    assert actual == (
        kFp8DynamicTokenSym if matching and not requires_unquantized else None
    )
