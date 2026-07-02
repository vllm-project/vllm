# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops


_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _deterministic_uniform01(
    seed: int,
    offset: int,
    row: int,
    vocab_idx: int,
) -> float:
    key = int(seed) & _MASK64
    key ^= (int(offset) * 0x9E3779B97F4A7C15) & _MASK64
    key ^= (int(row) * 0xBF58476D1CE4E5B9) & _MASK64
    key ^= (int(vocab_idx) * 0x94D049BB133111EB) & _MASK64
    mantissa = (_splitmix64(key) >> 40) & 0xFFFFFF
    return (float(mantissa) + 0.5) * (2.0**-24)


def _deterministic_gumbels(
    rows: int,
    vocab_size: int,
    *,
    vocab_start_index: int,
    rng_seed: int,
    rng_offset: int,
) -> torch.Tensor:
    uniforms = torch.empty(rows, vocab_size, dtype=torch.float32)
    for row in range(rows):
        for vocab_offset in range(vocab_size):
            uniforms[row, vocab_offset] = _deterministic_uniform01(
                rng_seed,
                rng_offset,
                row,
                vocab_start_index + vocab_offset,
            )
    return -torch.log(-torch.log(uniforms))


def _require_native_flashdenoise_op(name: str) -> None:
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, name):
        pytest.skip(f"native op {name} is unavailable in this vLLM extension")


def _dense_local_state_reference(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    logit_scale: torch.Tensor,
    vocab_start_index: int,
    final_logit_softcapping: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.mm(hidden, lm_head_weight.t(), out_dtype=torch.float32)
    if final_logit_softcapping > 0.0:
        logits = final_logit_softcapping * torch.tanh(
            logits / final_logit_softcapping
        )
    logits = logits * logit_scale.reshape(-1, 1)

    local_max, local_indices = logits.max(dim=-1)
    local_exp = torch.exp(logits - local_max.unsqueeze(-1))
    local_sum_exp = local_exp.sum(dim=-1)
    local_weighted_logits = (local_exp * logits).sum(dim=-1)
    local_soft_part = torch.mm(
        local_exp.to(torch.bfloat16), lm_head_weight, out_dtype=torch.float32
    )
    clean_indices = local_indices.to(torch.int64) + vocab_start_index
    gumbels = _deterministic_gumbels(
        hidden.shape[0],
        lm_head_weight.shape[0],
        vocab_start_index=vocab_start_index,
        rng_seed=1234,
        rng_offset=99,
    ).to(logits.device)
    noisy = logits + gumbels
    sample_values, sample_local_indices = noisy.max(dim=-1)
    sample_indices = sample_local_indices.to(torch.int64) + vocab_start_index
    return (
        local_max,
        local_sum_exp,
        local_weighted_logits,
        local_soft_part,
        clean_indices,
        sample_values,
        sample_indices,
    )


@pytest.mark.parametrize("scale_mode", ["per_row", "scalar"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_state_scaled_matches_dense_reference_for_local_vocab_shard(
    scale_mode: str,
):
    _require_native_flashdenoise_op(
        "diffusion_gemma_flashdenoise_local_state_scaled"
    )

    rows, hidden_size, local_vocab = 7, 32, 40
    vocab_start_index = 320
    final_logit_softcapping = 1.3
    generator = torch.Generator(device="cuda").manual_seed(20250621)

    hidden = (
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.35
    ).to(torch.bfloat16)
    lm_head_weight = (
        torch.randn(
            local_vocab,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.35
    ).to(torch.bfloat16)
    if scale_mode == "per_row":
        logit_scale = torch.linspace(
            0.55, 1.85, rows, device="cuda", dtype=torch.float32
        )
    else:
        logit_scale = torch.tensor(1.15, device="cuda", dtype=torch.float32)

    local_max = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_sum_exp = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_weighted_logits = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_soft_part = torch.empty(
        rows, hidden_size, device="cuda", dtype=torch.float32
    )
    clean_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    clean_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    sample_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_indices = torch.empty(rows, device="cuda", dtype=torch.int64)

    ops.diffusion_gemma_flashdenoise_local_state_scaled(
        local_max,
        local_sum_exp,
        local_weighted_logits,
        local_soft_part,
        clean_values,
        clean_indices,
        sample_values,
        sample_indices,
        hidden,
        lm_head_weight,
        logit_scale,
        vocab_start_index,
        final_logit_softcapping,
        rng_seed=1234,
        rng_offset=99,
    )
    torch.cuda.synchronize()

    expected = _dense_local_state_reference(
        hidden,
        lm_head_weight,
        logit_scale,
        vocab_start_index,
        final_logit_softcapping,
    )

    torch.testing.assert_close(
        local_max.cpu(), expected[0].cpu(), atol=2e-3, rtol=0
    )
    torch.testing.assert_close(
        clean_values.cpu(), expected[0].cpu(), atol=2e-3, rtol=0
    )
    torch.testing.assert_close(
        local_sum_exp.cpu(), expected[1].cpu(), atol=3e-3, rtol=3e-3
    )
    torch.testing.assert_close(
        local_weighted_logits.cpu(), expected[2].cpu(), atol=3e-3, rtol=3e-3
    )
    torch.testing.assert_close(
        local_soft_part.cpu(), expected[3].cpu(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(clean_indices.cpu(), expected[4].cpu())
    torch.testing.assert_close(
        sample_values.cpu(), expected[5].cpu(), atol=3e-3, rtol=0
    )
    torch.testing.assert_close(sample_indices.cpu(), expected[6].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_local_state_soft_part_uses_bf16_tensor_core_semantics():
    _require_native_flashdenoise_op(
        "diffusion_gemma_flashdenoise_local_state_scaled"
    )

    rows, hidden_size, local_vocab = 5, 64, 128
    vocab_start_index = 4096
    generator = torch.Generator(device="cuda").manual_seed(20260622)
    hidden = (
        torch.randn(
            rows,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.45
    ).to(torch.bfloat16)
    lm_head_weight = (
        torch.randn(
            local_vocab,
            hidden_size,
            device="cuda",
            dtype=torch.float32,
            generator=generator,
        )
        * 0.7
    ).to(torch.bfloat16)
    logit_scale = torch.linspace(0.7, 1.9, rows, device="cuda", dtype=torch.float32)

    local_max = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_sum_exp = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_weighted_logits = torch.empty(rows, device="cuda", dtype=torch.float32)
    local_soft_part = torch.empty(
        rows, hidden_size, device="cuda", dtype=torch.float32
    )
    clean_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    clean_indices = torch.empty(rows, device="cuda", dtype=torch.int64)
    sample_values = torch.empty(rows, device="cuda", dtype=torch.float32)
    sample_indices = torch.empty(rows, device="cuda", dtype=torch.int64)

    ops.diffusion_gemma_flashdenoise_local_state_scaled(
        local_max,
        local_sum_exp,
        local_weighted_logits,
        local_soft_part,
        clean_values,
        clean_indices,
        sample_values,
        sample_indices,
        hidden,
        lm_head_weight,
        logit_scale,
        vocab_start_index,
        0.0,
        rng_seed=1234,
        rng_offset=99,
    )
    torch.cuda.synchronize()

    logits = torch.mm(hidden, lm_head_weight.t(), out_dtype=torch.float32)
    logits = logits * logit_scale.unsqueeze(-1)
    exp_weights = torch.exp(logits - logits.max(dim=-1).values.unsqueeze(-1))
    expected_soft_part = torch.mm(
        exp_weights.to(torch.bfloat16), lm_head_weight, out_dtype=torch.float32
    )

    torch.testing.assert_close(
        local_soft_part.cpu(),
        expected_soft_part.cpu(),
        atol=1e-3,
        rtol=2e-4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pack_local_state_matches_torch_merge_scale():
    _require_native_flashdenoise_op("diffusion_gemma_flashdenoise_pack_local_state")

    rows, hidden_size = 6, 17
    generator = torch.Generator(device="cuda").manual_seed(20260623)
    local_max = torch.randn(rows, device="cuda", generator=generator)
    global_max = local_max + torch.rand(
        rows, device="cuda", generator=generator
    )
    local_sum_exp = (
        torch.rand(rows, device="cuda", generator=generator) * 3.0 + 0.1
    )
    local_weighted_logits = torch.randn(
        rows, device="cuda", generator=generator
    )
    local_soft_part = torch.randn(
        rows, hidden_size, device="cuda", generator=generator
    )
    packed = torch.empty(
        rows, hidden_size + 2, device="cuda", dtype=torch.float32
    )

    ops.diffusion_gemma_flashdenoise_pack_local_state(
        packed,
        local_max,
        global_max,
        local_sum_exp,
        local_weighted_logits,
        local_soft_part,
    )
    torch.cuda.synchronize()

    scale = torch.exp(local_max - global_max)
    expected = torch.cat(
        [
            (local_sum_exp * scale).unsqueeze(-1),
            (local_weighted_logits * scale).unsqueeze(-1),
            local_soft_part * scale.unsqueeze(-1),
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        packed.cpu(),
        expected.cpu(),
        atol=1e-6,
        rtol=1e-6,
    )
