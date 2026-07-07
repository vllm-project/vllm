# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Distributed TP4 correctness test for the native DiffusionGemma FlashDenoise
# sampler-state contract. Run exactly with this shell command:
#   torchrun --standalone --nproc-per-node=4 -m pytest -q \
#       tests/models/test_diffusion_gemma_flashdenoise_tp_distributed.py

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

import vllm._custom_ops as ops


TORCHRUN_CMD = (
    "torchrun --standalone --nproc-per-node=4 -m pytest -q "
    "tests/models/test_diffusion_gemma_flashdenoise_tp_distributed.py"
)
TP_SIZE = 4
OP_NAME = "diffusion_gemma_flashdenoise_local_state_scaled"
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
    rng_seed: int,
    rng_offset: int,
    device: torch.device,
) -> torch.Tensor:
    uniforms = torch.empty(rows, vocab_size, dtype=torch.float32)
    for row in range(rows):
        for vocab_idx in range(vocab_size):
            uniforms[row, vocab_idx] = _deterministic_uniform01(
                rng_seed,
                rng_offset,
                row,
                vocab_idx,
            )
    return -torch.log(-torch.log(uniforms)).to(device)


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        pytest.skip(f"{name}={value!r} is not an integer")


def _require_tp4_torchrun_and_native_op():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.device_count() < TP_SIZE:
        pytest.skip(f"requires at least {TP_SIZE} visible CUDA devices")
    if not dist.is_available():
        pytest.skip("requires torch.distributed")
    if not dist.is_nccl_available():
        pytest.skip("requires NCCL")

    world_size = _env_int("WORLD_SIZE")
    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK")
    if world_size != TP_SIZE or rank is None or local_rank is None:
        pytest.skip(f"run this test with: {TORCHRUN_CMD}")
    if local_rank >= torch.cuda.device_count():
        pytest.skip(
            f"LOCAL_RANK={local_rank} exceeds visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )

    native_op = getattr(ops, OP_NAME, None)
    if native_op is None:
        pytest.skip(f"vllm._custom_ops.{OP_NAME} wrapper is unavailable")
    if not (hasattr(torch.ops, "_C") and hasattr(torch.ops._C, OP_NAME)):
        pytest.skip(f"native torch.ops._C.{OP_NAME} op is unavailable")

    return rank, local_rank, native_op


def _make_dense_inputs(device: torch.device):
    rows = 6
    hidden_size = 16
    vocab_size = 32
    generator = torch.Generator(device="cpu").manual_seed(20260621)
    hidden = torch.randn(
        rows, hidden_size, dtype=torch.float32, generator=generator
    ) * 0.125
    weight = torch.randn(
        vocab_size, hidden_size, dtype=torch.float32, generator=generator
    ) * 0.125
    logit_scale = torch.linspace(0.7, 1.35, rows, dtype=torch.float32)

    # Row 0 has equal clean maxima on two different TP shards. The global clean
    # argmax reducer must choose the smaller global token id, not the lower rank.
    hidden[0].zero_()
    hidden[0, 0] = 1.0
    weight[:, 0] = torch.linspace(-0.4, 0.3, vocab_size)
    weight[:, 1:].zero_()
    weight[9, 0] = 2.0
    weight[25, 0] = 2.0
    logit_scale[0] = 1.0

    return (
        hidden.to(device=device, dtype=torch.bfloat16),
        weight.to(device=device, dtype=torch.bfloat16),
        logit_scale.to(device=device),
    )


def _dense_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    logit_scale: torch.Tensor,
    normalizer: float,
    final_logit_softcapping: float,
    rng_seed: int,
    rng_offset: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    logits = torch.mm(hidden, weight.t(), out_dtype=torch.float32)
    if final_logit_softcapping > 0.0:
        logits = final_logit_softcapping * torch.tanh(
            logits / final_logit_softcapping
        )
    logits = logits * logit_scale.unsqueeze(-1)

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    soft_embed = torch.matmul(probs, weight.float()) * normalizer
    clean_values, clean_indices = logits.max(dim=-1)
    gumbels = _deterministic_gumbels(
        logits.shape[0],
        logits.shape[1],
        rng_seed=rng_seed,
        rng_offset=rng_offset,
        device=logits.device,
    )
    sample_values, sample_indices = (logits + gumbels).max(dim=-1)
    return (
        entropy,
        soft_embed,
        clean_values,
        clean_indices.to(torch.int64),
        sample_values,
        sample_indices.to(torch.int64),
    )


def _merge_local_sampler_state(
    local_max: torch.Tensor,
    local_sum_exp: torch.Tensor,
    local_weighted_logits: torch.Tensor,
    local_soft_part: torch.Tensor,
    normalizer: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)

    scale = torch.exp(local_max - global_max)
    global_sum_exp = local_sum_exp * scale
    global_weighted = local_weighted_logits * scale
    global_soft_part = local_soft_part * scale.unsqueeze(-1)

    dist.all_reduce(global_sum_exp, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_weighted, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_soft_part, op=dist.ReduceOp.SUM)

    entropy = torch.log(global_sum_exp) + global_max - global_weighted / global_sum_exp
    soft_embed = global_soft_part / global_sum_exp.unsqueeze(-1) * normalizer
    return entropy, soft_embed


def _reduce_clean_argmax(
    clean_values: torch.Tensor,
    clean_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_parts = [torch.empty_like(clean_values) for _ in range(TP_SIZE)]
    index_parts = [torch.empty_like(clean_indices) for _ in range(TP_SIZE)]
    dist.all_gather(value_parts, clean_values)
    dist.all_gather(index_parts, clean_indices)

    values = torch.stack(value_parts, dim=-1)
    indices = torch.stack(index_parts, dim=-1)
    winning_values = values.max(dim=-1, keepdim=True).values
    masked_indices = torch.where(
        values == winning_values,
        indices,
        torch.full_like(indices, torch.iinfo(indices.dtype).max),
    )
    winning_rank = masked_indices.argmin(dim=-1, keepdim=True)
    reduced_values = values.gather(dim=-1, index=winning_rank).squeeze(-1)
    reduced_indices = indices.gather(dim=-1, index=winning_rank).squeeze(-1)
    return reduced_values, reduced_indices


def test_tp4_native_local_state_merge_matches_dense_reference():
    rank, local_rank, native_op = _require_tp4_torchrun_and_native_op()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))

    try:
        hidden, full_weight, logit_scale = _make_dense_inputs(device)
        rows, hidden_size = hidden.shape
        vocab_size = full_weight.shape[0]
        assert vocab_size % TP_SIZE == 0
        local_vocab = vocab_size // TP_SIZE
        vocab_start_index = rank * local_vocab
        local_weight = full_weight[
            vocab_start_index : vocab_start_index + local_vocab
        ].contiguous()
        normalizer = 0.75

        for final_logit_softcapping in (0.0, 1.3):
            local_max = torch.empty(rows, device=device, dtype=torch.float32)
            local_sum_exp = torch.empty(rows, device=device, dtype=torch.float32)
            local_weighted_logits = torch.empty(
                rows, device=device, dtype=torch.float32
            )
            local_soft_part = torch.empty(
                rows, hidden_size, device=device, dtype=torch.float32
            )
            clean_values = torch.empty(rows, device=device, dtype=torch.float32)
            clean_indices = torch.empty(rows, device=device, dtype=torch.int64)
            sample_values = torch.empty(rows, device=device, dtype=torch.float32)
            sample_indices = torch.empty(rows, device=device, dtype=torch.int64)

            try:
                native_op(
                    local_max,
                    local_sum_exp,
                    local_weighted_logits,
                    local_soft_part,
                    clean_values,
                    clean_indices,
                    sample_values,
                    sample_indices,
                    hidden,
                    local_weight,
                    logit_scale,
                    vocab_start_index=vocab_start_index,
                    final_logit_softcapping=final_logit_softcapping,
                    rng_seed=1234,
                    rng_offset=99,
                )
            except RuntimeError as exc:
                if "unavailable" in str(exc):
                    pytest.skip(str(exc))
                raise
            torch.cuda.synchronize(device)

            entropy, soft_embed = _merge_local_sampler_state(
                local_max,
                local_sum_exp,
                local_weighted_logits,
                local_soft_part,
                normalizer,
            )
            reduced_clean_values, reduced_clean_indices = _reduce_clean_argmax(
                clean_values, clean_indices
            )
            reduced_sample_values, reduced_sample_indices = _reduce_clean_argmax(
                sample_values, sample_indices
            )
            (
                expected_entropy,
                expected_soft_embed,
                expected_clean_values,
                expected_clean_indices,
                expected_sample_values,
                expected_sample_indices,
            ) = _dense_reference(
                hidden,
                full_weight,
                logit_scale,
                normalizer,
                final_logit_softcapping,
                rng_seed=1234,
                rng_offset=99,
            )

            torch.testing.assert_close(
                entropy, expected_entropy, atol=3e-3, rtol=3e-3
            )
            torch.testing.assert_close(
                soft_embed, expected_soft_embed, atol=5e-3, rtol=3e-2
            )
            torch.testing.assert_close(
                reduced_clean_values, expected_clean_values, atol=3e-3, rtol=0
            )
            torch.testing.assert_close(reduced_clean_indices,
                                       expected_clean_indices)
            torch.testing.assert_close(
                reduced_sample_values,
                expected_sample_values,
                atol=3e-3,
                rtol=0,
            )
            torch.testing.assert_close(reduced_sample_indices,
                                       expected_sample_indices)
            assert reduced_clean_indices[0].item() == 9
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
