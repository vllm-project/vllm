# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm._custom_ops as ops
from vllm.models.kimi_k3.nvidia.mla import _reserve_query_head_storage
from vllm.models.kimi_k3.nvidia.ops.fused_mla_key_concat_kv_cache import (
    fused_mla_decode_q_concat_kv_cache_insert,
    fused_mla_key_concat_ds_mla_insert,
    fused_mla_key_concat_kv_cache_insert,
    fused_mla_qkv_quant_kv_cache_fp8_insert,
)
from vllm.platforms import current_platform

NUM_TOKENS = 4
NUM_HEADS = 2
KV_LORA_RANK = 512
NOPE_HEAD_DIM = 128
ROPE_HEAD_DIM = 64
QK_HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
V_HEAD_DIM = 128
CACHE_ENTRY = KV_LORA_RANK + ROPE_HEAD_DIM
DS_MLA_CACHE_ENTRY = 656
BLOCK_SIZE = 8
OWNED_TOKEN_INDICES = [1, 3]

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Kimi-K3 fused MLA ops require CUDA"
)


def test_dcp_gathered_query_reserves_cutlass_head_storage() -> None:
    query = torch.randn(3, 24, CACHE_ENTRY, device="cuda", dtype=torch.bfloat16)

    padded = _reserve_query_head_storage(query, 128)

    assert padded.shape == query.shape
    assert padded.stride() == query.stride()
    assert padded.untyped_storage().nbytes() >= 3 * 128 * CACHE_ENTRY * 2
    assert torch.equal(padded, query)


def _inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)

    def make(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)

    return {
        "q": make(NUM_TOKENS, NUM_HEADS, QK_HEAD_DIM),
        "k_nope": make(NUM_TOKENS, NUM_HEADS, NOPE_HEAD_DIM),
        "k_pe": make(NUM_TOKENS, ROPE_HEAD_DIM),
        "kv_c": make(NUM_TOKENS, KV_LORA_RANK),
        "v": make(NUM_TOKENS, NUM_HEADS, V_HEAD_DIM),
        "ql_nope": make(NUM_TOKENS, NUM_HEADS, KV_LORA_RANK),
        "q_pe": make(NUM_TOKENS, NUM_HEADS, ROPE_HEAD_DIM),
    }


def _slot_mapping() -> torch.Tensor:
    return torch.tensor([-1, 0, -1, 5], device="cuda", dtype=torch.int64)


def _owned(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[OWNED_TOKEN_INDICES]


def _full_key(k_nope: torch.Tensor, k_pe: torch.Tensor) -> torch.Tensor:
    shared_pe = k_pe[:, None, :].expand(-1, k_nope.shape[1], -1)
    return torch.cat((k_nope, shared_pe), dim=-1)


def _latent_query(ql_nope: torch.Tensor, q_pe: torch.Tensor) -> torch.Tensor:
    return torch.cat((ql_nope, q_pe), dim=-1)


def _make_cache(cache_format: str) -> torch.Tensor:
    if cache_format == "bf16":
        return torch.full(
            (1, BLOCK_SIZE, CACHE_ENTRY),
            -7,
            device="cuda",
            dtype=torch.bfloat16,
        )
    if cache_format == "fp8":
        return torch.full(
            (1, BLOCK_SIZE, CACHE_ENTRY),
            1.0,
            device="cuda",
            dtype=torch.float32,
        ).to(torch.float8_e4m3fn)
    assert cache_format == "fp8_ds_mla"
    return torch.full(
        (1, BLOCK_SIZE, DS_MLA_CACHE_ENTRY),
        165,
        device="cuda",
        dtype=torch.uint8,
    )


def _reference_cache(
    cache_format: str,
    inputs: dict[str, torch.Tensor],
    initial_cache: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    reference_cache = initial_cache.clone()
    valid_slots = _owned(slots)
    if cache_format == "fp8_ds_mla":
        ops.concat_and_cache_mla(
            _owned(inputs["kv_c"]),
            _owned(inputs["k_pe"]),
            reference_cache,
            valid_slots,
            cache_format,
            torch.ones(1, device="cuda", dtype=torch.float32),
        )
        return reference_cache

    latent = torch.cat(
        (_owned(inputs["kv_c"]), _owned(inputs["k_pe"])),
        dim=-1,
    )
    flat_reference_cache = reference_cache.view(-1, CACHE_ENTRY)
    for row, slot in zip(latent, valid_slots.tolist()):
        flat_reference_cache[slot].copy_(row.to(reference_cache.dtype))
    return reference_cache


def _assert_cache_matches_reference(
    cache: torch.Tensor,
    reference_cache: torch.Tensor,
    initial_cache: torch.Tensor,
) -> None:
    assert torch.equal(cache, reference_cache)
    for slot in (0, 5):
        assert not torch.equal(cache[0, slot], initial_cache[0, slot])
    for slot in (1, 2, 3, 4, 6, 7):
        assert torch.equal(cache[0, slot], initial_cache[0, slot])


@pytest.mark.parametrize("cache_format", ["bf16", "fp8_ds_mla"])
def test_prefill_concat_ignores_negative_slots_for_cache(cache_format: str) -> None:
    inputs = _inputs()
    slots = _slot_mapping()
    initial_cache = _make_cache(cache_format)
    mixed_cache = initial_cache.clone()
    reference_cache = _reference_cache(cache_format, inputs, initial_cache, slots)

    op = (
        fused_mla_key_concat_kv_cache_insert
        if cache_format == "bf16"
        else fused_mla_key_concat_ds_mla_insert
    )
    output = op(inputs["k_nope"], inputs["k_pe"], inputs["kv_c"], mixed_cache, slots)

    torch.testing.assert_close(
        output, _full_key(inputs["k_nope"], inputs["k_pe"]), rtol=0, atol=0
    )
    _assert_cache_matches_reference(mixed_cache, reference_cache, initial_cache)


def test_fp8_prefill_outputs_ignore_negative_slots_for_cache() -> None:
    inputs = _inputs()
    slots = _slot_mapping()
    initial_cache = _make_cache("fp8")
    mixed_cache = initial_cache.clone()
    reference_cache = _reference_cache("fp8", inputs, initial_cache, slots)
    scale_inv = torch.ones(1, device="cuda", dtype=torch.float32)

    q, k, v = fused_mla_qkv_quant_kv_cache_fp8_insert(
        inputs["q"],
        inputs["k_nope"],
        inputs["k_pe"],
        inputs["kv_c"],
        inputs["v"],
        mixed_cache,
        slots,
        scale_inv,
        scale_inv,
        scale_inv,
        scale_inv,
    )

    expected = (
        inputs["q"].to(torch.float8_e4m3fn),
        _full_key(inputs["k_nope"], inputs["k_pe"]).to(torch.float8_e4m3fn),
        inputs["v"].to(torch.float8_e4m3fn),
    )
    for actual, reference in zip((q, k, v), expected):
        torch.testing.assert_close(actual.float(), reference.float(), rtol=0, atol=0)
    _assert_cache_matches_reference(mixed_cache, reference_cache, initial_cache)


@pytest.mark.parametrize("cache_format", ["bf16", "fp8", "fp8_ds_mla"])
def test_decode_concat_ignores_negative_slots_for_cache(cache_format: str) -> None:
    inputs = _inputs()
    slots = _slot_mapping()
    initial_cache = _make_cache(cache_format)
    mixed_cache = initial_cache.clone()
    reference_cache = _reference_cache(cache_format, inputs, initial_cache, slots)
    scale_inv = torch.ones(1, device="cuda", dtype=torch.float32)
    kwargs = {
        "ds_mla": cache_format == "fp8_ds_mla",
        "q_scale_inv": scale_inv if cache_format == "fp8" else None,
        "cache_scale_inv": scale_inv if cache_format == "fp8" else None,
    }

    output = fused_mla_decode_q_concat_kv_cache_insert(
        inputs["ql_nope"],
        inputs["q_pe"],
        inputs["kv_c"],
        inputs["k_pe"],
        mixed_cache,
        slots,
        **kwargs,
    )

    expected = _latent_query(inputs["ql_nope"], inputs["q_pe"])
    if cache_format == "fp8":
        expected = expected.to(torch.float8_e4m3fn)
        torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)
    else:
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    _assert_cache_matches_reference(mixed_cache, reference_cache, initial_cache)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_ds_mla_cache_insert_bit_compatible_with_reference(seed: int) -> None:
    """Fused ds_mla insert must be bit-identical to concat_and_cache_mla.

    Guards the fp8 payload regression where the stable-ABI bf16 scalar fell
    through to an unspecialized fp8 conversion and stored garbage payload
    bytes while per-tile scales and rope bytes stayed correct, making the
    cache corruption silent.
    """
    torch.manual_seed(seed)
    num_tokens, num_blocks = 33, 16
    dt = torch.bfloat16
    kv_c = torch.randn(num_tokens, KV_LORA_RANK, device="cuda", dtype=dt)
    k_pe = torch.randn(num_tokens, 1, ROPE_HEAD_DIM, device="cuda", dtype=dt)
    k_nope = torch.randn(num_tokens, NUM_HEADS, NOPE_HEAD_DIM, device="cuda", dtype=dt)
    slots = torch.randperm(num_blocks * BLOCK_SIZE, device="cuda")[:num_tokens].long()
    slots[torch.rand(num_tokens, device="cuda") < 0.3] = -1
    ref = torch.full(
        (num_blocks, BLOCK_SIZE, DS_MLA_CACHE_ENTRY),
        165,
        device="cuda",
        dtype=torch.uint8,
    )
    got = ref.clone()
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    ops.concat_and_cache_mla(kv_c, k_pe.squeeze(1), ref, slots, "fp8_ds_mla", scale)
    fused_mla_key_concat_ds_mla_insert(k_nope, k_pe, kv_c, got, slots)
    assert torch.equal(ref, got)
