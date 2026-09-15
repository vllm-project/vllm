# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import get_rope

if not torch.cuda.is_available():
    pytest.skip("This test only runs on CUDA GPUs.", allow_module_level=True)

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.torch")

from vllm.models.kimi_k2_5.nvidia.ops.decode_rope_concat_quant_fp8_and_cache_mla import (  # noqa: E402, E501
    decode_rope_concat_quant_fp8_and_cache_mla,
)

KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
NUM_LOCAL_HEADS = 64
MAX_POSITION = 262144
ROPE_PARAMETERS = {
    "rope_type": "deepseek_yarn",
    "rope_theta": 50000.0,
    "factor": 64.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "original_max_position_embeddings": 4096,
}


def _make_mla_like_q_pe(num_tokens: int) -> torch.Tensor:
    query_base = torch.randn(
        num_tokens,
        NUM_LOCAL_HEADS,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return query_base[..., QK_NOPE_HEAD_DIM:]


@pytest.mark.parametrize("num_tokens", [1, 7])
@torch.inference_mode()
def test_kimik25_decode_rope_concat_quant_fp8_and_cache_mla_matches_reference(
    default_vllm_config,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    num_blocks = 4
    block_size = 16

    rope = get_rope(
        head_size=QK_ROPE_HEAD_DIM,
        max_position=MAX_POSITION,
        is_neox_style=False,
        rope_parameters=ROPE_PARAMETERS,
        dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16)

    positions = torch.randint(
        0, MAX_POSITION, (num_tokens,), device="cuda", dtype=torch.long
    )
    ql_nope = torch.randn(
        NUM_LOCAL_HEADS, num_tokens, KV_LORA_RANK, device="cuda", dtype=torch.bfloat16
    ).transpose(0, 1)
    q_pe = _make_mla_like_q_pe(num_tokens)
    kv_c = (
        torch.randn(num_tokens, KV_LORA_RANK, device="cuda", dtype=torch.bfloat16) * 0.3
    )
    k_pe = (
        torch.randn(num_tokens, QK_ROPE_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        * 0.3
    )
    slot_mapping = torch.arange(num_tokens, device="cuda", dtype=torch.long)
    if num_tokens > 1:
        slot_mapping[1] = -1  # exercise the padding-slot skip path
    q_scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    kv_scale = torch.tensor(0.03, device="cuda", dtype=torch.float32)

    cache_dim = KV_LORA_RANK + QK_ROPE_HEAD_DIM

    # ---- reference query: rope(q_pe) -> concat -> static fp8 quant ----
    expected_q_pe, _ = rope.forward_native(
        positions,
        q_pe.clone(),
        torch.zeros(
            num_tokens, 1, QK_ROPE_HEAD_DIM, device="cuda", dtype=torch.bfloat16
        ),
    )
    expected_full = torch.cat((ql_nope, expected_q_pe), dim=-1).reshape(num_tokens, -1)
    expected_q, _ = ops.scaled_fp8_quant(expected_full, q_scale)
    expected_q = expected_q.view(num_tokens, NUM_LOCAL_HEADS, cache_dim)

    # ---- reference KV cache: generic concat_and_cache_mla ----
    expected_cache = torch.empty(
        num_blocks, block_size, cache_dim, device="cuda", dtype=torch.uint8
    )
    expected_cache.fill_(123)
    ops.concat_and_cache_mla(
        kv_c,
        k_pe.clone(),
        expected_cache,
        slot_mapping,
        kv_cache_dtype="fp8",
        scale=kv_scale,
    )

    actual_cache = torch.empty_like(expected_cache)
    actual_cache.fill_(123)
    actual_q = decode_rope_concat_quant_fp8_and_cache_mla(
        positions=positions,
        ql_nope=ql_nope,
        q_pe=q_pe,
        cos_sin_cache=rope.cos_sin_cache,
        q_scale=q_scale,
        kv_c=kv_c,
        k_pe=k_pe,
        kv_cache=actual_cache,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp8",
        kv_scale=kv_scale,
    )
    torch.cuda.synchronize()

    # The non-positional ("nope") query columns are not rotated, so they quantize
    # bit-identically to the generic static fp8 quant.
    actual_bytes = actual_q.view(torch.uint8)
    expected_bytes = expected_q.view(torch.uint8)
    torch.testing.assert_close(
        actual_bytes[..., :KV_LORA_RANK],
        expected_bytes[..., :KV_LORA_RANK],
        atol=0,
        rtol=0,
    )

    # The rotary columns may differ by a single fp8 step: the kernel rotates in
    # bf16 while the reference rotates in fp32. Compare the dequantized query.
    torch.testing.assert_close(
        actual_q.float() * q_scale,
        expected_q.float() * q_scale,
        atol=2e-2,
        rtol=0.15,
    )

    # Compare the dequantized KV cache against the generic op.
    expected_deq = torch.empty_like(expected_cache, dtype=torch.float16)
    actual_deq = torch.empty_like(actual_cache, dtype=torch.float16)
    ops.convert_fp8(
        expected_deq, expected_cache.contiguous(), kv_scale.item(), kv_dtype="fp8"
    )
    ops.convert_fp8(
        actual_deq, actual_cache.contiguous(), kv_scale.item(), kv_dtype="fp8"
    )
    torch.testing.assert_close(actual_deq, expected_deq, atol=0.02, rtol=0.1)
