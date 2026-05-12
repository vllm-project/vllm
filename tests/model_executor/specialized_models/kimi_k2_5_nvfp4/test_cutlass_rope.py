# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )

cutlass = pytest.importorskip("cutlass")
cutlass_torch = pytest.importorskip("cutlass.torch")
from_dlpack = pytest.importorskip("cutlass.cute.runtime").from_dlpack

from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (  # noqa: E402
    _run_kimik25_decode_rope_concat_quant_fp8,
    _run_kimik25_rope,
    kimik25_rope,
)

QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
NUM_LOCAL_HEADS = 64
MAX_POSITION = 262144
KERNEL_ATOL = 2e-2
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


def _make_mla_like_query_view(
    num_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    query_base = torch.randn(
        num_tokens,
        NUM_LOCAL_HEADS,
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=dtype,
    )
    query = query_base[..., QK_NOPE_HEAD_DIM:]
    return query


@pytest.mark.parametrize("num_tokens", [13, 64])
@torch.inference_mode()
def test_kimik25_cutlass_rope_matches_pytorch_reference(
    default_vllm_config,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    rope = get_rope(
        head_size=QK_ROPE_HEAD_DIM,
        max_position=MAX_POSITION,
        is_neox_style=False,
        rope_parameters=ROPE_PARAMETERS,
        dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16)

    positions = torch.randint(
        0,
        MAX_POSITION,
        (num_tokens,),
        device="cuda",
        dtype=torch.long,
    )

    ref_query = _make_mla_like_query_view(num_tokens, torch.bfloat16)
    ref_key = torch.empty(
        num_tokens,
        1,
        QK_ROPE_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected_query, _ = rope.forward_native(positions, ref_query, ref_key)

    actual_query = _make_mla_like_query_view(num_tokens, torch.bfloat16)
    actual_query.copy_(ref_query)

    assert actual_query.stride() == (
        NUM_LOCAL_HEADS * (QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM),
        QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM,
        1,
    )

    kimik25_rope(
        from_dlpack(positions),
        from_dlpack(actual_query).mark_layout_dynamic(),
        from_dlpack(rope.cos_sin_cache),
        NUM_LOCAL_HEADS,
        QK_ROPE_HEAD_DIM // 2,
        cutlass_torch.current_stream(),
    )
    torch.cuda.synchronize()

    # The Cutlass kernel uses Blackwell bf16 arithmetic directly, which differs
    # from vLLM's PyTorch-native reference by about two bf16 ULPs in practice.
    torch.testing.assert_close(
        actual_query,
        expected_query,
        atol=max(get_default_atol(actual_query), KERNEL_ATOL),
        rtol=get_default_rtol(actual_query),
    )


@pytest.mark.parametrize("num_tokens", [1, 7])
@torch.inference_mode()
def test_kimik25_cutlass_decode_rope_concat_quant_fp8_matches_reference(
    default_vllm_config,
    num_tokens: int,
) -> None:
    torch.manual_seed(0)

    rope = get_rope(
        head_size=QK_ROPE_HEAD_DIM,
        max_position=MAX_POSITION,
        is_neox_style=False,
        rope_parameters=ROPE_PARAMETERS,
        dtype=torch.bfloat16,
    ).to(device="cuda", dtype=torch.bfloat16)

    positions = torch.randint(
        0,
        MAX_POSITION,
        (num_tokens,),
        device="cuda",
        dtype=torch.long,
    )
    ql_nope = torch.randn(
        NUM_LOCAL_HEADS,
        num_tokens,
        512,
        device="cuda",
        dtype=torch.bfloat16,
    ).transpose(0, 1)
    q_pe = _make_mla_like_query_view(num_tokens, torch.bfloat16)
    expected_q_pe = _make_mla_like_query_view(num_tokens, torch.bfloat16)
    expected_q_pe.copy_(q_pe)
    _run_kimik25_rope(
        positions,
        expected_q_pe,
        rope.cos_sin_cache,
        NUM_LOCAL_HEADS,
        QK_ROPE_HEAD_DIM // 2,
    )
    scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)
    expected_q, _ = ops.scaled_fp8_quant(
        torch.cat((ql_nope, expected_q_pe), dim=-1).reshape(num_tokens, -1),
        scale,
    )
    expected_q = expected_q.view(num_tokens, NUM_LOCAL_HEADS, 512 + QK_ROPE_HEAD_DIM)

    actual_q = _run_kimik25_decode_rope_concat_quant_fp8(
        positions=positions,
        ql_nope=ql_nope,
        q_pe=q_pe,
        cos_sin_cache=rope.cos_sin_cache,
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_q.float(),
        expected_q.float(),
        atol=0,
        rtol=0,
    )
