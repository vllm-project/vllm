# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.platforms import current_platform

if not torch.cuda.is_available() or not current_platform.is_device_capability_family(
    100
):
    pytest.skip(
        "This test only runs on Blackwell GPUs (SM10x).", allow_module_level=True
    )

pytest.importorskip("cutlass")
pytest.importorskip("cutlass.torch")

from vllm import _custom_ops as ops  # noqa: E402
from vllm.model_executor.specialized_models.kimi_k2_5_nvfp4.model import (
    _run_kimik25_concat_and_cache_mla,
)  # noqa: E402

KERNEL_ATOL = 2e-2


@torch.inference_mode()
def test_kimik25_cutlass_concat_and_cache_mla_matches_cuda_op() -> None:
    torch.manual_seed(0)

    num_tokens = 7
    num_blocks = 4
    block_size = 16
    kv_lora_rank = 512
    pe_dim = 64

    slot_mapping = torch.tensor(
        [0, 3, -1, 17, 31, 44, 55],
        device="cuda",
        dtype=torch.long,
    )
    kv_c = (
        torch.randn(
            num_tokens,
            kv_lora_rank,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.3
    )
    k_pe = (
        torch.randn(
            num_tokens,
            pe_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.3
    )
    scale = torch.tensor(0.02, device="cuda", dtype=torch.float32)

    expected_k_pe = k_pe.clone()
    actual_k_pe = k_pe.clone()

    expected = torch.empty(
        num_blocks,
        block_size,
        kv_lora_rank + pe_dim,
        device="cuda",
        dtype=torch.uint8,
    )
    actual = torch.empty_like(expected)
    expected.fill_(123)
    actual.fill_(123)

    ops.concat_and_cache_mla(
        kv_c,
        expected_k_pe,
        expected,
        slot_mapping,
        kv_cache_dtype="fp8",
        scale=scale,
    )
    _run_kimik25_concat_and_cache_mla(
        kv_c=kv_c,
        k_pe=actual_k_pe,
        kv_cache=actual,
        slot_mapping=slot_mapping,
        kv_cache_dtype="fp8",
        scale=scale,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual_k_pe,
        expected_k_pe,
        atol=max(get_default_atol(actual_k_pe), KERNEL_ATOL),
        rtol=get_default_rtol(actual_k_pe),
    )

    expected_dequant = torch.empty_like(expected, dtype=torch.float16)
    actual_dequant = torch.empty_like(actual, dtype=torch.float16)
    ops.convert_fp8(
        expected_dequant,
        expected.contiguous(),
        scale.item(),
        kv_dtype="fp8",
    )
    ops.convert_fp8(
        actual_dequant,
        actual.contiguous(),
        scale.item(),
        kv_dtype="fp8",
    )
    torch.testing.assert_close(actual_dequant, expected_dequant, atol=0.02, rtol=0.1)
