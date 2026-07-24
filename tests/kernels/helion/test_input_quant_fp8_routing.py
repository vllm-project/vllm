# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_helion_generated,
)

_GROUP_SIZE = 128


def _q_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    return int(
        (a.view(torch.uint8).to(torch.int16) - b.view(torch.uint8).to(torch.int16))
        .abs()
        .max()
        .item()
    )


@pytest.mark.parametrize(
    "column_major,tma_aligned",
    [(False, False), (True, False), (True, True)],
)
@pytest.mark.parametrize("num_tokens", [64, 65, 8193])
def test_generated_matches_native(
    num_tokens: int, column_major: bool, tma_aligned: bool
):
    """Cover exact, intermediate, and oversized token counts."""
    from vllm.kernels.helion_generated.dispatcher import _runtime_platform

    if _runtime_platform() not in {"nvidia_h100", "nvidia_b200"}:
        pytest.skip("Generated kernels are not available on this GPU")

    torch.manual_seed(0)
    x = (
        torch.randn(num_tokens, 2048, device="cuda", dtype=torch.bfloat16) * 8
    ).contiguous()
    native_q, native_s = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size=_GROUP_SIZE,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=False,
    )
    generated_q, generated_s = per_token_group_quant_fp8_helion_generated(
        x,
        group_size=_GROUP_SIZE,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=False,
    )

    assert generated_q.stride() == native_q.stride()
    assert generated_s.stride() == native_s.stride()
    assert torch.allclose(generated_s, native_s)
    assert _q_ulp_diff(generated_q, native_q) <= 1


def test_generated_ue8m0_matches_native():
    from vllm.kernels.helion_generated.dispatcher import _runtime_platform

    if _runtime_platform() not in {"nvidia_h100", "nvidia_b200"}:
        pytest.skip("Generated kernels are not available on this GPU")

    torch.manual_seed(0)
    x = (torch.randn(65, 2048, device="cuda", dtype=torch.bfloat16) * 8).contiguous()
    native_q, native_s = fp8_utils.per_token_group_quant_fp8(
        x, group_size=_GROUP_SIZE, use_ue8m0=True
    )
    generated_q, generated_s = per_token_group_quant_fp8_helion_generated(
        x, group_size=_GROUP_SIZE, use_ue8m0=True
    )

    assert torch.equal(generated_s, native_s)
    assert _q_ulp_diff(generated_q, native_q) <= 1


def test_generated_runs_inside_cuda_graph_capture():
    """A warmed generated launcher can be captured and replayed."""
    from vllm.kernels.helion_generated.dispatcher import (
        _runtime_platform,
        warmup_per_token_group_fp8_quant,
    )

    if _runtime_platform() not in {"nvidia_h100", "nvidia_b200"}:
        pytest.skip("Generated kernels are not available on this GPU")

    torch.manual_seed(0)
    x = (torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16) * 8).contiguous()
    native_q, native_s = fp8_utils.per_token_group_quant_fp8(
        x, group_size=_GROUP_SIZE, use_ue8m0=False
    )
    warmup_per_token_group_fp8_quant([64])

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured = x.clone()
    with torch.cuda.graph(graph):
        cap_q, cap_s = per_token_group_quant_fp8_helion_generated(
            captured, group_size=_GROUP_SIZE, use_ue8m0=False
        )
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.allclose(cap_s, native_s)
    assert _q_ulp_diff(cap_q, native_q) <= 1


def test_forward_cuda_compiles_with_generated_kernels_enabled(monkeypatch):
    """Dynamo keeps the native op so RMSNorm-plus-quant fusion can still match."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import vllm.envs as envs
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

    monkeypatch.setattr(envs, "VLLM_USE_HELION_KERNELS", True)
    with set_current_vllm_config(VllmConfig()):
        quant = QuantFP8(
            static=False, group_shape=GroupShape(1, _GROUP_SIZE), use_ue8m0=False
        )
        x = torch.randn(32, 4096, device="cuda", dtype=torch.bfloat16).contiguous()
        compiled = torch.compile(quant.forward_cuda, fullgraph=True)
        q, _ = compiled(x)

    assert q.shape == x.shape
    assert q.dtype == torch.float8_e4m3fn
