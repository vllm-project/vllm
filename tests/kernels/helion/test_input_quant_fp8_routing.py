# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip("Helion is not installed", allow_module_level=True)

from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_helion,
)

_GROUP_SIZE = 128


def _q_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max absolute difference in the raw fp8 bit representation (1 ULP allowed)."""
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
def test_helion_matches_native(column_major: bool, tma_aligned: bool):
    """The Helion kernel must match the native per_token_group_fp8_quant within
    1 fp8 ULP, for every scale layout the eager router can pass through."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    x = (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 8).contiguous()

    native_q, native_s = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size=_GROUP_SIZE,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=False,
    )
    helion_q, helion_s = per_token_group_quant_fp8_helion(
        x,
        group_size=_GROUP_SIZE,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=False,
    )

    assert helion_q.stride() == native_q.stride()
    assert helion_s.stride() == native_s.stride()
    assert torch.allclose(helion_s, native_s)
    assert _q_ulp_diff(helion_q, native_q) <= 1


def test_helion_runs_inside_cuda_graph_capture():
    """The routed path runs during CUDA-graph capture; cold capture + replay must
    produce the same result as the native kernel."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    torch.manual_seed(0)
    x = (torch.randn(64, 4096, device="cuda", dtype=torch.bfloat16) * 8).contiguous()

    native_q, native_s = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size=_GROUP_SIZE,
        column_major_scales=False,
        tma_aligned_scales=False,
        dtype=torch.float8_e4m3fn,
        use_ue8m0=False,
    )

    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    captured = x.clone()
    with torch.cuda.graph(graph):
        cap_q, cap_s = per_token_group_quant_fp8_helion(
            captured,
            group_size=_GROUP_SIZE,
            column_major_scales=False,
            tma_aligned_scales=False,
            use_ue8m0=False,
        )
    graph.replay()
    torch.accelerator.synchronize()

    assert torch.allclose(cap_s, native_s)
    assert _q_ulp_diff(cap_q, native_q) <= 1
