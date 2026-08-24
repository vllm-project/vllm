# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for W4AFP8 MoE execution with Humming."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def test_humming_supports_w4afp8_quant_scheme() -> None:
    pytest.importorskip("humming")

    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import (
        HummingExpertsBase,
    )
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        weight_schema_to_quant_key,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8StaticTensorSym,
        kInt4Static128Bf16,
    )
    from vllm.utils.humming import HummingWeightSchema, dtypes

    weight_schema = HummingWeightSchema(
        b_dtype=dtypes.uint4,
        bs_dtype=dtypes.bfloat16,
        weight_scale_group_size=128,
    )

    assert weight_schema_to_quant_key(weight_schema) == kInt4Static128Bf16
    assert HummingExpertsBase._supports_quant_scheme(
        kInt4Static128Bf16,
        kFp8StaticTensorSym,
    )


def test_humming_quantizes_with_static_fp8_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.fused_moe.experts import fused_humming_moe

    inputs = torch.ones((2, 4), dtype=torch.bfloat16)
    quanted_input = torch.empty((2, 4), dtype=torch.float8_e4m3fn)
    input_scale = torch.tensor([0.25], dtype=torch.float32)

    def scaled_fp8_quant(
        inputs_arg: torch.Tensor,
        scale: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert inputs_arg is inputs
        assert scale is input_scale
        assert output is quanted_input
        output.fill_(1)
        return output, scale

    monkeypatch.setattr(
        fused_humming_moe.ops,
        "scaled_fp8_quant",
        scaled_fp8_quant,
    )

    output, output_scale = fused_humming_moe.HummingExpertsBase.quantize_input(
        SimpleNamespace(
            quant_config=SimpleNamespace(a1_scale=input_scale, a2_scale=None)
        ),
        "w13",
        inputs,
        quanted_input,
    )

    assert output is quanted_input
    assert output_scale.shape == (inputs.size(0), 1)
    assert output_scale.is_contiguous()
    torch.testing.assert_close(
        output_scale,
        input_scale.expand(inputs.size(0), 1),
    )


def test_humming_moe_quant_config_preserves_w4afp8_layout() -> None:
    from vllm.model_executor.layers.quantization.utils.humming_utils import (
        make_humming_moe_quant_config,
    )

    a1_scale = torch.tensor([0.25], dtype=torch.float32)
    a2_scale = torch.tensor([0.5], dtype=torch.float32)
    config = make_humming_moe_quant_config(
        quant_dtype="float8_e4m3",
        weight_dtype="int4",
        weight_group_shape=GroupShape(1, 128),
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        humming_configs={"w13": object(), "w2": object()},
    )

    assert config.a1_scale is a1_scale
    assert config.a2_scale is a2_scale
    assert config._a1.shape == GroupShape.PER_TENSOR
    assert config._a2.shape == GroupShape.PER_TENSOR
    assert config._w1.shape == GroupShape(1, 128)
    assert config._w2.shape == GroupShape(1, 128)


@pytest.mark.parametrize(
    ("name", "shape_m", "shape_n", "shape_k", "topk"),
    [
        ("gemm1", 3, 1024, 1024, 8),
        ("gemm2", 24, 1024, 512, 1),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_humming_w4afp8_gemm_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    shape_m: int,
    shape_n: int,
    shape_k: int,
    topk: int,
) -> None:
    pytest.importorskip("humming")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("W4AFP8 Humming kernels require SM90")

    from humming import dtypes
    from humming.config import ComputeConfig, GemmType, LayerConfig
    from humming.testing import KernelTestCase, KernelTestRunner

    if topk == 8:
        topk_ids = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [0, 1, 2, 3, 4, 5, 6, 8],
                [0, 1, 2, 3, 4, 5, 9, 10],
            ],
            dtype=torch.int32,
            device="cuda",
        )
    else:
        topk_ids = (torch.arange(shape_m, device="cuda") % 11).int().unsqueeze(1)
    expert_counts = topk_ids.flatten().long().bincount(minlength=16)
    assert (expert_counts == 0).any()
    assert expert_counts.max() > expert_counts[expert_counts > 0].min()

    def make_topk_ids(
        requested_m: int,
        requested_experts: int,
        requested_topk: int,
    ) -> torch.Tensor:
        assert (requested_m, requested_experts, requested_topk) == (
            shape_m,
            16,
            topk,
        )
        return topk_ids

    monkeypatch.setattr(
        "humming.testing.runner.generate_random_topk_ids",
        make_topk_ids,
    )
    test_case = KernelTestCase(
        name=name,
        layer_config=LayerConfig(
            shape_n=shape_n,
            shape_k=shape_k,
            num_experts=16,
            a_dtype=dtypes.float8e4m3,
            b_dtype=dtypes.uint4,
            c_dtype=dtypes.bfloat16,
            bs_dtype=dtypes.bfloat16,
            weight_scale_group_size=128,
        ),
        compute_config=ComputeConfig(gemm_type=GemmType.INDEXED),
        top_k=topk,
        seed=7,
        atol=0.25,
        rtol=0.05,
    )
    results = KernelTestRunner(test_case).run((shape_m,))

    assert results
    for result in results:
        torch.testing.assert_close(
            result.outputs,
            result.outputs_ref,
            atol=test_case.atol,
            rtol=test_case.rtol,
        )
