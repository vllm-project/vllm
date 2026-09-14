# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.models.deepseek_v2 import (
    _is_deep_gemm_mega_moe_requested,
    _scale_mega_moe_output_for_deferred_reduce,
)
from vllm.models.common.deep_gemm_mega_moe import (
    DeepGemmMegaMoEExperts,
    make_mega_moe_expert_params_mapping,
)
from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration
from vllm.platforms import current_platform


class _TestQuantConfig:
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(
        self,
        name: str,
        *,
        quant_format: str | None = None,
        ignore: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.quant_format = quant_format
        self.ignore = ignore

    def get_name(self) -> str:
        return self.name

    def get_scheme_dict(self, _layer, _prefix):
        return {"format": self.quant_format} if self.quant_format is not None else None


def _make_checkpoint_experts(
    mma_type: str, *, parallel_config=None, **source_kwargs
) -> DeepGemmMegaMoEExperts:
    return DeepGemmMegaMoEExperts(
        SimpleNamespace(
            parallel_config=parallel_config
            or SimpleNamespace(enable_expert_parallel=False),
            scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
            compilation_config=SimpleNamespace(static_forward_context={}),
        ),
        num_experts=1,
        num_local_experts=1,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=512,
        mma_type=mma_type,
        **source_kwargs,
    )


def test_mega_moe_request_applies_to_mtp_model_config():
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe")
    )

    assert _is_deep_gemm_mega_moe_requested(vllm_config)


@pytest.mark.parametrize(
    ("is_sequence_parallel", "expected"), [(False, 1.0), (True, 8.0)]
)
def test_mega_moe_deferred_reduction_scaling(is_sequence_parallel, expected):
    output = torch.full((2, 4), 8.0)

    actual = _scale_mega_moe_output_for_deferred_reduce(
        output,
        tp_size=8,
        is_sequence_parallel=is_sequence_parallel,
        reduce_results=False,
    )

    assert torch.equal(actual, torch.full_like(output, expected))


@pytest.mark.parametrize(
    ("quant_config", "prefix", "is_mxfp4"),
    [
        pytest.param(None, "model.layers.0.mlp", False, id="bf16"),
        pytest.param(
            _TestQuantConfig("deepseek_v4_fp8"),
            "model.layers.0.mlp",
            False,
            id="model-specific",
        ),
        pytest.param(
            _TestQuantConfig("compressed-tensors", quant_format="mxfp4-pack-quantized"),
            "model.layers.0.mlp",
            True,
            id="mxfp4",
        ),
        pytest.param(
            _TestQuantConfig(
                "compressed-tensors",
                quant_format="mxfp4-pack-quantized",
                ignore=(r"re:model.layers.1.*",),
            ),
            "model.layers.1.mtp_block.mlp",
            False,
            id="ignored-mtp",
        ),
    ],
)
def test_mega_moe_checkpoint_format_selection(quant_config, prefix, is_mxfp4):
    layer = torch.nn.Identity()

    assert (
        DeepGemmMegaMoEExperts.source_is_mxfp4(quant_config, layer, prefix) is is_mxfp4
    )


def test_mega_moe_rejects_unsupported_compressed_tensors_checkpoint():
    quant_config = _TestQuantConfig(
        "compressed-tensors", quant_format="nvfp4-pack-quantized"
    )

    with pytest.raises(
        NotImplementedError,
        match="compressed-tensors integration supports MXFP4",
    ):
        DeepGemmMegaMoEExperts.source_is_mxfp4(
            quant_config,
            torch.nn.Identity(),
            "model.layers.0.mlp",
        )


@pytest.mark.parametrize("target", ["Linear", r"re:.*\.experts\.0\..*_proj"])
@pytest.mark.parametrize("suffix", ["", ".experts"])
@pytest.mark.parametrize("ignored", [False, True])
def test_mega_moe_resolves_projection_targets_and_ignores(target, suffix, ignored):
    quant_config = CompressedTensorsConfig(
        target_scheme_map={target: {"format": "mxfp4-pack-quantized"}},
        ignore=[r"re:.*\.experts\.0\..*_proj"] if ignored else [],
        quant_format="pack-quantized",
    )

    assert DeepGemmMegaMoEExperts.source_is_mxfp4(
        quant_config, torch.nn.Identity(), f"model.layers.0.mlp{suffix}"
    ) is (not ignored)


@pytest.mark.parametrize("ignore_gate", [False, True])
def test_mega_moe_rejects_inconsistent_projection_schemes(ignore_gate):
    quant_config = CompressedTensorsConfig(
        target_scheme_map={
            r"re:.*\.gate_proj": {"format": "pack-quantized"},
            "Linear": {"format": "mxfp4-pack-quantized"},
        },
        ignore=[r"re:.*\.gate_proj"] if ignore_gate else [],
        quant_format="pack-quantized",
    )

    with pytest.raises(ValueError, match="All MoE projections"):
        DeepGemmMegaMoEExperts.source_is_mxfp4(
            quant_config, torch.nn.Identity(), "model.layers.0.mlp"
        )


@pytest.mark.parametrize("placement", ["linear", "round_robin"])
@pytest.mark.parametrize("weight_filter", [False, True])
@pytest.mark.parametrize("eplb", [False, True])
def test_mega_moe_rejects_incompatible_ep_weight_filter(placement, weight_filter, eplb):
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True,
        enable_ep_weight_filter=weight_filter,
        enable_eplb=eplb,
        expert_placement_strategy=placement,
    )
    if placement == "round_robin" and weight_filter and not eplb:
        with pytest.raises(ValueError, match="requires contiguous expert placement"):
            _make_checkpoint_experts("bf16xbf16", parallel_config=parallel_config)
    else:
        _make_checkpoint_experts("bf16xbf16", parallel_config=parallel_config)


def test_glm5next_multimodal_post_load_finalizes_language_model():
    calls = []
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            process_weights_after_loading=lambda: calls.append("finalized")
        )
    )

    Glm5NextForConditionalGeneration.process_weights_after_loading(model)

    assert calls == ["finalized"]


@pytest.mark.parametrize(
    ("mma_type", "source_kwargs", "param_name", "dtype"),
    [
        pytest.param("bf16xbf16", {}, "w13_weight", torch.bfloat16, id="bf16"),
        pytest.param(
            "fp8xfp4",
            {"source_mxfp4": True},
            "w13_weight_packed",
            torch.uint8,
            id="mxfp4",
        ),
    ],
)
def test_mega_moe_checkpoint_loading_and_finalization(
    monkeypatch, mma_type, source_kwargs, param_name, dtype
):
    experts = _make_checkpoint_experts(mma_type, **source_kwargs)
    param = getattr(experts, param_name)
    loaded = torch.ones((512, param.shape[-1]), dtype=dtype)

    assert experts.weight_loader(
        param,
        loaded,
        f"experts.{param_name}",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert torch.equal(param[0, :512], loaded)
    if mma_type == "bf16xbf16":
        expected_l1 = torch.empty(1)
        expected_l2 = torch.empty(1)
    else:
        expected_l1 = (torch.empty(1, dtype=torch.int8), torch.empty(1))
        expected_l2 = (torch.empty(1, dtype=torch.int8), torch.empty(1))

    monkeypatch.setattr(experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(
        "vllm.utils.deep_gemm._import_deep_gemm",
        lambda: SimpleNamespace(
            transform_sf_into_required_layout=lambda sf, *args: sf,
            transform_weights_for_mega_moe=lambda *args, **kwargs: (
                expected_l1,
                expected_l2,
            ),
        ),
    )

    experts.finalize_weights()

    assert experts._transformed_l1_weights is expected_l1
    assert experts._transformed_l2_weights is expected_l2


@pytest.mark.skipif(
    not current_platform.is_device_capability_family(100),
    reason="DeepGEMM MegaMoE requires CUDA SM100",
)
def test_mxfp4_checkpoint_finalizes_with_deep_gemm():
    experts = _make_checkpoint_experts("fp8xfp4", source_mxfp4=True).cuda()
    assert experts.w13_weight_packed is not None
    assert experts.w13_weight_scale is not None
    assert experts.w2_weight_packed is not None
    assert experts.w2_weight_scale is not None
    experts.w13_weight_packed.data.fill_(0x11)
    experts.w13_weight_scale.data.fill_(127)
    experts.w2_weight_packed.data.fill_(0x11)
    experts.w2_weight_scale.data.fill_(127)

    experts.finalize_weights()

    assert experts._transformed_l1_weights is not None
    assert experts._transformed_l2_weights is not None


def test_mega_moe_expert_mapping():
    mapping = make_mega_moe_expert_params_mapping(2)

    assert mapping == [
        ("experts.w13_", "experts.0.w1.", 0, "w1"),
        ("experts.w2_", "experts.0.w2.", 0, "w2"),
        ("experts.w13_", "experts.0.w3.", 0, "w3"),
        ("experts.w13_", "experts.1.w1.", 1, "w1"),
        ("experts.w2_", "experts.1.w2.", 1, "w2"),
        ("experts.w13_", "experts.1.w3.", 1, "w3"),
    ]
    assert make_mega_moe_expert_params_mapping(
        1,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    ) == [
        ("experts.w13_", "experts.0.gate_proj.", 0, "w1"),
        ("experts.w2_", "experts.0.down_proj.", 0, "w2"),
        ("experts.w13_", "experts.0.up_proj.", 0, "w3"),
    ]
