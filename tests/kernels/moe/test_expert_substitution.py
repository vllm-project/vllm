# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
    ConstantExpertSubstitutionSpec,
    ExpertLayout,
    ExpertSubstitutionTarget,
    intercept_expert_substitution_weights,
    make_expert_substitution,
    parse_expert_substitution_config,
    validate_expert_substitution_model,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.substituted_routed_experts import (
    SubstitutedRoutedExperts,
)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

MODULE_PATH = "model.layers.1.mlp.experts"


def _target(
    substituted_ids: tuple[int, ...] = (1, 3),
    num_logical_experts: int = 5,
) -> ExpertSubstitutionTarget:
    return ExpertSubstitutionTarget(
        module_path=MODULE_PATH,
        num_logical_experts=num_logical_experts,
        replacements=tuple(
            ConstantExpertSubstitutionSpec(
                expert_id,
                f"model.layers.1.mlp.expert_replacements.{expert_id}.value",
            )
            for expert_id in substituted_ids
        ),
    )


def _raw_config() -> dict:
    return {
        "producer": {"name": "llm-compressor", "version": "test"},
        "provenance": {"algorithm": "mone"},
        "transform_config": {
            "expert_substitution": {
                "version": 1,
                "router_semantics": {
                    "preserve_logical_expert_ids": True,
                    "preserve_router_weights": True,
                    "renormalize_after_substitution": False,
                },
                "targets": {
                    MODULE_PATH: {
                        "num_logical_experts": 5,
                        "weight_layout": "compact_retained_experts",
                        "replacements": {
                            "1": {
                                "format": "constant-v1",
                                "tensors": {
                                    "value": "model.layers.1.mlp."
                                    "expert_replacements.1.value"
                                },
                            },
                            "3": {
                                "format": "constant-v1",
                                "tensors": {
                                    "value": "model.layers.1.mlp."
                                    "expert_replacements.3.value"
                                },
                            },
                        },
                    }
                },
            }
        },
    }


def _config(raw_config: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        compression_config=_raw_config() if raw_config is None else raw_config,
        dtype=torch.float32,
    )


def _config_for_target(
    module_path: str,
    num_logical_experts: int,
    substituted_ids: tuple[int, ...],
) -> SimpleNamespace:
    raw_config = deepcopy(_raw_config())
    targets = raw_config["transform_config"]["expert_substitution"]["targets"]
    target = targets.pop(MODULE_PATH)
    target["num_logical_experts"] = num_logical_experts
    target["replacements"] = {
        str(expert_id): {
            "format": "constant-v1",
            "tensors": {
                "value": f"{module_path}.replacement.{expert_id}.value",
            },
        }
        for expert_id in substituted_ids
    }
    targets[module_path] = target
    return _config(raw_config)


def _model_with_routed_experts(
    *routed_experts: RoutedExperts | None,
    wrapper: tuple[str, ...] = (),
) -> torch.nn.Module:
    model = torch.nn.Module()
    container = model
    for name in wrapper:
        child = torch.nn.Module()
        container.add_module(name, child)
        container = child

    layers = torch.nn.ModuleList(torch.nn.Module() for _ in routed_experts)
    container.add_module("layers", layers)
    for layer, routed in zip(layers, routed_experts):
        if routed is None:
            continue
        experts = torch.nn.Module()
        experts.add_module("routed_experts", routed)
        mlp = torch.nn.Module()
        mlp.add_module("experts", experts)
        layer.add_module("mlp", mlp)
    return model


def test_parse_expert_substitution_config():
    parsed = parse_expert_substitution_config(_config())

    assert parsed is not None
    assert parsed.version == 1
    target = parsed.get_target(MODULE_PATH)
    assert target is not None
    assert target.num_logical_experts == 5
    assert target.substituted_expert_ids == (1, 3)
    assert target.replacements[1].value_tensor.endswith("expert_replacements.3.value")


def test_target_lookup_accepts_an_unambiguous_module_suffix():
    parsed = parse_expert_substitution_config(_config())

    assert parsed is not None
    target = parsed.get_target("layers.1.mlp.experts")
    assert target is not None
    assert target.module_path == MODULE_PATH


def test_target_lookup_rejects_ambiguous_module_suffix():
    raw_config = deepcopy(_raw_config())
    targets = raw_config["transform_config"]["expert_substitution"]["targets"]
    targets[f"language_{MODULE_PATH}"] = deepcopy(targets[MODULE_PATH])
    parsed = parse_expert_substitution_config(_config(raw_config))

    assert parsed is not None
    with pytest.raises(ValueError, match="ambiguous"):
        parsed.get_target("layers.1.mlp.experts")


def test_no_substitution_metadata_is_a_noop():
    config = SimpleNamespace(
        compression_config={
            "producer": {"name": "llm-compressor"},
            "provenance": {"algorithm": "mone"},
            "transform_config": {},
        }
    )

    assert parse_expert_substitution_config(config) is None
    assert (
        make_expert_substitution(
            config=config,
            module_path=MODULE_PATH,
            num_logical_experts=5,
            hidden_size=16,
        )
        is None
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda config: config["transform_config"]["expert_substitution"].update(
                version=2
            ),
            "version",
        ),
        (
            lambda config: config["transform_config"]["expert_substitution"].update(
                version=True
            ),
            "version",
        ),
        (
            lambda config: config["transform_config"]["expert_substitution"][
                "router_semantics"
            ].update(renormalize_after_substitution=True),
            "router_semantics",
        ),
        (
            lambda config: config["transform_config"]["expert_substitution"]["targets"][
                MODULE_PATH
            ].update(weight_layout="packed"),
            "weight_layout",
        ),
        (
            lambda config: config["transform_config"]["expert_substitution"]["targets"][
                MODULE_PATH
            ]["replacements"]["1"].update(format="low-rank-v1"),
            "format",
        ),
        (
            lambda config: config["transform_config"]["expert_substitution"]["targets"][
                MODULE_PATH
            ]["replacements"]["1"].update(tensors={}),
            "tensors.value",
        ),
    ],
)
def test_parse_expert_substitution_config_rejects_unsupported_contracts(mutate, match):
    raw_config = deepcopy(_raw_config())
    mutate(raw_config)

    with pytest.raises(ValueError, match=match):
        parse_expert_substitution_config(_config(raw_config))


def test_validate_expert_substitution_model_requires_every_target():
    with pytest.raises(ValueError, match="unmatched targets"):
        validate_expert_substitution_model(_config(), torch.nn.Module())


def test_intercept_expert_substitution_weights_loads_explicit_values():
    model = torch.nn.Module()
    substitution = ConstantExpertSubstitution(_target((1,)), 2, torch.float32)
    model.add_module("substitution", substitution)
    value_name = _target((1,)).replacements[0].value_tensor
    remaining, loaded = intercept_expert_substitution_weights(
        model,
        [
            (value_name, torch.tensor([1.0, 2.0])),
            ("other.weight", torch.tensor([3.0])),
        ],
    )

    remaining_weights = list(remaining)
    assert [name for name, _ in remaining_weights] == ["other.weight"]
    torch.testing.assert_close(remaining_weights[0][1], torch.tensor([3.0]))
    assert loaded == {"substitution.values"}
    torch.testing.assert_close(substitution.values, torch.tensor([[1.0, 2.0]]))


def test_expert_layout_uses_sorted_logical_ids_for_physical_rows():
    layout = ExpertLayout.from_substitutions(6, [4, 1])

    assert layout.compute_expert_ids == (0, 2, 3, 5)
    assert layout.substituted_expert_ids == (1, 4)
    assert layout.logical_to_physical == (0, -1, 1, 2, -1, 3)


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 7])
@pytest.mark.parametrize("hidden_size", [2, 4])
def test_constant_substitution_transforms_routes_and_computes_side_output(
    num_tokens, hidden_size
):
    substitution = ConstantExpertSubstitution(_target(), 2, torch.float32)
    substitution.values.data.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    rows = torch.arange(num_tokens) % 3
    topk_ids = torch.tensor([[1, 2], [4, 3], [99, -1]])[rows]
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]])[rows]
    compute_weights, compute_ids, substitution_output = substitution.transform_routes(
        torch.zeros(num_tokens, hidden_size),
        topk_weights,
        topk_ids,
    )

    assert substitution.compute_expert_ids == (0, 2, 4)
    assert compute_weights.data_ptr() == topk_weights.data_ptr()
    torch.testing.assert_close(
        compute_weights,
        torch.tensor([[0.0, 0.75], [0.6, 0.0], [0.0, 0.0]])[rows],
    )
    torch.testing.assert_close(
        compute_ids,
        torch.tensor([[0, 1], [2, 0], [0, 0]])[rows],
    )
    expected = torch.zeros(num_tokens, hidden_size)
    expected[:, :2] = torch.tensor([[2.5, 5.0], [12.0, 16.0], [0.0, 0.0]])[rows]
    torch.testing.assert_close(substitution_output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda_alike(),
                reason="requires a CUDA-like platform",
            ),
        ),
    ],
)
def test_constant_substitution_preserves_router_weight_precision(dtype, device):
    substitution = ConstantExpertSubstitution(_target(), 1, dtype).to(device)
    substitution.values.data.copy_(
        torch.tensor([[60000.0], [-60000.0]], dtype=dtype, device=device)
    )
    topk_weights = torch.tensor([[0.5001, 0.4999]], dtype=torch.float32, device=device)
    topk_ids = torch.tensor([[1, 3]], device=device)
    # Nearly cancelling constants must retain the difference between FP32 weights.
    expected = (topk_weights.double() @ substitution.values.double()).to(dtype)

    _, _, actual = substitution.transform_routes(
        torch.zeros(1, 1, dtype=dtype, device=device), topk_weights, topk_ids
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
def test_constant_substitution_supports_cuda_graph_replay():
    substitution = ConstantExpertSubstitution(_target(), 30, torch.bfloat16).cuda()
    substitution.values.data.copy_(torch.arange(60, device="cuda").reshape(2, 30) / 8)
    hidden_states = torch.zeros(3, 32, dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[1, 2], [4, 3], [99, -1]], device="cuda")
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]], device="cuda")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            substitution.transform_routes(
                hidden_states, topk_weights.clone(), topk_ids.clone()
            )
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, output = substitution.transform_routes(
            hidden_states, topk_weights.clone(), topk_ids.clone()
        )

    for scale in (1.0, 0.5):
        topk_weights.mul_(scale)
        expected = torch.zeros_like(hidden_states)
        expected[0, :30] = substitution.values[0].float() * topk_weights[0, 0]
        expected[1, :30] = substitution.values[1].float() * topk_weights[1, 1]
        graph.replay()
        torch.testing.assert_close(output, expected)


def test_constant_substitution_loads_explicit_tensor_and_compacts_mapping():
    substitution = ConstantExpertSubstitution(
        _target((1, 3), num_logical_experts=4), 3, torch.float32
    )
    substitution.load_value(torch.tensor([1.0, 2.0, 3.0]), expert_id=3)
    with pytest.raises(ValueError, match="logical expert IDs: \\[1\\]"):
        substitution.validate_loaded_values("test layer")

    mapping = substitution.make_expert_params_mapping(
        moe_prefix="layers.1.mlp.experts",
        ckpt_prefix="layers.1.mlp.experts",
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )
    assert len(mapping) == 2 * 3
    assert (
        "layers.1.mlp.experts.routed_experts.w13_",
        "layers.1.mlp.experts.2.gate_proj.",
        1,
        "w1",
    ) in mapping


def test_model_mapping_delegates_to_routed_experts_layout():
    regular_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(regular_experts)
    regular_experts.moe_config = SimpleNamespace(num_logical_experts=5)
    substitution = ConstantExpertSubstitution(_target(), 3, torch.float32)
    routed_experts = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.expert_substitution = substitution

    model = _model_with_routed_experts(regular_experts, routed_experts)

    mapping = RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=5,
    )

    assert mapping[0][0] == "layers.0.mlp.experts.routed_experts.w13_"
    assert mapping[-1][0] == "layers.1.mlp.experts.routed_experts.w13_"
    assert mapping[-1][1] == "layers.1.mlp.experts.4.up_proj."


def test_model_mapping_accepts_runtime_wrapper_prefix():
    routed_experts = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.expert_substitution = ConstantExpertSubstitution(
        _target(), 3, torch.float32
    )

    model = _model_with_routed_experts(
        None,
        routed_experts,
        wrapper=("language_model", "model"),
    )

    mapping = RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=5,
    )

    assert mapping[0] == (
        "language_model.model.layers.1.mlp.experts.routed_experts.w13_",
        "model.layers.1.mlp.experts.0.gate_proj.",
        0,
        "w1",
    )
    assert mapping[-1][1] == "model.layers.1.mlp.experts.4.up_proj."


def test_model_mapping_rejects_invalid_runtime_module_path():
    routed_experts = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.expert_substitution = ConstantExpertSubstitution(
        _target(), 3, torch.float32
    )
    model = torch.nn.Module()
    model.add_module("routed_experts", routed_experts)

    with pytest.raises(
        ValueError, match="runtime module path 'routed_experts' must end"
    ):
        RoutedExperts.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=5,
        )


def test_model_mapping_rejects_mismatched_target_path():
    routed_experts = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.expert_substitution = ConstantExpertSubstitution(
        _target(), 3, torch.float32
    )
    model = _model_with_routed_experts(routed_experts)

    with pytest.raises(ValueError, match="does not match runtime MoE module"):
        RoutedExperts.make_expert_params_mapping(
            model,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=5,
        )


def test_constant_substitution_requires_a_vector_tensor():
    substitution = ConstantExpertSubstitution(
        _target((1,), num_logical_experts=4), 3, torch.float32
    )

    with pytest.raises(ValueError, match=r"has shape \(1, 3\), expected \(3,\)"):
        substitution.load_value(torch.ones(1, 3), expert_id=1)


def test_make_expert_substitution_matches_canonical_module_path():
    substitution = make_expert_substitution(
        config=_config(),
        module_path=MODULE_PATH,
        num_logical_experts=5,
        hidden_size=16,
        params_dtype=torch.bfloat16,
    )

    assert substitution is not None
    assert substitution.values.dtype == torch.bfloat16
    assert substitution.compute_expert_ids == (0, 2, 4)
    assert (
        make_expert_substitution(
            config=_config(),
            module_path="model.layers.2.mlp.experts",
            num_logical_experts=5,
            hidden_size=16,
        )
        is None
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
def test_standard_fused_moe_model_discovers_substitution_without_adapter(dist_init):
    prefix = "model.layers.0.block_sparse_moe"
    target_path = f"{prefix}.experts"
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=_config_for_target(target_path, 4, (1,)),
        dtype=torch.bfloat16,
        is_moe=True,
    )
    vllm_config.kernel_config.moe_backend = "triton"

    with set_current_vllm_config(vllm_config):
        moe = MixtralMoE(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix=prefix,
        )

    substitution = moe.experts.routed_experts.expert_substitution
    assert substitution is not None
    assert substitution.compute_expert_ids == (0, 2, 3)
    assert moe.experts.routed_experts.w13_weight.shape[0] == 3
    assert moe.experts.moe_config.require_decomposed_backend


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        pytest.param("triton", id="triton"),
        pytest.param(
            "flashinfer_cutlass",
            id="flashinfer-cutlass",
            marks=pytest.mark.skipif(
                not FlashInferExperts._supports_current_device(),
                reason="FlashInfer CUTLASS is unavailable on this platform",
            ),
        ),
    ],
)
def test_fused_moe_combines_compact_experts_and_substitution(dist_init, moe_backend):
    hidden_size = 256
    prefix = "test_constant_expert_substitution"
    vllm_config = VllmConfig()
    vllm_config.model_config = SimpleNamespace(
        hf_config=_config_for_target(prefix, 4, (1, 3)),
        dtype=torch.bfloat16,
        is_moe=True,
    )
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = moe_backend

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        layer = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=hidden_size,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix=prefix,
            renormalize=False,
        ).cuda()
        substitution = layer.routed_experts.expert_substitution
        assert substitution is not None
        assert layer.routed_experts.w13_weight.shape[0] == 2
        substitution.load_value(
            torch.full((hidden_size,), 0.5, dtype=torch.bfloat16, device="cuda"),
            expert_id=1,
        )
        substitution.load_value(
            torch.full((hidden_size,), 0.75, dtype=torch.bfloat16, device="cuda"),
            expert_id=3,
        )
        substitution.validate_loaded_values("test layer")
        with torch.no_grad():
            layer.routed_experts.w13_weight.normal_(0, 0.01)
            layer.routed_experts.w2_weight.normal_(0, 0.01)
            substitution.values.normal_(0, 0.01)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        hidden_states = torch.randn(8, hidden_size, dtype=torch.bfloat16, device="cuda")
        router_logits = torch.randn(8, 4, dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = layer.router.select_experts(
            hidden_states.clone(), router_logits
        )
        compute_weights, compute_ids, substitution_output = (
            substitution.transform_routes(
                hidden_states,
                topk_weights,
                topk_ids,
            )
        )
        expected = layer._quant_method.apply(
            layer=layer.routed_experts,
            x=hidden_states.clone(),
            topk_weights=compute_weights,
            topk_ids=compute_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        expected += substitution_output

        get_forward_context().all_moe_layers = None
        actual = layer(hidden_states.clone(), router_logits)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _tp_substitution_worker(
    process_group: ProcessGroupInfo,
    vllm_config: VllmConfig,
    _cpu_group,
) -> None:
    with set_forward_context(None, vllm_config):
        init_workspace_manager(process_group.local_rank)
        layer = FusedMoEFactory(
            num_experts=4,
            top_k=1,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="test_tp_constant_expert_substitution",
            renormalize=True,
        )
        substitution = layer.routed_experts.expert_substitution
        assert substitution is not None
        with torch.no_grad():
            layer.routed_experts.w13_weight.zero_()
            layer.routed_experts.w2_weight.zero_()
            substitution.values.fill_(0.25)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        hidden_states = torch.zeros(2, 256, dtype=torch.bfloat16)
        router_logits = torch.tensor(
            [[-100.0, 100.0, -100.0, -100.0]] * 2,
            dtype=torch.float32,
        )
        get_forward_context().all_moe_layers = None
        actual = layer(hidden_states, router_logits)
        torch.testing.assert_close(
            actual,
            torch.full_like(actual, 0.25),
            atol=0,
            rtol=0,
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or current_platform.device_count() < 2,
    reason="requires two CUDA-like devices",
)
def test_constant_substitution_is_added_once_with_tp2():
    prefix = "test_tp_constant_expert_substitution"
    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
    vllm_config.model_config = SimpleNamespace(
        hf_config=_config_for_target(prefix, 4, (1,)),
        dtype=torch.bfloat16,
        is_moe=True,
    )
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = "triton"

    parallel_launch_with_config(
        2,
        _tp_substitution_worker,
        vllm_config,
        None,
    )


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("output_is_reduced", [None, False, True])
@pytest.mark.parametrize("padded_input", [False, True])
def test_substituted_routed_experts_adapts_routes_and_adds_constant_once(
    tp_rank, output_is_reduced, padded_input
):
    """The specialized layer owns adaptation, including TP and output padding."""
    layer = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(tp_rank=tp_rank)
    layer.expert_substitution = ConstantExpertSubstitution(_target(), 2, torch.float32)
    layer.expert_substitution.values.data.copy_(
        torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    )

    def apply(*, layer, x, topk_weights, topk_ids, **kwargs):
        torch.testing.assert_close(topk_weights, torch.tensor([[0.0, 0.75]]))
        torch.testing.assert_close(topk_ids, torch.tensor([[0, 1]]))
        return torch.ones(x.shape[0], 2)

    layer.quant_method = SimpleNamespace(
        is_monolithic=False,
        apply=apply,
        moe_kernel=(
            None
            if output_is_reduced is None
            else SimpleNamespace(output_is_reduced=lambda: output_is_reduced)
        ),
    )
    actual = layer.forward_modular(
        torch.zeros(1, 4 if padded_input else 2),
        torch.tensor([[0.25, 0.75]]),
        torch.tensor([[1, 2]]),
    )
    expected = torch.ones(1, 2)
    if tp_rank == 0 or output_is_reduced:
        expected += torch.tensor([[2.5, 5.0]])
    torch.testing.assert_close(actual, expected)


def test_substitution_constants_are_not_reported_as_expert_mlp_weights():
    layer = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.local_num_experts = 3
    layer.w13_weight = torch.nn.Parameter(torch.zeros(3, 4, 2))
    layer.w2_weight = torch.nn.Parameter(torch.zeros(3, 2, 2))
    layer.expert_substitution = ConstantExpertSubstitution(_target(), 2, torch.float32)

    weights = list(layer.get_expert_weights())
    assert len(weights) == 2
    assert [weight.data_ptr() for weight in weights] == [
        layer.w13_weight.data_ptr(),
        layer.w2_weight.data_ptr(),
    ]
