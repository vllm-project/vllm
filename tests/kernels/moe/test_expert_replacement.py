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
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.expert_replacement import (
    ConstantExpertReplacement,
    ConstantExpertReplacementSpec,
    ExpertLayout,
    ExpertSubstitutionTarget,
    make_expert_replacement,
    parse_expert_substitution_config,
    validate_expert_substitution_targets,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

MODULE_PATH = "model.layers.1.mlp.experts"


def _target(
    replacement_ids: tuple[int, ...] = (1, 3),
    num_logical_experts: int = 5,
) -> ExpertSubstitutionTarget:
    return ExpertSubstitutionTarget(
        module_path=MODULE_PATH,
        num_logical_experts=num_logical_experts,
        replacements=tuple(
            ConstantExpertReplacementSpec(
                expert_id,
                f"model.layers.1.mlp.expert_replacements.{expert_id}.value",
            )
            for expert_id in replacement_ids
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


def test_parse_expert_substitution_config():
    parsed = parse_expert_substitution_config(_config())

    assert parsed is not None
    assert parsed.version == 1
    target = parsed.get_target(MODULE_PATH)
    assert target is not None
    assert target.num_logical_experts == 5
    assert target.replacement_expert_ids == (1, 3)
    assert target.replacements[1].value_tensor.endswith("expert_replacements.3.value")


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
        make_expert_replacement(
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


def test_validate_expert_substitution_targets_rejects_unknown_module():
    with pytest.raises(ValueError, match="not supported by this model"):
        validate_expert_substitution_targets(_config(), set())


def test_expert_layout_uses_sorted_logical_ids_for_physical_rows():
    layout = ExpertLayout.from_replacements(6, [4, 1])

    assert layout.compute_expert_ids == (0, 2, 3, 5)
    assert layout.replacement_expert_ids == (1, 4)
    assert layout.logical_to_physical == (0, -1, 1, 2, -1, 3)


def test_constant_replacement_transforms_routes_and_computes_side_output():
    replacement = ConstantExpertReplacement(_target(), 2, torch.float32)
    replacement.values.data.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    topk_ids = torch.tensor([[1, 2], [4, 3], [99, -1]])
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]])
    compute_weights, compute_ids, replacement_output = replacement.transform_routes(
        torch.zeros(3, 2), topk_weights, topk_ids
    )

    assert replacement.compute_expert_ids == (0, 2, 4)
    assert compute_weights.data_ptr() == topk_weights.data_ptr()
    torch.testing.assert_close(
        compute_weights,
        torch.tensor([[0.0, 0.75], [0.6, 0.0], [0.0, 0.0]]),
    )
    torch.testing.assert_close(compute_ids, topk_ids)
    torch.testing.assert_close(
        replacement_output,
        torch.tensor([[2.5, 5.0], [12.0, 16.0], [0.0, 0.0]]),
    )


def test_constant_replacement_loads_explicit_tensor_and_compacts_mapping():
    replacement = ConstantExpertReplacement(
        _target((1, 3), num_logical_experts=4), 3, torch.float32
    )
    replacement.values.weight_loader(
        replacement.values,
        torch.tensor([1.0, 2.0, 3.0]),
        expert_id=3,
    )
    with pytest.raises(ValueError, match="logical expert IDs: \\[1\\]"):
        replacement.validate_loaded_values("test layer")

    mapping = replacement.make_expert_params_mapping(
        moe_prefix="layers.1.mlp.experts",
        ckpt_prefix="layers.1.mlp.experts",
        checkpoint_prefix_to_strip="model.",
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
    )
    assert len(mapping) == 2 * 3 + 2
    assert (
        "layers.1.mlp.experts.routed_experts.expert_replacement.values",
        "layers.1.mlp.expert_replacements.3.value",
        3,
        "constant",
    ) in mapping
    assert (
        "layers.1.mlp.experts.routed_experts.w13_",
        "layers.1.mlp.experts.2.gate_proj.",
        1,
        "w1",
    ) in mapping


def test_model_mapping_delegates_to_routed_experts_layout():
    regular_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(regular_experts)
    regular_experts.expert_replacement = None
    regular_experts.moe_config = SimpleNamespace(num_logical_experts=5)
    replacement = ConstantExpertReplacement(_target(), 3, torch.float32)
    routed_experts = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed_experts)
    routed_experts.expert_replacement = replacement

    model = torch.nn.Module()
    layers = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    model.add_module("layers", layers)
    regular_runner = torch.nn.Module()
    regular_runner.add_module("routed_experts", regular_experts)
    regular_mlp = torch.nn.Module()
    regular_mlp.add_module("experts", regular_runner)
    layers[0].add_module("mlp", regular_mlp)
    experts = torch.nn.Module()
    experts.add_module("routed_experts", routed_experts)
    mlp = torch.nn.Module()
    mlp.add_module("experts", experts)
    layers[1].add_module("mlp", mlp)

    mapping = RoutedExperts.make_expert_params_mapping(
        model,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=5,
    )

    assert mapping[0][0] == "layers.0.mlp.experts.routed_experts.w13_"
    assert mapping[-1][0].endswith("expert_replacement.values")
    assert mapping[-1][1] == "layers.1.mlp.expert_replacements.3.value"


def test_constant_replacement_requires_a_vector_tensor():
    replacement = ConstantExpertReplacement(
        _target((1,), num_logical_experts=4), 3, torch.float32
    )

    with pytest.raises(ValueError, match=r"has shape \(1, 3\), expected \(3,\)"):
        replacement.values.weight_loader(
            replacement.values,
            torch.ones(1, 3),
            expert_id=1,
        )


def test_make_expert_replacement_matches_canonical_module_path():
    replacement = make_expert_replacement(
        config=_config(),
        module_path=MODULE_PATH,
        num_logical_experts=5,
        hidden_size=16,
        params_dtype=torch.bfloat16,
    )

    assert replacement is not None
    assert replacement.values.dtype == torch.bfloat16
    assert replacement.compute_expert_ids == (0, 2, 4)
    assert (
        make_expert_replacement(
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
def test_invalid_replacement_routes_are_not_scheduled_or_reduced():
    topk_ids = torch.tensor([[0, 1], [2, 3]], device="cuda")
    expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int32, device="cuda")
    _, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids,
        block_size=4,
        num_experts=4,
        expert_map=expert_map,
        ignore_invalid_experts=True,
    )
    assert num_tokens_post_pad.item() == 8
    torch.testing.assert_close(
        expert_ids[:2].cpu(), torch.tensor([0, 1], dtype=torch.int32)
    )

    route_outputs = torch.full((2, 2, 16), 1000.0, device="cuda")
    route_outputs[0, 0] = 1.0
    route_outputs[1, 0] = 2.0
    output = torch.empty(2, 16, device="cuda")
    ops.moe_sum(route_outputs, output, topk_ids, expert_map)
    torch.testing.assert_close(
        output, torch.tensor([[1.0] * 16, [2.0] * 16], device="cuda")
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
def test_fused_moe_combines_compact_experts_and_replacement(dist_init):
    hidden_size = 256
    replacement = ConstantExpertReplacement(
        _target((1, 3), num_logical_experts=4), hidden_size, torch.bfloat16
    )
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = "triton"

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        layer = FusedMoEFactory(
            num_experts=2,
            expert_replacement=replacement,
            top_k=2,
            hidden_size=hidden_size,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="test_constant_expert_replacement",
            renormalize=False,
        ).cuda()
        replacement.values.weight_loader(
            replacement.values,
            torch.full((hidden_size,), 0.5, dtype=torch.bfloat16, device="cuda"),
            expert_id=1,
        )
        replacement.values.weight_loader(
            replacement.values,
            torch.full((hidden_size,), 0.75, dtype=torch.bfloat16, device="cuda"),
            expert_id=3,
        )
        replacement.validate_loaded_values("test layer")
        with torch.no_grad():
            layer.routed_experts.w13_weight.normal_(0, 0.01)
            layer.routed_experts.w2_weight.normal_(0, 0.01)
            replacement.values.normal_(0, 0.01)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        hidden_states = torch.randn(8, hidden_size, dtype=torch.bfloat16, device="cuda")
        router_logits = torch.randn(8, 4, dtype=torch.float32, device="cuda")
        topk_weights, topk_ids = layer.router.select_experts(
            hidden_states.clone(), router_logits
        )
        compute_weights, compute_ids, replacement_output = replacement.transform_routes(
            hidden_states, topk_weights, topk_ids
        )
        expected = layer._quant_method.apply(
            layer=layer.routed_experts,
            x=hidden_states.clone(),
            topk_weights=compute_weights,
            topk_ids=compute_ids,
            shared_experts=None,
            shared_experts_input=None,
        )
        expected += replacement_output

        get_forward_context().all_moe_layers = None
        actual = layer(hidden_states.clone(), router_logits)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _tp_replacement_worker(
    process_group: ProcessGroupInfo,
    vllm_config: VllmConfig,
    _cpu_group,
) -> None:
    replacement = ConstantExpertReplacement(
        _target((1,), num_logical_experts=4), 256, torch.bfloat16
    )
    with set_forward_context(None, vllm_config):
        init_workspace_manager(process_group.local_rank)
        layer = FusedMoEFactory(
            num_experts=3,
            expert_replacement=replacement,
            top_k=1,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="test_tp_constant_expert_replacement",
            renormalize=True,
        )
        with torch.no_grad():
            layer.routed_experts.w13_weight.zero_()
            layer.routed_experts.w2_weight.zero_()
            replacement.values.fill_(0.25)
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
def test_constant_replacement_is_added_once_with_tp2():
    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = "triton"

    parallel_launch_with_config(
        2,
        _tp_replacement_worker,
        vllm_config,
        None,
    )
