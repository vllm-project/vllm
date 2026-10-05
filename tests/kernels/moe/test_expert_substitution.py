# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    SubstitutedRoutedExperts,
    get_expert_substitution_spec,
    make_expert_substitution,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.models.mixtral import MixtralMoE
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

MOE_PREFIX = "model.layers.1.mlp.experts"


def _model_config(approximate_experts, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(approximate_experts=approximate_experts),
        **kwargs,
    )


def _expert_substitution(targets: dict[str, dict[int, str]]) -> dict:
    """The ``compression_config.transform_config.expert_substitution`` format."""
    return {
        "version": 1,
        "router_semantics": {
            "preserve_logical_expert_ids": True,
            "preserve_router_weights": True,
            "renormalize_after_substitution": False,
        },
        "targets": {
            path: {
                "num_logical_experts": 5,
                "weight_layout": "compact_retained_experts",
                "replacements": {
                    str(expert_id): {
                        "format": "constant-v1",
                        "tensors": {"value": name},
                    }
                    for expert_id, name in replacements.items()
                },
            }
            for path, replacements in targets.items()
        },
    }


def _compressed_model_config(expert_substitution: dict, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            compression_config={
                "transform_config": {"expert_substitution": expert_substitution}
            }
        ),
        **kwargs,
    )


def _substitution(hidden_size: int, dtype=torch.float32):
    # Experts 1 and 3 of 5 are substituted; 0, 2, 4 map to physical 0, 1, 2.
    return ConstantExpertSubstitution(1, 5, (1, 3), hidden_size, dtype)


def test_parse_approximate_experts():
    assert get_expert_substitution_spec(None) is None
    assert get_expert_substitution_spec(_model_config(None)) is None
    spec = get_expert_substitution_spec(_model_config({"1": [3, 1], "2": [], 5: (0,)}))
    assert spec.experts == {1: (1, 3), 5: (0,)}
    assert spec.value_names == {}
    text_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(approximate_experts={"0": [2]}),
        hf_config=SimpleNamespace(),
    )
    assert get_expert_substitution_spec(text_config).experts == {0: (2,)}


@pytest.mark.parametrize(
    "approximate_experts",
    [[1, 2], {"layer": [1]}, {"1": [1, 1]}, {"1": [-1]}, {"1": ["x"]}],
)
def test_parse_approximate_experts_rejects_invalid_entries(approximate_experts):
    with pytest.raises(ValueError, match="approximate_experts"):
        get_expert_substitution_spec(_model_config(approximate_experts))


def test_parse_expert_substitution_with_explicit_value_names():
    spec = get_expert_substitution_spec(
        _compressed_model_config(
            _expert_substitution(
                {
                    MOE_PREFIX: {3: "shared", 1: "layer_1.expert_1"},
                    "model.layers.2.mlp.experts": {0: "shared"},
                }
            )
        )
    )
    assert spec.experts == {1: (1, 3), 2: (0,)}
    assert spec.value_names == {
        "shared": ((1, 3), (2, 0)),
        "layer_1.expert_1": ((1, 1),),
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update(version=2),
        lambda raw: raw["router_semantics"].update(renormalize_after_substitution=True),
        lambda raw: raw["targets"].update({"mlp.experts": raw["targets"][MOE_PREFIX]}),
        lambda raw: raw["targets"][MOE_PREFIX].update(weight_layout="dense"),
        lambda raw: raw["targets"][MOE_PREFIX]["replacements"]["1"].update(
            format="linear-v1"
        ),
        lambda raw: raw["targets"][MOE_PREFIX]["replacements"]["1"].pop("tensors"),
    ],
)
def test_parse_expert_substitution_rejects_unsupported_metadata(mutate):
    raw = _expert_substitution({MOE_PREFIX: {1: "value"}})
    mutate(raw)
    with pytest.raises(ValueError, match="invalid expert_substitution metadata"):
        get_expert_substitution_spec(_compressed_model_config(raw))


def test_parse_rejects_both_formats():
    config = _compressed_model_config(_expert_substitution({MOE_PREFIX: {1: "v"}}))
    config.hf_config.approximate_experts = {"1": [1]}
    with pytest.raises(ValueError, match="both"):
        get_expert_substitution_spec(config)


def test_make_expert_substitution_matches_layer_index():
    config = _model_config({"1": [3, 1]})
    substitution = make_expert_substitution(config, MOE_PREFIX, 5, 16, torch.bfloat16)

    assert substitution is not None
    assert substitution.layer_idx == 1
    assert substitution.substituted_expert_ids == (1, 3)
    assert substitution.num_compute_experts == 3
    assert substitution.values.dtype == torch.bfloat16
    assert [substitution.physical_expert_id(i) for i in range(5)] == [0, -1, 1, -1, 2]
    assert (
        make_expert_substitution(config, "model.layers.2.mlp.experts", 5, 16, None)
        is None
    )
    assert make_expert_substitution(None, MOE_PREFIX, 5, 16, None) is None
    explicit = _compressed_model_config(_expert_substitution({MOE_PREFIX: {1: "v"}}))
    substitution = make_expert_substitution(explicit, MOE_PREFIX, 5, 16, None)
    assert substitution is not None
    assert substitution.substituted_expert_ids == (1,)
    with pytest.raises(ValueError, match="exactly one layer index"):
        make_expert_substitution(config, "mlp.experts", 5, 16, None)
    with pytest.raises(ValueError, match="within"):
        make_expert_substitution(config, MOE_PREFIX, 3, 16, None)
    with pytest.raises(ValueError, match="retain at least one expert"):
        make_expert_substitution(_model_config({"1": [0, 1]}), MOE_PREFIX, 2, 16, None)


def test_constant_substitution_loads_values():
    substitution = _substitution(2)
    substitution.weight_loader(substitution.values, torch.tensor([1.0, 2.0]), 3)
    with pytest.raises(ValueError, match="missing approx_value .* \\[1\\]"):
        substitution.validate_loaded_values("layer")
    substitution.weight_loader(substitution.values, torch.tensor([3.0, 4.0]), 1)
    substitution.validate_loaded_values("layer")
    torch.testing.assert_close(
        substitution.values, torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    )
    with pytest.raises(ValueError, match="not listed"):
        substitution.weight_loader(substitution.values, torch.ones(2), 2)
    with pytest.raises(ValueError, match="has shape"):
        substitution.weight_loader(substitution.values, torch.ones(3), 1)


@pytest.mark.parametrize("num_tokens", [0, 1, 3, 7])
@pytest.mark.parametrize("hidden_size", [2, 4])
def test_constant_substitution_transforms_routes_and_computes_side_output(
    num_tokens, hidden_size
):
    substitution = _substitution(2)
    substitution.values.data.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    rows = torch.arange(num_tokens) % 3
    topk_ids = torch.tensor([[1, 2], [4, 3], [99, -1]])[rows]
    topk_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [0.5, 0.5]])[rows]
    constant_output = substitution.transform_routes(
        torch.zeros(num_tokens, hidden_size), topk_weights, topk_ids
    )

    torch.testing.assert_close(
        topk_weights, torch.tensor([[0.0, 0.75], [0.6, 0.0], [0.0, 0.0]])[rows]
    )
    torch.testing.assert_close(topk_ids, torch.tensor([[0, 1], [2, 0], [0, 0]])[rows])
    expected = torch.zeros(num_tokens, hidden_size)
    expected[:, :2] = torch.tensor([[2.5, 5.0], [12.0, 16.0], [0.0, 0.0]])[rows]
    torch.testing.assert_close(constant_output, expected)


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
    substitution = _substitution(1, dtype).to(device)
    substitution.values.data.copy_(
        torch.tensor([[60000.0], [-60000.0]], dtype=dtype, device=device)
    )
    topk_weights = torch.tensor([[0.5001, 0.4999]], dtype=torch.float32, device=device)
    topk_ids = torch.tensor([[1, 3]], device=device)
    # Nearly cancelling constants must retain the difference between FP32 weights.
    expected = (topk_weights.double() @ substitution.values.double()).to(dtype)

    actual = substitution.transform_routes(
        torch.zeros(1, 1, dtype=dtype, device=device), topk_weights, topk_ids
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
def test_constant_substitution_supports_cuda_graph_replay():
    substitution = _substitution(30, torch.bfloat16).cuda()
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
        output = substitution.transform_routes(
            hidden_states, topk_weights.clone(), topk_ids.clone()
        )

    for scale in (1.0, 0.5):
        topk_weights.mul_(scale)
        expected = torch.zeros_like(hidden_states)
        expected[0, :30] = substitution.values[0].float() * topk_weights[0, 0]
        expected[1, :30] = substitution.values[1].float() * topk_weights[1, 1]
        graph.replay()
        torch.testing.assert_close(output, expected)


def _bare_substituted_routed_experts(**attrs) -> SubstitutedRoutedExperts:
    layer = SubstitutedRoutedExperts.__new__(SubstitutedRoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.expert_substitution = _substitution(2)
    for name, value in attrs.items():
        setattr(layer, name, value)
    return layer


def test_substituted_routed_experts_loads_logical_ids_into_compact_rows():
    layer = _bare_substituted_routed_experts(
        expert_map_manager=SimpleNamespace(map_global_to_local=lambda i: i)
    )
    assert [layer._map_global_expert_id_to_local_expert_id(i) for i in range(5)] == [
        0,
        -1,
        1,
        -1,
        2,
    ]


@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("output_is_reduced", [None, False, True])
@pytest.mark.parametrize("padded_input", [False, True])
def test_substituted_routed_experts_adds_constant_once(
    tp_rank, output_is_reduced, padded_input
):
    def apply(*, layer, x, topk_weights, topk_ids, **kwargs):
        torch.testing.assert_close(topk_weights, torch.tensor([[0.0, 0.75]]))
        torch.testing.assert_close(topk_ids, torch.tensor([[0, 1]]))
        return torch.ones(x.shape[0], 2)

    layer = _bare_substituted_routed_experts(
        moe_config=SimpleNamespace(tp_rank=tp_rank),
        quant_method=SimpleNamespace(
            is_monolithic=False,
            apply=apply,
            moe_kernel=(
                None
                if output_is_reduced is None
                else SimpleNamespace(output_is_reduced=lambda: output_is_reduced)
            ),
        ),
    )
    layer.expert_substitution.values.data.copy_(
        torch.tensor([[10.0, 20.0], [30.0, 40.0]])
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


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
def test_standard_fused_moe_model_discovers_substitution(dist_init):
    vllm_config = VllmConfig()
    vllm_config.model_config = _model_config(
        {"0": [1]}, dtype=torch.bfloat16, is_moe=True
    )
    vllm_config.kernel_config.moe_backend = "triton"

    with set_current_vllm_config(vllm_config):
        moe = MixtralMoE(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.block_sparse_moe",
        )

    routed_experts = moe.experts.routed_experts
    assert isinstance(routed_experts, SubstitutedRoutedExperts)
    assert routed_experts.expert_substitution.substituted_expert_ids == (1,)
    assert routed_experts.w13_weight.shape[0] == 3
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
    vllm_config = VllmConfig()
    vllm_config.model_config = _model_config(
        {"0": [1, 3]}, dtype=torch.bfloat16, is_moe=True
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
            prefix="model.layers.0.mlp.experts",
            renormalize=False,
        ).cuda()
        substitution = layer.routed_experts.expert_substitution
        assert layer.routed_experts.w13_weight.shape[0] == 2
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
        expected = substitution.transform_routes(hidden_states, topk_weights, topk_ids)
        expected += layer._quant_method.apply(
            layer=layer.routed_experts,
            x=hidden_states.clone(),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            shared_experts=None,
            shared_experts_input=None,
        )

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
            prefix="model.layers.0.mlp.experts",
            renormalize=True,
        )
        with torch.no_grad():
            layer.routed_experts.w13_weight.zero_()
            layer.routed_experts.w2_weight.zero_()
            layer.routed_experts.expert_substitution.values.fill_(0.25)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        hidden_states = torch.zeros(2, 256, dtype=torch.bfloat16)
        router_logits = torch.tensor(
            [[-100.0, 100.0, -100.0, -100.0]] * 2,
            dtype=torch.float32,
        )
        get_forward_context().all_moe_layers = None
        actual = layer(hidden_states, router_logits)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 0.25), atol=0, rtol=0
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or current_platform.device_count() < 2,
    reason="requires two CUDA-like devices",
)
def test_constant_substitution_is_added_once_with_tp2():
    vllm_config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=2))
    vllm_config.model_config = _model_config(
        {"0": [1]}, dtype=torch.bfloat16, is_moe=True
    )
    vllm_config.compilation_config.static_forward_context = {}
    vllm_config.kernel_config.moe_backend = "triton"

    parallel_launch_with_config(2, _tp_substitution_worker, vllm_config, None)
