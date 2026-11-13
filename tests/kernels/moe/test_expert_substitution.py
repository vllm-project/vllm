# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import MixtralConfig

from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    parallel_launch_with_config,
)
from vllm.config import (
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.kernel import MoEBackend
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, RoutedScaleMode
from vllm.model_executor.layers.fused_moe.expert_substitution import (
    ConstantExpertSubstitution,
    get_expert_substitution_spec,
    make_expert_substitution,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_bf16_moe import (
    TrtLlmBf16ExpertsMonolithic,
)
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsSink,
)
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import init_workspace_manager

MOE_PREFIX = "model.layers.1.mlp.experts"


def _moe_sparsity_config(num_experts):
    """The `quantization_config.moe_sparsity_config` novice format."""
    return {"method": "novice", "num_experts": num_experts}


def _model_config(num_experts, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=SimpleNamespace(
            quantization_config={
                "moe_sparsity_config": _moe_sparsity_config(num_experts)
            }
        ),
        **kwargs,
    )


def _substitution(hidden_size: int, dtype=torch.float32):
    # Experts 0..2 are retained; experts 3 and 4 are novices.
    return ConstantExpertSubstitution(1, 5, 3, hidden_size, dtype)


def test_parse_moe_sparsity_config():
    assert get_expert_substitution_spec(None) is None
    assert (
        get_expert_substitution_spec(
            SimpleNamespace(hf_config=SimpleNamespace(quantization_config={}))
        )
        is None
    )
    spec = get_expert_substitution_spec(_model_config([None, 3, None, 5]))
    assert spec is not None
    assert spec.retained_experts == {1: 3, 3: 5}
    text_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(
            quantization_config={"moe_sparsity_config": _moe_sparsity_config([2])}
        ),
        hf_config=SimpleNamespace(),
    )
    text_spec = get_expert_substitution_spec(text_config)
    assert text_spec is not None
    assert text_spec.retained_experts == {0: 2}


@pytest.mark.parametrize(
    "moe_sparsity_config",
    [
        {"method": "prune", "num_experts": [3]},
        {"method": "novice"},
        {"method": "novice", "num_experts": 3},
        {"method": "novice", "num_experts": [0]},
        {"method": "novice", "num_experts": ["x"]},
        {"method": "novice", "num_experts": [1.5]},
        {"method": "novice", "num_experts": [True]},
    ],
)
def test_parse_moe_sparsity_config_rejects_invalid_entries(moe_sparsity_config):
    with pytest.raises(ValueError, match="invalid moe_sparsity_config"):
        get_expert_substitution_spec(
            SimpleNamespace(
                hf_config=SimpleNamespace(
                    quantization_config={"moe_sparsity_config": moe_sparsity_config}
                )
            )
        )


def test_make_expert_substitution_matches_layer_index():
    # Layer 1 retains 3 of 5 experts; the trailing 2 become novices.
    config = _model_config([None, 3])
    substitution = make_expert_substitution(config, MOE_PREFIX, 5, 16, torch.bfloat16)

    assert substitution is not None
    assert substitution.layer_idx == 1
    assert substitution.num_logical_experts == 5
    assert substitution.num_compute_experts == 3
    assert substitution.novices.dtype == torch.bfloat16
    assert (
        make_expert_substitution(config, "model.layers.0.mlp.experts", 5, 16, None)
        is None
    )
    assert make_expert_substitution(None, MOE_PREFIX, 5, 16, None) is None
    # A layer that retains every expert has no novices.
    assert (
        make_expert_substitution(_model_config([None, 5]), MOE_PREFIX, 5, 16, None)
        is None
    )
    with pytest.raises(ValueError, match="exactly one layer index"):
        make_expert_substitution(config, "mlp.experts", 5, 16, None)
    with pytest.raises(ValueError, match="retains 6 experts"):
        make_expert_substitution(_model_config([None, 6]), MOE_PREFIX, 5, 16, None)


def test_constant_substitution_loads_novices():
    substitution = _substitution(2)
    novices = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="missing or incomplete"):
        substitution.validate_loaded_values("layer")
    substitution.weight_loader(substitution.novices, novices)
    substitution.validate_loaded_values("layer")
    torch.testing.assert_close(substitution.novices, novices)
    with pytest.raises(ValueError, match="has shape"):
        substitution.weight_loader(substitution.novices, torch.ones(3, 2))


@pytest.mark.parametrize("num_tokens,hidden_size", [(0, 2), (6, 2), (6, 4)])
def test_constant_substitution_computes_novices_and_transforms_routes(
    num_tokens, hidden_size
):
    substitution = _substitution(2)
    substitution.weight_loader(
        substitution.novices, torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    )
    rows = torch.arange(num_tokens) % 6
    topk_ids = torch.tensor([[3, 1], [2, 4], [99, -1], [3, 4], [1, 2], [3, 3]])[rows]
    topk_weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4], [0.5, 0.5], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
    )[rows]
    actual = substitution.compute_output(
        torch.zeros(num_tokens, hidden_size), topk_weights, topk_ids
    )
    substitution.transform_routes(topk_weights, topk_ids)
    expected = torch.zeros(num_tokens, hidden_size)
    expected[:, :2] = torch.tensor(
        [[2.5, 5.0], [12.0, 16.0], [0.0, 0.0], [25.0, 35.0], [0.0, 0.0], [10.0, 20.0]]
    )[rows]
    torch.testing.assert_close(actual, expected)

    torch.testing.assert_close(
        topk_weights,
        torch.tensor(
            [[0.0, 0.75], [0.6, 0.0], [0.0, 0.0], [0.0, 0.0], [0.25, 0.75], [0.0, 0.0]]
        )[rows],
    )
    torch.testing.assert_close(
        topk_ids, torch.tensor([[0, 1], [2, 0], [0, 0], [0, 0], [1, 2], [0, 0]])[rows]
    )


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
def test_constant_substitution_uses_novice_dtype(dtype, device):
    substitution = _substitution(1, dtype).to(device)
    substitution.weight_loader(
        substitution.novices,
        torch.tensor([[60000.0], [-60000.0]], dtype=dtype, device=device),
    )
    topk_weights = torch.tensor(
        [[0.5001, 0.4999], [0.7501, 0.2499]], dtype=torch.float32, device=device
    )
    expected = (topk_weights.to(dtype).double() @ substitution.novices.double()).to(
        dtype
    )
    actual = substitution.compute_output(
        torch.zeros(2, 1, dtype=dtype, device=device),
        topk_weights,
        torch.tensor([[3, 4], [3, 4]], device=device),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _semantic_config(
    model_dir: Path, backend: MoEBackend, tp_size: int = 1
) -> VllmConfig:
    config = MixtralConfig(
        architectures=["MixtralForCausalLM"],
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=32,
    )
    config.quantization_config = {
        "quant_method": "compressed-tensors",
        "config_groups": {},
        "format": "fakequant",
        "quantization_status": "compressed",
        "moe_sparsity_config": _moe_sparsity_config([2]),
    }
    config.save_pretrained(model_dir)
    return VllmConfig(
        model_config=ModelConfig(
            model=str(model_dir),
            skip_tokenizer_init=True,
            dtype="bfloat16",
            max_model_len=32,
            enforce_eager=True,
        ),
        parallel_config=ParallelConfig(tensor_parallel_size=tp_size),
        kernel_config=KernelConfig(moe_backend=backend),
    )


def _semantic_example():
    """Exact BF16 outputs, with useful retained work in both TP partitions."""
    hidden, intermediate = 256, 512
    gate_up = torch.zeros(
        2, 2 * intermediate, hidden, dtype=torch.bfloat16, device="cpu"
    )
    down = torch.zeros(2, hidden, intermediate, dtype=torch.bfloat16, device="cpu")
    patterns = torch.tensor(
        [[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.bfloat16, device="cpu"
    ).repeat(1, hidden // 4)
    for expert in range(2):
        for neuron in (0, intermediate // 2):
            gate_up[expert, neuron, 0] = 16
            gate_up[expert, intermediate + neuron, 0] = expert + 1
            down[expert, :, neuron] = (3 / 32) * patterns[expert]
    novices = -torch.tensor([[9], [15]], device="cpu") * patterns
    inputs = torch.ones(8, hidden, dtype=torch.bfloat16, device="cpu")
    inputs[:, 0] = torch.tensor([1, 0, 0.5, 2, 1, 0.5, 2, 2], device="cpu")
    # Retained-only, novice-only, and mixed pairs in both score orders.
    pairs = torch.tensor(
        [[0, 1], [2, 3], [0, 2], [1, 3], [2, 0], [3, 1], [1, 0], [3, 2]],
        device="cpu",
    )
    probabilities = torch.full((8, 4), 0.125, device="cpu")
    probabilities.scatter_(
        1, pairs, torch.tensor([[0.5, 0.25]], device="cpu").expand(8, 2)
    )
    return inputs, probabilities.log(), gate_up, down, novices


def _reference_mone(inputs, logits, gate_up, down, novices, renormalize, scale):
    """Evaluate logical experts directly, without vLLM routing or MoE helpers."""
    weights, ids = logits.double().softmax(dim=-1).topk(2, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    expected = torch.zeros_like(inputs, dtype=torch.float64)
    for token in range(inputs.shape[0]):
        for slot in range(2):
            expert = int(ids[token, slot])
            if expert < gate_up.shape[0]:
                gate, up = torch.nn.functional.linear(
                    inputs[token].double(), gate_up[expert].double()
                ).chunk(2)
                value = torch.nn.functional.linear(
                    torch.nn.functional.silu(gate) * up, down[expert].double()
                )
            else:
                value = novices[expert - gate_up.shape[0]].double()
            expected[token] += weights[token, slot] * value
    return (expected * scale).to(inputs.dtype)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="requires a CUDA-like platform"
)
@pytest.mark.parametrize(
    "moe_backend,renormalize,scale,scale_mode",
    [
        ("triton", False, 1.0, RoutedScaleMode.ROUTER),
        ("triton", True, 1.0, RoutedScaleMode.ROUTER),
        ("triton", False, 2.0, RoutedScaleMode.ROUTER),
        ("triton", True, 2.0, RoutedScaleMode.OUTPUT),
        ("auto", True, 2.0, RoutedScaleMode.ROUTER),
        pytest.param(
            "flashinfer_cutlass",
            True,
            2.0,
            RoutedScaleMode.OUTPUT,
            id="flashinfer-cutlass",
            marks=pytest.mark.skipif(
                not FlashInferExperts._supports_current_device(),
                reason="FlashInfer CUTLASS is unavailable on this platform",
            ),
        ),
    ],
)
def test_fused_moe_combines_compact_experts_and_substitution(
    dist_init, tmp_path, moe_backend, renormalize, scale, scale_mode
):
    _check_fused_moe_semantics(tmp_path, moe_backend, renormalize, scale, scale_mode)


def _check_fused_moe_semantics(
    tmp_path, moe_backend, renormalize, scale, scale_mode, capture_graph=False
):
    vllm_config = _semantic_config(tmp_path, moe_backend)
    inputs, logits, gate_up, down, novices = _semantic_example()
    expected = _reference_mone(
        inputs, logits, gate_up, down, novices, renormalize, scale
    )

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        init_workspace_manager(torch.accelerator.current_device_index())
        layer = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=inputs.shape[1],
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            renormalize=renormalize,
            routed_scaling_factor=scale,
            routed_scale_mode=scale_mode,
            use_grouped_topk=scale_mode == RoutedScaleMode.ROUTER and scale != 1.0,
            num_expert_group=1,
            topk_group=1,
        ).cuda()
        substitution = layer.routed_experts.expert_substitution
        if not renormalize or scale_mode == RoutedScaleMode.ROUTER and scale != 1.0:
            assert layer.moe_config.require_decomposed_backend
            assert not layer._quant_method.is_monolithic
        assert layer.routed_experts.w13_weight.shape[0] == 2
        with torch.no_grad():
            layer.routed_experts.w13_weight.copy_(gate_up)
            layer.routed_experts.w2_weight.copy_(down)
            substitution.weight_loader(substitution.novices, novices)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        captured_ids = None
        if layer._quant_method.is_monolithic:
            captured_buffer = torch.empty(8, 2, device="cuda", dtype=torch.int16)
            captured_ids = captured_buffer
            layer.routed_experts.routing_sink = RoutedExpertsSink(
                layer.moe_config, lambda ids: captured_buffer.copy_(ids)
            )

        get_forward_context().all_moe_layers = None
        actual = layer(inputs.cuda(), logits.cuda())
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
        if capture_graph:
            graph = torch.cuda.CUDAGraph()
            hidden, scores = inputs.cuda(), logits.cuda()
            with torch.cuda.graph(graph):
                graph_output = layer(hidden, scores)
            # Change the selected experts after capture to check replay uses
            # current routing IDs, including all-novice and retained-only rows.
            scores.copy_(logits.flip(0).cuda())
            graph.replay()
            replay_expected = _reference_mone(
                inputs, logits.flip(0), gate_up, down, novices, renormalize, scale
            )
            torch.testing.assert_close(
                graph_output.cpu(), replay_expected, atol=0, rtol=0
            )
            expected_ids = logits.flip(0).topk(2, dim=-1).indices
            assert captured_ids is not None
            torch.testing.assert_close(
                captured_ids.cpu().long().sort(-1).values,
                expected_ids.sort(-1).values,
            )


@pytest.mark.skipif(
    not TrtLlmBf16ExpertsMonolithic._supports_current_device(),
    reason="requires FlashInfer TRTLLM BF16",
)
def test_monolithic_mone_matches_logical_expert_reference(dist_init, tmp_path):
    _check_fused_moe_semantics(
        tmp_path,
        "flashinfer_trtllm",
        True,
        2.0,
        RoutedScaleMode.OUTPUT,
        capture_graph=True,
    )


def _tp_substitution_worker(
    process_group: ProcessGroupInfo,
    vllm_config: VllmConfig,
    _cpu_group,
) -> None:
    inputs, logits, gate_up, down, novices = _semantic_example()
    expected = _reference_mone(inputs, logits, gate_up, down, novices, True, 2.0)
    with set_forward_context(None, vllm_config):
        init_workspace_manager(process_group.local_rank)
        layer = FusedMoEFactory(
            num_experts=4,
            top_k=2,
            hidden_size=256,
            intermediate_size=512,
            params_dtype=torch.bfloat16,
            prefix="model.layers.0.mlp.experts",
            renormalize=True,
            routed_scaling_factor=2.0,
            routed_scale_mode=RoutedScaleMode.OUTPUT,
        )
        # Shard the canonical weights explicitly; each rank owns one active neuron.
        start = process_group.rank * 256
        stop = start + 256
        gate, up = gate_up.chunk(2, dim=1)
        with torch.no_grad():
            layer.routed_experts.w13_weight.copy_(
                torch.cat([gate[:, start:stop], up[:, start:stop]], dim=1)
            )
            layer.routed_experts.w2_weight.copy_(down[:, :, start:stop])
            substitution = layer.routed_experts.expert_substitution
            substitution.weight_loader(substitution.novices, novices)
        layer._quant_method.process_weights_after_loading(layer.routed_experts)

        get_forward_context().all_moe_layers = None
        actual = layer(inputs.to(process_group.device), logits.to(process_group.device))
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or current_platform.device_count() < 2,
    reason="requires two CUDA-like devices",
)
@pytest.mark.parametrize(
    "backend",
    [
        "triton",
        pytest.param(
            "flashinfer_trtllm",
            marks=pytest.mark.skipif(
                not TrtLlmBf16ExpertsMonolithic._supports_current_device(),
                reason="requires FlashInfer TRTLLM BF16",
            ),
        ),
    ],
)
def test_constant_substitution_is_added_once_with_tp2(tmp_path, backend):
    vllm_config = _semantic_config(tmp_path, backend, tp_size=2)
    parallel_launch_with_config(2, _tp_substitution_worker, vllm_config, None)
