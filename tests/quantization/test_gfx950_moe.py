# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quark MXFP4 MoE contracts on MI355 (GFX950).

Covers backend selection, deterministic kernel execution, and real-checkpoint
execution at TP1, TP2, and TP8.
"""

import contextlib
import types
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm is required", allow_module_level=True)

from vllm.platforms.rocm import on_gfx950  # noqa: E402

if not on_gfx950():
    pytest.skip("GFX950 is required", allow_module_level=True)

from tests.quantization.reference_mxfp4 import qdq_mxfp4_torch  # noqa: E402
from vllm._aiter_ops import is_aiter_found_and_supported  # noqa: E402
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)  # noqa: E402
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    select_mxfp4_moe_backend,
)  # noqa: E402
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
)  # noqa: E402
from vllm.utils.import_utils import has_triton_kernels  # noqa: E402

REQUIRES_AITER = pytest.mark.skipif(
    not is_aiter_found_and_supported(), reason="Requires supported AITER"
)
REQUIRES_TRITON_KERNELS = pytest.mark.skipif(
    not has_triton_kernels(), reason="Requires triton_kernels"
)

LLAMA4_MXFP4 = "mawong-amd/Llama-4-Scout-17B-16E-Instruct-2-layers-mxfp4"
QWEN_MXFP4 = "fxmarty/qwen_1.5-moe-a2.7b-mxfp4"
DEEPSEEK_MXFP4 = "fxmarty/deepseek_r1_3_layers_mxfp4"


def _refresh_rocm_aiter_env(*, cache_enabled: bool) -> None:
    import vllm.envs as envs_module
    from vllm._aiter_ops import rocm_aiter_ops

    envs_module.disable_envs_cache()
    if cache_enabled:
        envs_module.enable_envs_cache()
    rocm_aiter_ops.refresh_env_variables()


@contextlib.contextmanager
def _temporary_rocm_aiter_state(enabled: bool):
    """Set AITER flags while preserving vLLM's environment-cache state."""
    import vllm.envs as envs_module

    cache_was_enabled = envs_module._is_envs_cache_enabled()
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            value = "1" if enabled else "0"
            monkeypatch.setenv("VLLM_ROCM_USE_AITER", value)
            monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", value)
            _refresh_rocm_aiter_env(cache_enabled=cache_was_enabled)
            yield
    finally:
        _refresh_rocm_aiter_env(cache_enabled=cache_was_enabled)


@pytest.fixture
def enable_rocm_aiter():
    with _temporary_rocm_aiter_state(enabled=True):
        yield


@pytest.fixture
def disable_rocm_aiter():
    with _temporary_rocm_aiter_state(enabled=False):
        yield


def _make_w4a4_moe_config(moe_backend: str = "auto") -> FusedMoEConfig:
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=256,
        intermediate_size=256,
        num_local_experts=8,
        num_logical_experts=8,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
        moe_backend=moe_backend,
    )


@pytest.fixture
def mxfp4_oracle_config():
    """Stub the config the oracle reads (``model_config.quantization_config``)
    so backend dispatch resolves without a real model / user override."""
    from unittest.mock import patch

    with patch(
        "vllm.model_executor.layers.fused_moe.oracle.mxfp4.get_current_vllm_config"
    ) as mock_get_config:
        mock_get_config.return_value.model_config.quantization_config = None
        yield


@REQUIRES_AITER
def test_w4a4_dispatches_to_aiter(mxfp4_oracle_config, enable_rocm_aiter):
    """With AITER enabled + GFX950, W4A4 selects AITER_MXFP4_MXFP4."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    assert experts_cls is not None


def test_w4a4_falls_back_to_emulation_without_aiter(
    mxfp4_oracle_config, disable_rocm_aiter
):
    """Without AITER and no --moe-backend, W4A4 falls back to emulation."""
    config = _make_w4a4_moe_config()
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


def test_w4a4_dispatches_to_emulation_with_moe_backend(mxfp4_oracle_config):
    """With --moe-backend emulation, W4A4 selects EMULATION."""
    config = _make_w4a4_moe_config(moe_backend="emulation")
    backend, experts_cls = select_mxfp4_moe_backend(
        config, activation_key=kMxfp4Dynamic
    )
    assert backend == Mxfp4MoeBackend.EMULATION
    assert experts_cls is not None


def _check_w4a4_aiter_moe(model: torch.nn.Module) -> int:
    from vllm.model_executor.layers.quantization.quark.quark_moe import (
        QuarkOCP_MX_MoEMethod,
    )

    methods = []
    for module in model.modules():
        method: Any = getattr(module, "quant_method", None)
        while hasattr(method, "old_quant_method"):
            method = method.old_quant_method
        if isinstance(method, QuarkOCP_MX_MoEMethod):
            methods.append(method)

    assert methods, "the checkpoint must contain Quark OCP-MX MoE layers"
    for method in methods:
        assert method.ocp_mx_scheme == "w_mxfp4_a_mxfp4"
        assert method.mxfp4_backend is Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    return len(methods)


def _load_real_checkpoint_and_generate(
    vllm_runner,
    model: str,
    tensor_parallel_size: int,
) -> None:
    assert torch.accelerator.device_count() >= tensor_parallel_size
    with vllm_runner(
        model,
        tensor_parallel_size=tensor_parallel_size,
        load_format="auto",
        moe_backend="aiter",
        compilation_config={"cudagraph_capture_sizes": [1]},
        gpu_memory_utilization=0.8,
    ) as llm:
        layer_counts = llm.apply_model(_check_w4a4_aiter_moe)
        assert len(layer_counts) == tensor_parallel_size
        assert all(count > 0 for count in layer_counts)
        output = llm.generate_greedy_logprobs(
            ["Hello"], max_tokens=2, num_logprobs=None
        )

    assert len(output) == 1
    token_ids, _, _ = output[0]
    assert 1 <= len(token_ids) <= 2


@REQUIRES_AITER
@pytest.mark.usefixtures("enable_rocm_aiter", "enable_pickle")
def test_llama4_quark_mxfp4_real_weights_tp1(vllm_runner) -> None:
    _load_real_checkpoint_and_generate(
        vllm_runner, LLAMA4_MXFP4, tensor_parallel_size=1
    )


@REQUIRES_AITER
@pytest.mark.distributed(num_gpus=2)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Requires at least 2 GPUs",
)
@pytest.mark.usefixtures("enable_rocm_aiter", "enable_pickle")
def test_qwen_quark_mxfp4_real_weights_tp2(vllm_runner) -> None:
    _load_real_checkpoint_and_generate(vllm_runner, QWEN_MXFP4, tensor_parallel_size=2)


@REQUIRES_AITER
@pytest.mark.distributed(num_gpus=8)
@pytest.mark.skipif(
    torch.accelerator.device_count() < 8,
    reason="Requires at least 8 GPUs",
)
@pytest.mark.usefixtures("enable_rocm_aiter", "enable_pickle")
def test_deepseek_quark_mxfp4_real_weights_tp8(vllm_runner) -> None:
    _load_real_checkpoint_and_generate(
        vllm_runner, DEEPSEEK_MXFP4, tensor_parallel_size=8
    )


# Deterministic kernel-level contracts.

NUM_EXPERTS = NUM_TOKENS = 8
HIDDEN_SIZE = INTERMEDIATE_SIZE = 8 * 32


@dataclass(frozen=True)
class BackendCase:
    backend: str
    activation: str
    topk: int
    kernel_index: int = 0
    use_bias: bool = False


BACKEND_CASES = [
    pytest.param(
        BackendCase("TRITON", "SWIGLUOAI", 1),
        id="triton-w4a16-monolithic",
    ),
    pytest.param(
        BackendCase("TRITON_UNFUSED", "SWIGLUOAI", 4),
        id="triton-w4a16-modular-topk4",
    ),
    pytest.param(
        BackendCase("AITER_MXFP4_BF16", "SWIGLUOAI", 4, use_bias=True),
        id="aiter-w4a16-topk4-bias",
    ),
    pytest.param(
        BackendCase("AITER_MXFP4_FP8", "SWIGLUOAI", 1),
        id="aiter-w4a8",
    ),
    pytest.param(
        BackendCase("AITER_MXFP4_MXFP4", "SILU", 1),
        id="aiter-w4a4",
    ),
    pytest.param(
        BackendCase("EMULATION", "SILU", 4),
        id="emulation-w4a4-topk4",
    ),
]


def _set_mxfp4(
    packed: torch.Tensor,
    expert: int,
    row: int,
    column: int,
    code: int,
) -> None:
    """Write one E2M1 code; even columns occupy the low nibble."""
    byte = column // 2
    shift = 4 * (column % 2)
    packed[expert, row, byte] |= code << shift


def _scale_pattern(rows: int, width: int) -> torch.Tensor:
    expert = torch.arange(NUM_EXPERTS, device="cuda").reshape(-1, 1, 1)
    row = torch.arange(rows, device="cuda").reshape(1, -1, 1)
    group = torch.arange(width // 32, device="cuda").reshape(1, 1, -1)
    # Fill every scale slot with a valid E8M0 value spanning 2^-3 through
    # 2^3. Scales for analytically selected nonzero weights are set below.
    return (127 + (expert + 3 * row + group) % 7 - 3).to(torch.uint8)


def _checkpoint_weights(
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13 = torch.zeros(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w2 = torch.zeros(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE // 2,
        dtype=torch.uint8,
        device="cuda",
    )
    w13_scale = _scale_pattern(2 * INTERMEDIATE_SIZE, HIDDEN_SIZE)
    w2_scale = _scale_pattern(HIDDEN_SIZE, INTERMEDIATE_SIZE)

    for expert in range(NUM_EXPERTS):
        intermediate = expert * 32 + 5
        output = (NUM_EXPERTS - 1 - expert) * 32 + 7
        if activation == "SWIGLUOAI":
            gate_row, up_row = 2 * intermediate, 2 * intermediate + 1
            gate_code, up_code = 7, 3  # 6 and 1.5
        else:
            gate_row, up_row = intermediate, INTERMEDIATE_SIZE + intermediate
            gate_code, up_code = 4, 2  # 2 and 1

        for input_group in range(HIDDEN_SIZE // 32):
            source = input_group * 32 + 1
            input_exponent = input_group % 5 - 2
            weight_exponent = (
                2 - input_exponent if activation == "SWIGLUOAI" else 3 - input_exponent
            )
            _set_mxfp4(w13, expert, gate_row, source, gate_code)
            _set_mxfp4(w13, expert, up_row, source, up_code)
            w13_scale[expert, gate_row, input_group] = 127 + weight_exponent
            w13_scale[expert, up_row, input_group] = 127 + weight_exponent

        output_exponent = expert % 5 - 2
        _set_mxfp4(w2, expert, output, intermediate, 2)  # 1 * 2^exponent
        w2_scale[expert, output, intermediate // 32] = 127 + output_exponent

    return w13, w2, w13_scale, w2_scale


def _hidden_states() -> torch.Tensor:
    x = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
    for token in range(NUM_TOKENS):
        exponent = token % 5 - 2
        scale = 2.0**exponent
        group_start = token * 32
        x[token, group_start] = 1.625 * scale
        x[token, group_start + 1] = 0.5 * scale

    even = qdq_mxfp4_torch(x, "even")
    sources = torch.arange(NUM_TOKENS, device="cuda") * 32 + 1
    rows = torch.arange(NUM_TOKENS, device="cuda")
    assert torch.equal(even[rows, sources], x[rows, sources])
    return x


def _checkpoint_biases() -> tuple[torch.Tensor, torch.Tensor]:
    w13_bias = torch.zeros(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    w2_bias = torch.zeros(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    for expert in range(NUM_EXPERTS):
        intermediate = expert * 32 + 5
        w13_bias[expert, 2 * intermediate] = -6
        w13_bias[expert, 2 * intermediate + 1] = 1
        w2_bias[expert, expert * 32 + 11] = 2.0 ** (expert % 4 - 1)
    return w13_bias, w2_bias


def _routes(topk: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if topk == 1:
        ids = torch.arange(NUM_TOKENS, device="cuda", dtype=torch.int32).reshape(-1, 1)
        weights = torch.ones(NUM_TOKENS, 1, dtype=torch.float32, device="cuda")
    else:
        offsets = torch.arange(topk, device="cuda", dtype=torch.int32)
        ids = (
            torch.arange(NUM_TOKENS, device="cuda", dtype=torch.int32).reshape(-1, 1)
            + offsets
        ) % NUM_EXPERTS
        weights = torch.tensor(
            [0.5, 0.25, 0.125, 0.125], dtype=torch.float32, device="cuda"
        ).expand(NUM_TOKENS, -1)

    logits = torch.full(
        (NUM_TOKENS, NUM_EXPERTS), -100.0, dtype=torch.float32, device="cuda"
    )
    logits.scatter_(1, ids[:, :1].long(), 100.0)
    return ids.contiguous(), weights.contiguous(), logits


def _expected_output(
    case: BackendCase,
    ids: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # The fixtures make each GEMM a single product. For SILU, gate=8 and
    # up=4, so the exact activation is 32/(1+exp(-8)); its distance from 32
    # is < 0.011. The preceding BF16 value is 31.875, so this is safely above
    # their midpoint, 31.9375, and rounds to 32. For SWIGLUOAI,
    # clamp(gate)=7 and up=3, so the result is 28/(1+exp(-1.702*7)); its
    # distance from 28 is < 0.001, below half a BF16 ULP at 28 (0.0625).
    if case.activation == "SWIGLUOAI":
        # With bias, gate=12-6 and up=3+1. GPT-OSS SWIGLUOAI applies
        # gate * sigmoid(1.702 * gate) * (up + 1), which rounds to 30.
        activation_value = 30.0 if case.use_bias else 28.0
    else:
        activation_value = 32.0
    expected = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
    for token in range(NUM_TOKENS):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            output = (NUM_EXPERTS - 1 - expert) * 32 + 7
            w2_value = 2.0 ** (expert % 5 - 2)
            expected[token, output] += (
                activation_value * w2_value * weights[token, route]
            )
            if case.use_bias:
                bias_output = expert * 32 + 11
                bias = 2.0 ** (expert % 4 - 1)
                expected[token, bias_output] += bias * weights[token, route]
    return expected.to(torch.bfloat16)


def _layer(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
    static_scale = torch.full((NUM_EXPERTS,), 0.25, dtype=torch.float32, device="cuda")
    layer.w13_input_scale = torch.nn.Parameter(static_scale, requires_grad=False)
    layer.w2_input_scale = torch.nn.Parameter(static_scale.clone(), requires_grad=False)
    return layer


@REQUIRES_AITER
@REQUIRES_TRITON_KERNELS
@pytest.mark.usefixtures("enable_rocm_aiter")
@pytest.mark.parametrize("case", BACKEND_CASES)
@torch.inference_mode()
def test_rocm_mxfp4_moe_contract(
    case: BackendCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed.parallel_state as parallel_state
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
        backend_to_kernel_cls,
        convert_gpt_oss_weight_to_mxfp4_moe_kernel_format,
        make_mxfp4_moe_kernel,
        make_mxfp4_moe_quant_config,
    )
    from vllm.v1.worker.workspace import init_workspace_manager

    monkeypatch.setattr(parallel_state, "_TP", types.SimpleNamespace(world_size=1))
    init_workspace_manager(torch.accelerator.current_device_index())

    backend = Mxfp4MoeBackend[case.backend]
    activation = MoEActivation[case.activation]
    experts_cls = backend_to_kernel_cls(backend)[case.kernel_index]
    moe_config = FusedMoEConfig(
        num_experts=NUM_EXPERTS,
        experts_per_token=case.topk,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_local_experts=NUM_EXPERTS,
        num_logical_experts=NUM_EXPERTS,
        moe_parallel_config=FusedMoEParallelConfig.make_no_parallel(),
        activation=activation,
        in_dtype=torch.bfloat16,
        device="cuda",
        routing_method=RoutingMethodType.Renormalize,
    )

    w13, w2, w13_scale, w2_scale = _checkpoint_weights(case.activation)
    w13_bias, w2_bias = _checkpoint_biases() if case.use_bias else (None, None)
    layer = _layer(w13, w2, w13_scale, w2_scale)
    converted = convert_gpt_oss_weight_to_mxfp4_moe_kernel_format(
        mxfp4_backend=backend,
        layer=layer,
        w13_weight=w13,
        w2_weight=w2,
        w13_weight_scale=w13_scale,
        w2_weight_scale=w2_scale,
        w13_bias=w13_bias,
        w2_bias=w2_bias,
    )
    (
        w13_converted,
        w2_converted,
        w13_scale_converted,
        w2_scale_converted,
        w13_bias_converted,
        w2_bias_converted,
    ) = converted
    quant_config = make_mxfp4_moe_quant_config(
        mxfp4_backend=backend,
        w1_scale=w13_scale_converted,
        w2_scale=w2_scale_converted,
        w1_bias=w13_bias_converted,
        w2_bias=w2_bias_converted,
        a1_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    assert quant_config is not None

    ids, weights, logits = _routes(case.topk)
    expected = _expected_output(case, ids, weights)
    x = _hidden_states()

    with set_current_vllm_config(VllmConfig()):
        kernel = make_mxfp4_moe_kernel(
            moe_quant_config=quant_config,
            moe_config=moe_config,
            mxfp4_backend=backend,
            experts_cls=experts_cls,
        )
        if kernel.is_monolithic:
            assert case.topk == 1
            actual = kernel.apply_monolithic(
                hidden_states=x,
                w1=w13_converted,
                w2=w2_converted,
                router_logits=logits,
                activation=activation,
                global_num_experts=NUM_EXPERTS,
                expert_map=None,
                apply_router_weight_on_input=False,
            )
        else:
            actual = kernel.apply(
                hidden_states=x,
                w1=w13_converted,
                w2=w2_converted,
                topk_weights=weights,
                topk_ids=ids,
                activation=activation,
                global_num_experts=NUM_EXPERTS,
                expert_map=None,
                apply_router_weight_on_input=False,
            )

    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    assert torch.equal(
        actual.contiguous().view(torch.uint16),
        expected.contiguous().view(torch.uint16),
    )
