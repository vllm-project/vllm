# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the MOE layer.

Run `pytest tests/kernels/test_moe_layer.py`.
"""

import functools
import os
import traceback
import types
from collections.abc import Callable
from dataclasses import astuple, dataclass, fields
from itertools import product
from typing import get_args

import pytest
import torch

import vllm.model_executor.layers.quantization.utils.w8a8_utils
from tests.kernels.moe.modular_kernel_tools.parallel_utils import (
    ProcessGroupInfo,
    _set_vllm_config,
    parallel_launch_with_config,
)
from tests.kernels.moe.utils import TestMLP, make_test_weights, moe_quantize_weights
from vllm.config import (
    CompilationConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
from vllm.distributed.parallel_state import (
    get_ep_group,
    get_eplb_group,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoE, fused_experts
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.router.router_factory import (
    create_fused_moe_router,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptNvFp4Config,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    has_flashinfer_nvlink_one_sided,
    has_flashinfer_nvlink_two_sided,
)
from vllm.utils.import_utils import has_deep_ep, has_mori, has_nixl_ep
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    is_workspace_manager_initialized,
)

fp8_dtype = torch.float8_e4m3fn  # current_platform.fp8_dtype

SHAPE_COMBOS = [
    (1, 128, 256),
    (32, 512, 512),
    (222, 1024, 2048),
]
MAX_M = max([x[0] for x in SHAPE_COMBOS])

NUM_EXPERTS = [8, 64]
TOP_KS = [2, 6]

# dp_size, tp_size, use_ep
# Note: DP+TP is not yet supported in the FusedMoE layer.
PARALLEL_COMBOS = [
    [1, 2, False],
    [1, 4, False],
    [2, 1, True],
    [4, 1, True],
]

# TODO: should this even be set manually?  let oracles handle this
BACKENDS = ["allgather_reducescatter"]

if has_mori():
    BACKENDS += ["mori"]

if has_flashinfer_nvlink_two_sided():
    BACKENDS += ["flashinfer_nvlink_two_sided"]

if has_flashinfer_nvlink_one_sided():
    BACKENDS += ["flashinfer_nvlink_one_sided"]

if has_deep_ep():
    BACKENDS += ["deepep_high_throughput", "deepep_low_latency"]

if has_nixl_ep():
    BACKENDS += ["nixl_ep"]

QUANT_METHODS = [
    None,
    "fp8",
    "fp8_blocked",
    "modelopt_fp8",
    "modelopt_fp4",
]

# Which quantization methods each backend supports.
# fmt: off
BACKEND_SUPPORTED_QUANTS: dict[str, set[str | None]] = {
    "allgather_reducescatter":     {None, "fp8", "modelopt_fp8", "modelopt_fp4"},
    "mori":                        {None, "fp8", "modelopt_fp8"},
    "flashinfer_nvlink_two_sided": {None,        "modelopt_fp8", "modelopt_fp4"},
    "flashinfer_nvlink_one_sided": {None,        "modelopt_fp8", "modelopt_fp4"},
    "deepep_low_latency":          {None, "fp8_blocked", "modelopt_fp4"},
    "deepep_high_throughput":      {None, "fp8_blocked", "modelopt_fp8", "modelopt_fp4"}, # noqa: E501
    "nixl_ep":                     {None, "fp8", "modelopt_fp8"},
}

# Map from backend -> (DP/EP support, DP support, TP support)
BACKEND_EP_DP_TP_SUPPORT: dict[str, tuple[bool, bool, bool]] = {
    "allgather_reducescatter":     (True,  True,  True),
    "mori":                        (True, False, False),
    "flashinfer_nvlink_two_sided": (False, True, False),
    "flashinfer_nvlink_one_sided": (False, True, False),
    "deepep_low_latency":          (True, False, False),
    "deepep_high_throughput":      (True, False, False),
    "nixl_ep":                     (True, False, False),
}
# fmt: on

# Which quantization methods support EPLB.
# ModelOptFp8MoEMethod inherits supports_eplb=False from FusedMoEMethodBase.
# TODO: double check modelopt fp8
# modelopt_fp4 excluded: get_expert_weights() can't handle NvFP4 packed format.
EPLB_SUPPORTED_QUANTS: list[str | None] = [None, "fp8"]

# Which backends support EPLB.
# deepep backends fail in get_expert_weights / rearrange_expert_weights_inplace.
# TODO(bnell): check this
EPLB_SUPPORTED_BACKENDS: list[str] = ["allgather_reducescatter"]


def mock_normalize_e4m3fn_to_e4m3fnuz(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor | None = None,
):
    return weight, weight_scale, input_scale


# Needed since weights will already be in e4m3fnuz format on platforms that
# use the fnuz fp8 format and the normalize_e4m3fn_to_e4m3fnuz() function
# is not being tested here.
# NOTE: The weights are quantized by moe_quantize_weights_2d in
# _quantize_fp8_halves.
# NOTE: Not able to use monkeypatch because of the spawned parallel workers.
def override_normalize_e4m3fn_to_e4m3fnuz():
    vllm.model_executor.layers.quantization.utils.w8a8_utils.normalize_e4m3fn_to_e4m3fnuz = mock_normalize_e4m3fn_to_e4m3fnuz  # noqa: E501


def maybe_roundup_layer_hidden_size(
    hidden_size: int,
    act_dtype: torch.dtype,
    backend: str | None,
) -> int:
    """
    Given layer hidden size and MoE configurations, round up hidden_size
    if necessary.

    Args:
        hidden_size: Layer hidden-size
        act_dtype: Data type of the layer activations.
        moe_parallel_config: Fused MoE parallelization strategy configuration.

    Return:
        Rounded up hidden_size if rounding up is required based on the configs
        and all2all backend.
        Original hidden size otherwise.
    """
    if backend == "deepep_high_throughput":
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import (
            DeepEPHTPrepareAndFinalize,
        )

        hidden_size = DeepEPHTPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size, act_dtype
        )

    if backend == "deepep_low_latency":
        from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
            DeepEPLLPrepareAndFinalize,
        )

        hidden_size = DeepEPLLPrepareAndFinalize.maybe_roundup_layer_hidden_size(
            hidden_size
        )

    return hidden_size


def rank_chunk(num: int, r: int, w: int) -> int:
    rem = num % w
    return (num // w) + (1 if r < rem else 0)


def chunk_by_rank(
    t: torch.Tensor,
    r: int,
    w: int,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    chunk = cdiv(t.shape[dim], w)
    t = t.narrow(dim, r * chunk, chunk)
    if device is not None:
        t = t.to(device)
    return t


def maybe_chunk_by_rank(
    t: torch.Tensor | None,
    r: int,
    w: int,
    dim: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None:
        return chunk_by_rank(t, r, w, dim, device)
    else:
        return t


def tp_chunk_gate_up(
    w: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    dim: int,
    device: torch.device | int | None = None,
) -> torch.Tensor:
    """TP-chunk a combined [gate; up] weight, splitting each half separately
    so every rank gets a portion of both gate and up."""
    half = w.shape[dim] // 2
    gate = chunk_by_rank(
        w.narrow(dim, 0, half), tp_rank, tp_size, dim=dim, device=device
    )
    up = chunk_by_rank(
        w.narrow(dim, half, half), tp_rank, tp_size, dim=dim, device=device
    )
    return torch.cat([gate, up], dim=dim)


@dataclass
class MoETestConfig:
    m: int
    n: int
    k: int
    num_experts: int
    top_k: int
    in_dtype: torch.dtype
    quantization: str | None
    use_shared_experts: bool
    use_gate: bool
    use_routed_input_transform: bool
    enable_eplb: bool = False
    backend: str | None = None
    ep_size: int = 1
    dp_size: int = 1
    tp_size: int = 1

    # TODO: add more error messages
    def id(self) -> str:
        def proc(s: str) -> str:
            return s.removeprefix("torch.")

        id_str = "-".join([proc(str(f)) for f in astuple(self)])
        return f"[{id_str}]"

    # TODO: add more error messages
    @staticmethod
    def from_id(id: str) -> "MoETestConfig":
        id = id[1:-1]
        str_values = id.split("-")

        def convert(v: str, ty):
            if isinstance(ty, types.UnionType):
                sub_ty = list(get_args(ty))
                assert len(sub_ty) == 2 and types.NoneType in sub_ty
                sub_ty.remove(types.NoneType)
                return sub_ty[0](v) if v != "None" else None
            elif ty is torch.dtype:
                ty_val = getattr(torch, v, None)
                assert isinstance(ty_val, torch.dtype)
                return ty_val
            elif ty is bool:
                return v == "True"
            else:
                return ty(v)

        values = tuple(
            [convert(v, f.type) for v, f in zip(str_values, fields(MoETestConfig))]
        )
        return MoETestConfig(*values)


def generate_valid_test_configs(
    backend: str,
    ep_size: int,
    dp_size: int,
    tp_size: int,
    enable_eplb: bool,
    verbosity: int = 0,
) -> list[MoETestConfig]:
    configs: list[MoETestConfig] = []

    for (
        shape,
        num_experts,
        top_k,
        quantization,
        use_shared_experts,
        use_gate,
        use_routed_input_transform,
    ) in product(
        SHAPE_COMBOS,
        NUM_EXPERTS,
        TOP_KS,
        QUANT_METHODS,
        [False, True],  # shared
        [False, True],  # gate
        [False, True],  # routed input exform
    ):
        config = MoETestConfig(
            shape[0],  # m
            shape[1],  # n
            shape[2],  # k
            num_experts,
            top_k,
            torch.bfloat16,
            quantization,
            use_shared_experts,
            use_gate,
            use_routed_input_transform,
            enable_eplb,
            backend,
            ep_size,
            dp_size,
            tp_size,
        )

        valid, reason = is_valid_config(config)
        if valid:
            configs.append(config)
        elif verbosity > 1:
            print(f"Skipping invalid config {config} - {reason}")

    return configs


# TODO: break this up into sections
def is_valid_config(config: MoETestConfig) -> tuple[bool, str | None]:
    # routed_input_transform only makes sense with shared_experts (latent MoE)
    # TODO: not sure this is true
    if config.use_routed_input_transform and not config.use_shared_experts:
        return False, "routed_input_transform requires shared_experts"

    # TODO: disable for now
    if config.use_routed_input_transform and config.enable_eplb:
        return False, "routed_input_transform not supported with EPLB."

    # TODO: disable for now
    if config.use_routed_input_transform and config.use_gate:
        return (
            False,
            "routed_input_transform not supported with gate because of "
            "padding problems",
        )

    # TODO: disable for now
    if config.use_routed_input_transform and config.backend in [
        "deepep_low_latency",
        "deepep_high_throughput",
    ]:
        return (
            False,
            "routed_input_transform not supported with DeepEP backends because "
            "of padding problems",
        )

    # routed_input_transform + quantization + high hidden dimensions
    # TODO: Disable >= 2048 for now due to insane errors.
    if (
        config.use_routed_input_transform
        and config.quantization is not None
        and config.k >= 2048
    ):
        return (
            False,
            "routed_input_transform + quantization + higher hidden dimensions "
            "leads to large differences.",
        )

    # gate requires shared_experts (use_overlapped mode)
    # TODO: also not sure this is true
    if config.use_gate and not config.use_shared_experts:
        return False, "gate requires shared_experts (use_overlapped mode)"

    # Skip modelopt_fp4 if not on B100+ (compute capability 10.0+)
    if (
        config.quantization == "modelopt_fp4"
        and not current_platform.has_device_capability(100)
    ):
        return False, "modelopt_fp4 not supported on H100+ GPUs"

    # Skip flashinfer_nvlink if not on H100+ (compute capability 10.0+)
    if (
        config.backend is not None
        and config.backend.startswith("flashinfer_nvlink")
        and not current_platform.has_device_capability(90)
    ):
        return False, "flashinfer_nvlink needs H100+ GPUs"

    # Backend-specific checks
    if config.backend is not None:
        supported_quants = BACKEND_SUPPORTED_QUANTS.get(config.backend)
        if supported_quants is not None and config.quantization not in supported_quants:
            return (
                False,
                f"{config.backend} does not support quantization={config.quantization}",
            )

        if config.backend == "deepep_low_latency":
            from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (  # noqa: E501
                DeepEPLLPrepareAndFinalize,
            )

            if config.k not in DeepEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
                return (
                    False,
                    f"Skipping unsupported K {config.k} in {config.backend} w/o EP.",
                )

        if config.backend == "nixl_ep":
            from vllm.model_executor.layers.fused_moe.nixl_ep_prepare_finalize import (  # noqa: E501
                NixlEPPrepareAndFinalize,
            )

            if config.k not in NixlEPPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
                return (
                    False,
                    f"Skipping unsupported K {config.k} in {config.backend} w/o EP.",
                )

    if config.backend is not None:
        supports_ep_dp, supports_dp, supports_tp = BACKEND_EP_DP_TP_SUPPORT[
            config.backend
        ]

        if config.tp_size > 1 and not supports_tp:
            return False, f"{config.backend} does not support TP."

        if config.dp_size > 1 and config.ep_size == 1 and not supports_dp:
            return False, f"{config.backend} does not support DP."

        if config.dp_size > 1 and config.ep_size > 1 and not supports_ep_dp:
            return False, f"{config.backend} does not support EP/DP."
    else:
        if config.tp_size > 1 or config.ep_size > 1 or config.dp_size > 1:
            return False, "An all2all backend is required for parallelism."

    if config.enable_eplb:
        if config.ep_size == 1:
            return False, "EPLB requires EP."

        if config.quantization not in EPLB_SUPPORTED_QUANTS:
            return False, f"EPLB not supported with {config.quantization} quantization."

        if config.backend not in EPLB_SUPPORTED_BACKENDS:
            return False, f"EPLB not supported with {config.backend}."

        if config.num_experts % config.dp_size != 0:
            return False, "EPLB requires num_experts divisible by ep_size"

    # Disable fp4 tests until flashinfer is updated or the Dockerfile is
    # modified to install cublasLt.h. See #39525.
    if (
        config.quantization == "modelopt_fp4"
        and current_platform.is_device_capability_family(100)
    ):
        return False, "Temporarily skip until #39525 is resolved"

    return True, None


def chunk_scales_by_rank(
    t: torch.Tensor | None,
    r: int,
    w: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        # Calculate start index by summing chunk sizes for all previous ranks
        # start = sum(rank_chunk(t.shape[0], i, w) for i in range(r))
        # chunk = rank_chunk(t.shape[0], r, w)
        # t = t[start:(start + chunk)]
        chunk = rank_chunk(t.shape[0], r, w)
        t = t[(r * chunk) : max(t.shape[0], (r + 1) * chunk)]

    if t is not None and device is not None:
        t = t.to(device)

    return t


def chunk_scales(
    t: torch.Tensor | None,
    start: int,
    end: int,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if t is not None and t.numel() > 1:
        t = t[start:end]

    if t is not None and device is not None:
        t = t.to(device)

    return t


@dataclass
class QuantizedWeights:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor | None = None
    w2_weight_scale: torch.Tensor | None = None
    w13_weight_scale_2: torch.Tensor | None = None
    w2_weight_scale_2: torch.Tensor | None = None
    w13_input_scale: torch.Tensor | None = None
    w2_input_scale: torch.Tensor | None = None


def _quantize_fp8_halves(
    w1: torch.Tensor,
    w2: torch.Tensor,
    block_shape: list[int] | None = None,
) -> QuantizedWeights:
    """Quantize w13 gate/up halves separately to FP8, producing per-shard scales."""
    half = w1.shape[1] // 2
    w1q_a, w1s_a, _ = moe_quantize_weights(
        w1[:, :half, :],
        None,
        fp8_dtype,
        False,
        block_shape,
    )
    w1q_b, w1s_b, _ = moe_quantize_weights(
        w1[:, half:, :],
        None,
        fp8_dtype,
        False,
        block_shape,
    )
    assert w1s_a is not None and w1s_b is not None

    w2q, w2s, _ = moe_quantize_weights(w2, None, fp8_dtype, False, block_shape)
    assert w2s is not None

    if block_shape is not None:
        # Blocked quantization: scales have shape (E, n_tiles, k_tiles)
        # Concatenate gate and up scales along the n_tiles dimension (dim=1)
        # to match the concatenation of gate and up weights
        w13_weight_scale = torch.cat([w1s_a, w1s_b], dim=1)
        # w2 scales keep their blocked shape (E, k_tiles, n_tiles)
        w2_weight_scale = w2s
    else:
        # Non-blocked quantization: scales have shape (E, 1, 1)
        # Each w1s_x is (E, 1, 1) -> reshape to (E, 1), cat to (E, 2)
        w13_weight_scale = torch.cat([w1s_a.view(-1, 1), w1s_b.view(-1, 1)], dim=1)
        # w2s is (E, 1, 1) -> reshape to (E,)
        w2_weight_scale = w2s.view(-1)

    return QuantizedWeights(
        w13_weight=torch.cat([w1q_a, w1q_b], dim=1),
        w2_weight=w2q,
        w13_weight_scale=w13_weight_scale,
        w2_weight_scale=w2_weight_scale,
    )


def quantization_to_quant_dtype(
    quantization: str | None,
) -> torch.dtype | str | None:
    if quantization is None:
        return None
    elif quantization in ["fp8", "fp8_blocked", "modelopt_fp8"]:
        return fp8_dtype
    elif quantization in ["modelopt_fp4"]:
        return "nvfp4"
    else:
        raise NotImplementedError(f"Unsupported quantization: {quantization}")


def make_quant_config(
    quantization: str | None,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
) -> tuple[QuantizationConfig | None, QuantizedWeights]:
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config

    if quantization is None:
        return None, QuantizedWeights(w13_weight=w1, w2_weight=w2)

    if quantization == "fp8":
        return Fp8Config(True), _quantize_fp8_halves(w1, w2)

    if quantization == "fp8_blocked":
        block_shape = [128, 128]
        return Fp8Config(True, weight_block_size=block_shape), _quantize_fp8_halves(
            w1, w2, block_shape
        )

    if quantization == "modelopt_fp8":
        qw = _quantize_fp8_halves(w1, w2)
        # why?
        qw.w13_input_scale = torch.ones(
            num_experts, dtype=torch.float32, device=w1.device
        )
        # why?
        qw.w2_input_scale = torch.ones(
            num_experts, dtype=torch.float32, device=w2.device
        )
        quant_config = ModelOptFp8Config(
            quant_method="FP8",
            is_checkpoint_fp8_serialized=True,
            kv_cache_quant_method=None,
            exclude_modules=[],
        )
        return quant_config, qw

    if quantization == "modelopt_fp4":
        # Quantize full w13 at once so both gate/up halves share the same
        # global scale per expert.  process_weights_after_loading uses
        # w13_weight_scale_2[:, 0] for the entire tensor, so the two shard
        # scales must match.
        w1q, w1s, w1gs = moe_quantize_weights(w1, None, "nvfp4", False, None)
        assert w1s is not None and w1gs is not None

        w2q, w2s, w2gs = moe_quantize_weights(w2, None, "nvfp4", False, None)
        assert w2s is not None and w2gs is not None

        qw = QuantizedWeights(
            w13_weight=w1q,
            w2_weight=w2q,
            w13_weight_scale=w1s,
            w2_weight_scale=w2s,
            # weight_scale_2 = 1/w_gs: the kernel computes
            # g_alphas = a_scale * w_scale_2, and correct dequant needs 1/w_gs.
            # Expand per-expert scalar to (E, 2) for the two shards.
            w13_weight_scale_2=(1.0 / w1gs).unsqueeze(1).expand(-1, 2).contiguous(),
            w2_weight_scale_2=1.0 / w2gs,
            w13_input_scale=torch.ones(
                (num_experts, 2), dtype=torch.float32, device=w1.device
            ),
            w2_input_scale=torch.ones(
                num_experts, dtype=torch.float32, device=w2.device
            ),
        )
        quant_config = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
        )
        return quant_config, qw

    raise NotImplementedError(f"Unsupported quantization: {quantization}")


@dataclass
class SharedExpertsConfig:
    w1: torch.Tensor
    w2: torch.Tensor
    w1_s: torch.Tensor | None = None
    w2_s: torch.Tensor | None = None
    quant_dtype: torch.dtype | str | None = None


@dataclass
class MoETestData:
    """Container for MOE test data and transforms."""

    w1: torch.Tensor
    w2: torch.Tensor
    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    shared_experts_config: SharedExpertsConfig | None
    gate: torch.nn.Module | None
    routed_input_transform: torch.nn.Module | None
    routed_output_transform: torch.nn.Module | None
    routed_expert_hidden_size: int


class SimpleGate(torch.nn.Module):
    """Simple gate module for testing: computes router logits from hidden states."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        dtype: torch.dtype,
        device: str = "cuda",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(num_experts, hidden_size, device=device, dtype=dtype) / 10
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Returns (router_logits, None) to match expected signature."""
        router_logits = torch.nn.functional.linear(hidden_states, self.weight)
        return router_logits, None


class SimpleRoutedInputTransform(torch.nn.Module):
    """Simple linear transform for testing routed input transform
    (e.g., latent projection).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        device: str = "cuda",
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, device=device, dtype=dtype) / 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.weight)


def create_shared_experts_from_config(
    shared_experts_config: SharedExpertsConfig | None,
    in_dtype: torch.dtype,
    tp_size: int = 1,
    tp_rank: int = 0,
    device: torch.device | str | None = None,
) -> TestMLP | None:
    """Create TestMLP for shared experts from config.

    Args:
        shared_experts_config: Configuration for shared experts
        in_dtype: Output data type
        tp_size: Tensor parallel size (for weight chunking)
        tp_rank: Tensor parallel rank (for weight chunking)
        device: Device to move weights to (optional)

    Returns:
        TestMLP instance or None if config is None
    """
    if shared_experts_config is None:
        return None

    s_w1 = shared_experts_config.w1
    s_w2 = shared_experts_config.w2

    # Apply TP chunking if needed
    if tp_size > 1:
        s_w1 = tp_chunk_gate_up(s_w1, tp_rank, tp_size, dim=1, device=device)
        s_w2 = chunk_by_rank(s_w2, tp_rank, tp_size, dim=0, device=device)
    else:
        s_w1 = s_w1.to(device)
        s_w2 = s_w2.to(device)

    return TestMLP(w1=s_w1, w2=s_w2, out_dtype=in_dtype)


# Make version that takes a MoETestConfig?
def setup_moe_test_data(
    m: int,
    k: int,
    n: int,
    num_experts: int,
    in_dtype: torch.dtype,
    use_shared_experts: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    backend: str | None,
    device: str = "cuda",
) -> MoETestData:
    """Setup test data and transforms for MOE tests.

    Args:
        m: Number of tokens
        k: Hidden size
        n: Intermediate size
        num_experts: Number of experts
        in_dtype: Data type for tensors
        use_shared_experts: Whether to create shared experts config
        use_gate: Whether to create gate module
        use_routed_input_transform: Whether to create routed input/output transforms
        device: Device to create tensors on ("cuda" or "cpu")

    Returns:
        MoETestData containing all test data and transforms
    """
    # For latent MoE: latent_size = k // 2
    latent_size = k // 2

    # k = maybe_roundup_layer_hidden_size(k, in_dtype, backend)
    # latent_size = maybe_roundup_layer_hidden_size(latent_size, in_dtype, backend)

    # Determine dimensions for routed experts (may be transformed)
    # For latent MoE, routed experts operate entirely in latent space
    # (k//2). The routed_output_transform then projects back to k before
    # adding with shared experts.
    # w1: (E, 2*N, latent_size) - input latent_size
    # w2: (E, latent_size, N) - output latent_size (fused_experts returns
    # same shape as input)
    routed_expert_hidden_size = latent_size if use_routed_input_transform else k

    # Create expert weights
    (w1, _, _, _), (w2, _, _, _) = make_test_weights(
        num_experts,
        n,
        routed_expert_hidden_size,  # Both w1 input and w2 output use latent_size
        in_dtype=in_dtype,
    )

    # Create shared experts config if needed
    if use_shared_experts:
        shared_experts_config = SharedExpertsConfig(
            w1=torch.randn((k, n * 2), device=device, dtype=in_dtype) / 15,
            w2=torch.randn((n, k), device=device, dtype=in_dtype) / 15,
        )
    else:
        shared_experts_config = None

    # Create routed input transform if needed
    routed_input_transform = (
        SimpleRoutedInputTransform(k, latent_size, in_dtype, device=device)
        if use_routed_input_transform
        else None
    )

    # Create gate if needed
    # Note: gate is called AFTER routed_input_transform, so it should expect
    # the transformed dimension (latent_size) when routed_input_transform is used
    gate_input_dim = latent_size if use_routed_input_transform else k
    gate = (
        SimpleGate(gate_input_dim, num_experts, in_dtype, device=device)
        if use_gate
        else None
    )

    # Create routed output transform if needed (projects latent space back to original)
    routed_output_transform = (
        SimpleRoutedInputTransform(latent_size, k, in_dtype, device=device)
        if use_routed_input_transform
        else None
    )

    # Create test inputs
    hidden_states = torch.randn((m, k), device=device, dtype=in_dtype) / 10
    router_logits = torch.randn((m, num_experts), device=device, dtype=in_dtype)

    return MoETestData(
        w1=w1,
        w2=w2,
        hidden_states=hidden_states,
        router_logits=router_logits,
        shared_experts_config=shared_experts_config,
        gate=gate,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
        routed_expert_hidden_size=routed_expert_hidden_size,
    )


def make_fused_moe_layer(
    quantization: str | None,
    use_ep: bool,
    hidden_size: int,
    intermediate_size: int,
    in_dtype: torch.dtype,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    renormalize: bool = False,
    shared_experts: torch.nn.Module | None = None,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: torch.dtype | None = None,
    expert_map: torch.Tensor | None = None,
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
    num_redundant_experts: int = 0,
    has_bias: bool = False,
    gate: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
    pcp_size: int | None = 1,
) -> FusedMoE:
    quant_config, qw = make_quant_config(quantization, w1, w2, global_num_experts)

    kwargs = dict()
    kwargs["shared_experts"] = shared_experts

    # Add gate and routed_input_transform if provided
    if gate is not None:
        kwargs["gate"] = gate

    if routed_input_transform is not None:
        kwargs["routed_input_transform"] = routed_input_transform
        kwargs["routed_output_transform"] = routed_output_transform

    layer = FusedMoE(
        num_experts=global_num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=in_dtype,
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        quant_config=quant_config,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        pcp_size=pcp_size,
        prefix="from_forward_context",
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        apply_router_weight_on_input=apply_router_weight_on_input,
        activation=activation,
        enable_eplb=enable_eplb,
        num_redundant_experts=num_redundant_experts,
        has_bias=has_bias,
        **kwargs,
    )

    weight_scale_name = getattr(layer.quant_method, "weight_scale_name", "weight_scale")

    for name, value in [
        ("w13_weight", qw.w13_weight),
        ("w2_weight", qw.w2_weight),
        (f"w13_{weight_scale_name}", qw.w13_weight_scale),
        (f"w2_{weight_scale_name}", qw.w2_weight_scale),
        ("w13_weight_scale_2", qw.w13_weight_scale_2),
        ("w2_weight_scale_2", qw.w2_weight_scale_2),
        ("w13_input_scale", qw.w13_input_scale),
        ("w2_input_scale", qw.w2_input_scale),
    ]:
        if value is not None:
            layer.register_parameter(
                name, torch.nn.Parameter(value, requires_grad=False)
            )

    layer.quant_method.process_weights_after_loading(layer)

    return layer


def make_fake_moe_layer(
    w1: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    global_num_experts: int,
    in_dtype: torch.dtype,
    quantization: str | None,
    renormalize: bool = False,
    shared_experts_config: SharedExpertsConfig | None = None,
    use_grouped_topk: bool = False,
    topk_group: int | None = None,
    num_expert_group: int | None = None,
    custom_routing_function: Callable | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
    apply_router_weight_on_input: bool = False,
    activation: str = "silu",
    indices_type: torch.dtype | None = None,
    expert_map: torch.Tensor | None = None,
    enable_eplb: bool = False,
    expert_load_view: torch.Tensor | None = None,
    logical_to_physical_map: torch.Tensor | None = None,
    logical_replica_count: torch.Tensor | None = None,
    gate: torch.nn.Module | None = None,
    routed_input_transform: torch.nn.Module | None = None,
    routed_output_transform: torch.nn.Module | None = None,
    use_ep: bool = False,
    tp_size: int = 1,
    dp_size: int = 1,
    ep_size: int = 1,
) -> Callable:
    quant_dtype = None
    activation = MoEActivation.from_str(activation)

    router = create_fused_moe_router(
        top_k=top_k,
        global_num_experts=global_num_experts,
        # eplb_state=None, # TODO
        renormalize=renormalize,
        use_grouped_topk=use_grouped_topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        custom_routing_function=custom_routing_function,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        e_score_correction_bias=e_score_correction_bias,
        num_fused_shared_experts=0,  # TODO
        enable_eplb=enable_eplb,
        # TODO(bnell): once we can construct the MK at init time, we
        # can make this a value.
        indices_type_getter=lambda: indices_type,
    )

    if quant_dtype is not None:
        w1, w1_s, _ = moe_quantize_weights(w1, None, quant_dtype, False, None)
        w2, w2_s, _ = moe_quantize_weights(w2, None, quant_dtype, False, None)
    else:
        w1_s = None
        w2_s = None

    shared_experts = create_shared_experts_from_config(
        shared_experts_config, in_dtype, 1, 0, "cuda"
    )

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype,
        w1_scale=w1_s,
        w2_scale=w2_s,
    )

    def _moe(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # Save original hidden_states for shared experts (before transform)
        original_hidden_states = hidden_states

        # Apply routed input transform if provided
        if routed_input_transform is not None:
            hidden_states = routed_input_transform(hidden_states)

        # If gate provided, compute router_logits from hidden_states
        # Note: gate operates on transformed hidden_states (after
        # routed_input_transform)
        if gate is not None:
            router_logits, _ = gate(hidden_states)

        topk_weights, topk_ids = router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )

        # Shared experts use original (untransformed) hidden_states
        if shared_experts is not None:
            shared_output = shared_experts(original_hidden_states)
        else:
            shared_output = None

        # Routed experts use transformed hidden_states
        output = fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            quant_config=quant_config,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )

        # Apply routed output transform if provided
        # (e.g., latent space -> original space)
        if routed_output_transform is not None:
            output = routed_output_transform(output)

        if shared_experts is not None:
            assert shared_output is not None
            output += shared_output

        # Apply TP/DP reduction if not already reduced
        # if (tp_size > 1 or dp_size > 1):
        #    output = tensor_model_parallel_all_reduce(output)

        return output

    return _moe


def _test_body_regular(
    moe_layer: Callable,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    vllm_config: VllmConfig,
    num_tokens: int,
    num_tokens_across_dp: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Regular MoE test body: compare layer output to baseline."""
    baseline_output = kwargs["baseline_output"]

    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output = moe_layer(hidden_states, router_logits)

    return baseline_output, output


def _test_body_eplb(
    moe_layer: FusedMoE,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    vllm_config: VllmConfig,
    num_tokens: int,
    num_tokens_across_dp: torch.Tensor,
    cpu_group,
    in_dtype: torch.dtype,
    quantization: str | None,
    use_ep: bool,
    tp_size: int,
    ep_size: int,
    dp_size: int,
    w1: torch.Tensor,
    w2: torch.Tensor,
    num_experts: int,
    k: int,
    n: int,
    top_k: int,
    shared_experts,
    gate: torch.nn.Module | None,
    routed_input_transform: torch.nn.Module | None,
    routed_output_transform: torch.nn.Module | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.accelerator.current_accelerator()

    """EPLB test body: compare output before and after expert weight rearrangement."""
    # Get "before" output with original weight arrangement
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output_before = moe_layer(hidden_states, router_logits)

    # Create a fresh FusedMoE layer with enable_eplb=True
    # Delete the original layer's registration so the constructor can
    # re-use the same "from_forward_context" prefix
    cc = vllm_config.compilation_config
    del cc.static_forward_context["from_forward_context"]
    cc.static_all_moe_layers.remove("from_forward_context")

    # Determine hidden size for MoE layer
    # When using routed_input_transform, experts operate in latent space
    hidden_size_for_layer = k // 2 if routed_input_transform is not None else k

    eplb_moe_layer = make_fused_moe_layer(
        quantization=quantization,
        use_ep=use_ep,
        hidden_size=hidden_size_for_layer,
        intermediate_size=n,
        in_dtype=in_dtype,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        w1=w1,
        w2=w2,
        top_k=top_k,
        global_num_experts=num_experts,
        shared_experts=shared_experts,
        enable_eplb=True,
        gate=gate,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
    )

    if eplb_moe_layer._expert_map is not None:
        eplb_moe_layer._expert_map = eplb_moe_layer._expert_map.to(device)

    # All ranks must generate the same permutation
    initial_indices = torch.arange(num_experts, dtype=torch.long)
    shuffled_indices = initial_indices[torch.randperm(num_experts)]

    expert_weights = [list(eplb_moe_layer.get_expert_weights())]

    communicator = create_eplb_communicator(
        group_coordinator=get_eplb_group(),
        backend=vllm_config.parallel_config.eplb_config.communicator,
        expert_weights=expert_weights[0],
    )

    # Rearrange expert weights across EP ranks
    rearrange_expert_weights_inplace(
        old_global_expert_indices=initial_indices.unsqueeze(0),
        new_global_expert_indices=shuffled_indices.unsqueeze(0),
        expert_weights=expert_weights,
        ep_group=cpu_group,
        communicator=communicator,
    )

    # Build logical_to_physical_map from shuffled_indices
    # shuffled_indices[physical] = logical, we need the inverse
    logical_to_physical = torch.empty(num_experts, dtype=torch.int32, device=device)
    logical_to_physical[shuffled_indices.to(device)] = torch.arange(
        num_experts, dtype=torch.int32, device=device
    )

    eplb_moe_layer.set_eplb_state(
        moe_layer_idx=0,
        expert_load_view=torch.zeros(
            (1, num_experts),
            dtype=torch.int32,
            device=device,
        ),
        logical_to_physical_map=logical_to_physical.reshape(num_experts, 1).unsqueeze(
            0
        ),
        logical_replica_count=torch.ones(
            (1, num_experts),
            dtype=torch.int32,
            device=device,
        ),
    )

    eplb_moe_layer.eplb_state.should_record_tensor = torch.ones(
        (), dtype=torch.bool, device=device
    )

    # Get "after" output with rearranged weights and EPLB routing
    with set_forward_context(
        None,
        vllm_config,
        num_tokens=num_tokens,
        num_tokens_across_dp=num_tokens_across_dp,
    ):
        output_after = eplb_moe_layer(hidden_states, router_logits)

    return output_before, output_after


# TODO: make this take a MoETestConfig
def _run_one_config(
    vllm_config: VllmConfig,
    ep_size: int,
    dp_size: int,
    tp_size: int,
    dp_rank: int,
    tp_rank: int,
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: str | None,
    backend: str | None,
    test_body_fn: Callable,
    use_shared_experts: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    **kwargs,
) -> None:
    set_random_seed(7)

    """Generic test loop that sets up environment and delegates to test_body_fn.

    This function is called directly by test_moe_layer and test_moe_layer_eplb
    via parallel_launch_with_config, passing either _test_body_regular or
    _test_body_eplb as the test_body_fn parameter.
    """
    world_size = tp_size * dp_size
    use_ep = ep_size > 1

    assert vllm_config.parallel_config.enable_expert_parallel == use_ep

    in_dtype = torch.bfloat16
    device = torch.accelerator.current_accelerator()

    if not is_workspace_manager_initialized():
        init_workspace_manager(device)

    # Create test data and transforms
    test_data = setup_moe_test_data(
        m=m,
        k=k,
        n=n,
        num_experts=num_experts,
        in_dtype=in_dtype,
        use_shared_experts=use_shared_experts,
        use_gate=use_gate,
        use_routed_input_transform=use_routed_input_transform,
        backend=backend,
        device=device,
    )

    # Extract data from test_data
    hidden_states = test_data.hidden_states
    router_logits = test_data.router_logits
    w1 = test_data.w1
    w2 = test_data.w2
    shared_experts_config = test_data.shared_experts_config
    gate = test_data.gate
    routed_input_transform = test_data.routed_input_transform
    routed_output_transform = test_data.routed_output_transform
    activation = "silu"

    baseline_layer = make_fake_moe_layer(
        w1=w1,
        w2=w2,
        top_k=top_k,
        global_num_experts=num_experts,
        in_dtype=in_dtype,
        quantization=quantization,
        renormalize=False,
        shared_experts_config=shared_experts_config,
        gate=gate,
        routed_input_transform=routed_input_transform,
        routed_output_transform=routed_output_transform,
        use_ep=use_ep,
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
        activation=activation,
    )

    baseline_output = baseline_layer(hidden_states, router_logits)

    del baseline_layer
    torch.accelerator.empty_cache()

    with set_current_vllm_config(vllm_config):
        # Chunk weights for EP/TP (after baseline is created)
        if ep_size > 1:
            w1 = chunk_by_rank(w1, dp_rank, dp_size, dim=0, device=device)
            w2 = chunk_by_rank(w2, dp_rank, dp_size, dim=0, device=device)

        if tp_size > 1:
            w1 = tp_chunk_gate_up(w1, tp_rank, tp_size, dim=1, device=device)
            w2 = chunk_by_rank(w2, tp_rank, tp_size, dim=2, device=device)

        # Setup shared experts if needed
        shared_experts = create_shared_experts_from_config(
            shared_experts_config, in_dtype, tp_size, tp_rank, device
        )

        # Determine hidden size for MoE layer
        # When using routed_input_transform, experts operate in latent space
        hidden_size_for_layer = k // 2 if routed_input_transform is not None else k

        # Create initial MoE layer
        moe_layer = make_fused_moe_layer(
            quantization=quantization,
            use_ep=use_ep,
            hidden_size=hidden_size_for_layer,
            intermediate_size=n,
            in_dtype=in_dtype,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            w1=w1,
            w2=w2,
            top_k=top_k,
            global_num_experts=num_experts,
            shared_experts=shared_experts,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            activation=activation,
        )

        if moe_layer._expert_map is not None:
            moe_layer._expert_map = moe_layer._expert_map.to(device)

        num_tokens = m
        num_tokens_across_dp = torch.tensor(
            [num_tokens] * world_size,
            device=device,
            dtype=torch.int,
        )

        # Call the test body function with all necessary context
        expected, actual = test_body_fn(
            moe_layer=moe_layer,
            hidden_states=hidden_states,
            router_logits=router_logits,
            vllm_config=vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            in_dtype=in_dtype,
            quantization=quantization,
            use_ep=use_ep,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            w1=w1,
            w2=w2,
            num_experts=num_experts,
            k=k,
            n=n,
            m=m,
            top_k=top_k,
            shared_experts=shared_experts,
            gate=gate,
            routed_input_transform=routed_input_transform,
            routed_output_transform=routed_output_transform,
            baseline_output=baseline_output,
            **kwargs,
        )

    # Common tolerance logic
    # TODO: consider associating tolerances with quant methods.
    if quantization is None:
        if k >= 2048:
            atol, rtol = 7.6e-2, 7.6e-2
        else:
            atol, rtol = 3.5e-2, 3.5e-2
    elif quantization in ("fp8", "fp8_blocked", "modelopt_fp8"):
        atol, rtol = 6e-2, 6e-2
    elif quantization == "modelopt_fp4":
        if k >= 2048:
            atol = rtol = 1e-1 + (k * 1e-4)
        else:
            atol = rtol = 1e-1

        if backend == "allgather_reducescatter" and tp_size > 1:
            atol += 2e-1
            rtol += 2e-1
    else:
        atol, rtol = 6e-2, 6e-2

    torch.accelerator.synchronize()  # TODO: Is this needed?
    torch.testing.assert_close(expected, actual, atol=atol, rtol=rtol)


# Test for non-parallel cases (world_size == 1) - backend doesn't matter
@pytest.mark.parametrize("m, n, k", SHAPE_COMBOS)
@pytest.mark.parametrize("num_experts", NUM_EXPERTS)
@pytest.mark.parametrize("top_k", TOP_KS)
@pytest.mark.parametrize("quantization", QUANT_METHODS)
@pytest.mark.parametrize("use_shared_experts", [False, True])
@pytest.mark.parametrize("use_gate", [False, True])
@pytest.mark.parametrize("use_routed_input_transform", [False, True])
def test_moe_layer_no_parallel(
    m: int,
    n: int,
    k: int,
    num_experts: int,
    top_k: int,
    quantization: str | None,
    use_shared_experts: bool,
    use_gate: bool,
    use_routed_input_transform: bool,
    monkeypatch,
):
    """Test MoE layer without parallelism (dp_size=1, tp_size=1, use_ep=False)."""

    if os.environ.get("VLLM_LOGGING_LEVEL") is None:
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "ERROR")

    # Needed since weights will already be in e4m3fnuz format and the
    # normalize_e4m3fn_to_e4m3fnuz() function is not being tested here.
    if current_platform.is_fp8_fnuz():
        override_normalize_e4m3fn_to_e4m3fnuz()

    test_config = MoETestConfig(
        m,
        n,
        k,
        num_experts,
        top_k,
        torch.bfloat16,
        quantization,
        use_shared_experts,
        use_gate,
        use_routed_input_transform,
    )

    valid, reason = is_valid_config(test_config)
    if not valid:
        pytest.skip(reason)

    set_random_seed(7)

    parallel_config = ParallelConfig()
    compilation_config = CompilationConfig()
    compilation_config.pass_config.fuse_allreduce_rms = False

    vllm_config = VllmConfig(
        parallel_config=parallel_config, compilation_config=compilation_config
    )

    # Initialize distributed environment for single GPU
    _set_vllm_config(vllm_config, 1, rank=0, local_rank=0)

    _run_one_config(
        vllm_config,
        test_config.ep_size,
        test_config.dp_size,
        test_config.tp_size,
        0,
        0,
        test_config.m,
        test_config.n,
        test_config.k,
        test_config.num_experts,
        test_config.top_k,
        test_config.quantization,
        test_config.backend,
        _test_body_regular,
        use_shared_experts=test_config.use_shared_experts,
        use_gate=test_config.use_gate,
        use_routed_input_transform=test_config.use_routed_input_transform,
    )


def _test_body_config(test_config: MoETestConfig, cpu_group, **kwargs):
    if not test_config.enable_eplb:
        return _test_body_regular(**kwargs)
    else:
        return _test_body_eplb(**kwargs, cpu_group=cpu_group)


def _parallel_worker(
    pgi: ProcessGroupInfo,
    vllm_config: VllmConfig,
    cpu_group,
    test_configs: list[MoETestConfig],
    verbosity: int,
    **kwargs,
) -> None:
    set_random_seed(7)

    total = 0
    passed = 0
    failed = 0
    fail_ids = []

    dp_rank = vllm_config.parallel_config.data_parallel_rank

    if current_platform.is_fp8_fnuz():
        override_normalize_e4m3fn_to_e4m3fnuz()

    for test_config in test_configs:
        cc = vllm_config.compilation_config
        if "from_forward_context" in cc.static_forward_context:
            del cc.static_forward_context["from_forward_context"]
            cc.static_all_moe_layers.remove("from_forward_context")

        tp_rank = pgi.rank % test_config.tp_size

        if verbosity > 0:
            print(f"subtest: {test_config.id()}", end="")

        try:
            _run_one_config(
                vllm_config,
                test_config.ep_size,
                test_config.dp_size,
                test_config.tp_size,
                dp_rank,
                tp_rank,
                test_config.m,
                test_config.n,
                test_config.k,
                test_config.num_experts,
                test_config.top_k,
                test_config.quantization,
                test_config.backend,
                functools.partial(
                    _test_body_config, test_config=test_config, cpu_group=cpu_group
                ),
                use_shared_experts=test_config.use_shared_experts,
                use_gate=test_config.use_gate,
                use_routed_input_transform=test_config.use_routed_input_transform,
            )
            if verbosity > 0:
                print(" PASSED")
            else:
                print(".", end="")
            passed = passed + 1
        except Exception as ex:
            fail_ids.append(test_config.id())
            failed = failed + 1
            if verbosity > 0:
                traceback.print_exc()
                print(f"\n{str(ex)}\nFAILED")
            else:
                print("F", end="")
        finally:
            # DeepEP managers are not reliably reusable across many subtests in
            # a single worker process. Tear them down after each DeepEP case so
            # later subtests do not inherit stale communication state.
            if test_config.backend in {
                "deepep_low_latency",
                "deepep_high_throughput",
            }:
                torch.accelerator.synchronize()
                all2all_manager = get_ep_group().device_communicator.all2all_manager
                if all2all_manager is not None:
                    all2all_manager.destroy()
            total = total + 1

    skipped = total - (passed + failed)

    fails = f"{failed} failed" if failed > 0 else ""
    sep = ", " if fails != "" else ""
    skips = f"{sep}{skipped} skipped" if skipped > 0 else ""
    sep = ", " if skips != "" or fails != "" else ""
    passes = f"{sep}{passed} passed" if passed > 0 else ""

    report = (
        f"============= {fails}{skips}{passes} of {total} total tests ============="
    )

    sep = "\n" if verbosity == 0 else ""
    print(f"{sep}{report}")

    if failed > 0:
        fail_ids_str = "\n".join(fail_ids)
        raise RuntimeError(
            f"\n============= Failed subtests =============\n{fail_ids_str}\n{report}"
        )


# TODO: add cudagraphs/torch.compile tests
@pytest.mark.parametrize("dp_size, tp_size, use_ep", PARALLEL_COMBOS)
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("enable_eplb", [False, True])
def test_moe_layer(
    dp_size: int,
    tp_size: int,
    use_ep: bool,
    backend: str,
    enable_eplb: bool,
    monkeypatch,
    pytestconfig,
    subtests,
):
    """Test MoE layer with parallelism (multi-GPU or TP/EP enabled).

    For non-parallel cases (world_size == 1), use test_moe_layer_no_parallel instead.
    """
    num_gpus = current_platform.device_count()
    world_size = tp_size * dp_size
    ep_size = 1 if not use_ep else world_size  # or dp_size?
    assert world_size > 1

    # Check if enough GPUs available
    if world_size is not None and num_gpus is not None and world_size > num_gpus:
        pytest.skip(f"Not enough GPUs got {num_gpus}, expected {world_size}.")

    if enable_eplb and not use_ep:
        pytest.skip("EPLB requires EP.")

    verbosity = pytestconfig.getoption("verbose")

    if os.environ.get("VLLM_LOGGING_LEVEL") is None:
        monkeypatch.setenv("VLLM_LOGGING_LEVEL", "ERROR")

    # TODO
    # VLLM_FLASHINFER_MOE_BACKEND=latency
    # VLLM_USE_FLASHINFER_MOE_FP16=1
    # VLLM_USE_FLASHINFER_MOE_FP8
    # VLLM_USE_FLASHINFER_MOE_FP4
    # VLLM_USE_FLASHINFER_MOE_INT4

    parallel_config = ParallelConfig(
        pipeline_parallel_size=1,
        data_parallel_size=dp_size,
        tensor_parallel_size=tp_size,
        enable_expert_parallel=use_ep,
        all2all_backend=backend,
        enable_eplb=enable_eplb,
    )

    compilation_config = CompilationConfig()
    # compilation_config.mode = CompilationMode.NONE  # for now
    compilation_config.pass_config.fuse_allreduce_rms = False  # for now

    vllm_config = VllmConfig(
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        scheduler_config=SchedulerConfig.default_factory(
            max_num_batched_tokens=next_power_of_2(MAX_M)
        ),
    )

    test_configs = generate_valid_test_configs(
        backend, ep_size, dp_size, tp_size, enable_eplb, verbosity
    )

    if subtests is not None:
        new_test_configs = []
        for subtest in subtests.split(","):
            sub_test_config = MoETestConfig.from_id(subtest)
            if sub_test_config in test_configs:
                new_test_configs.append(sub_test_config)
            else:
                pytest.skip(
                    f"subtest config {subtest} does not match any valid test "
                    "configuration"
                )
        test_configs = new_test_configs

    if len(test_configs) == 0:
        pytest.skip("No supported configs found for this testpoint.")

    try:
        parallel_launch_with_config(
            world_size,
            _parallel_worker,
            vllm_config,
            None,
            test_configs,
            verbosity,
        )
    finally:
        torch.accelerator.synchronize()  # TODO: Is this needed?
        torch.accelerator.empty_cache()
