# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the online MXFP4 weight quantization backends.

Each backend (Triton, aiter, XPU) quantizes a synthetic bf16/fp16
tensor to MXFP4, the result is dequantized back to bf16/fp16, and compared
against the pure-torch reference in `reference_mxfp4.py`.
"""

import pytest
import torch

from vllm.config.model import ModelConfig
from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP4_KERNELS
from vllm.model_executor.kernels.linear.mxfp4.aiter import (
    AiterMxfp4LinearKernel,
)
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.quark.utils import (
    quark_quantize_weight_to_mxfp4,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    downcast_to_mxfp,
    mxfp4_quantize,
    quant_dequant_mxfp4,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import PlatformEnum, current_platform

from .reference_mxfp4 import dq_mxfp4_torch, qdq_mxfp4_torch


def fix_negative_zeros(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize FP4 e2m1 negative-zero codewords (0b1000) to positive zero
    (0b0000) in a packed uint8 tensor (two e2m1 values per byte, low nibble
    first).

    -0.0 and +0.0 dequantize to the same float value, but different MXFP4
    quantization backends disagree on which one they emit for values that
    round to zero magnitude. This normalizes both packed representations so
    tensors from different backends can be compared with `torch.equal`.
    """
    assert tensor.dtype == torch.uint8, (
        f"Expected a torch.uint8 tensor, got {tensor.dtype}"
    )

    low_nibble = tensor & 0x0F
    high_nibble = tensor & 0xF0

    low_nibble = torch.where(
        low_nibble == 0x08, torch.zeros_like(low_nibble), low_nibble
    )
    high_nibble = torch.where(
        high_nibble == 0x80, torch.zeros_like(high_nibble), high_nibble
    )

    return low_nibble | high_nibble


def assert_quantized_weights_equal(
    checkpoint_layer: torch.nn.Module,
    online_layer: torch.nn.Module,
    keys: tuple[str, ...],
    packed_weight_keys: tuple[str, ...],
) -> None:
    """Assert the checkpoint and online paths produced byte-identical weights.

    `packed_weight_keys` lists the entries holding packed FP4 codewords, which
    need negative-zero normalization before comparison.
    """
    for key in keys:
        checkpoint_tensor = getattr(checkpoint_layer, key).view(torch.uint8)
        online_tensor = getattr(online_layer, key).view(torch.uint8)

        # NOTE: AMD Quark checkpoints use exclusively **positive** zeros,
        # while other mxfp4_quantize implementations from mxfp4_utils.py may not.
        if key in packed_weight_keys:
            checkpoint_tensor = fix_negative_zeros(checkpoint_tensor)
            online_tensor = fix_negative_zeros(online_tensor)

        assert checkpoint_tensor.shape == online_tensor.shape
        assert checkpoint_tensor.dtype == online_tensor.dtype
        num_mismatched = (checkpoint_tensor != online_tensor).sum().item()
        total = checkpoint_tensor.numel()

        assert torch.equal(checkpoint_tensor, online_tensor), (
            f"{key}: {num_mismatched}/{total} "
            f"({100 * num_mismatched / total:.4f}%) mismatched bytes"
        )


def _skip_reason_if_unavailable(backend: str, dtype: torch.dtype) -> str | None:
    """Return a skip reason if `backend` cannot run on the current host."""
    if backend == "triton":
        if not (current_platform.is_cuda() or current_platform.is_rocm()):
            return "Triton MXFP4 kernel requires a CUDA or ROCm GPU."
        return None
    if backend == "aiter":
        from vllm._aiter_ops import is_aiter_found_and_supported

        if not is_aiter_found_and_supported():
            return "aiter is not available/supported on this platform."
        if dtype != torch.bfloat16:
            return "aiter's dynamic_mxfp4_quant only supports bfloat16 input."
        return None
    if backend == "xpu":
        if not current_platform.is_xpu():
            return "not on XPU platform."
        return None
    if backend == "quark":
        try:
            import quark.torch.kernel.mx  # noqa: F401
        except ImportError:
            return "amd-quark is not installed."
        return None
    raise ValueError(f"Unknown backend {backend}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("backend", ["triton", "aiter", "xpu", "quark"])
def test_mxfp4_quantization_correctness(backend: str, dtype: torch.dtype):
    """Tests that the different implementations of mxfp4_quantize
    in mxfp4_utils.py all match.
    """
    skip_reason = _skip_reason_if_unavailable(backend, dtype)
    if skip_reason is not None:
        pytest.skip(skip_reason)

    torch.manual_seed(3)

    num_rows = 64
    hidden_size = 32 * 32  # multiple 32-element MXFP4 blocks
    device = current_platform.device_type

    x = (torch.rand(num_rows, hidden_size, dtype=dtype, device=device) - 0.5) * 2
    # Vary the magnitude block-to-block so several scale exponents are
    # exercised, rather than a single one for the whole tensor.
    scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
    for i in range(hidden_size // 32):
        x[:, i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]

    if backend == "triton":
        x_fp4, x_scale, _ = downcast_to_mxfp(x, axis=-1)
    elif backend == "aiter":
        x_fp4, x_scale = quark_quantize_weight_to_mxfp4(x)
    elif backend == "xpu":
        # TODO: enable this test on XPU
        pytest.skip("xpu mxfp4 quantization match to reference is untested")
        # from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
        #     xpu_mxfp4_quantize,
        # )

        # x_fp4, x_scale = xpu_mxfp4_quantize(x)
    elif backend == "quark":
        result = quant_dequant_mxfp4(x, scale_calculation_mode="even")
    else:
        raise ValueError(f"Unknown backend {backend}")

    if backend != "quark":
        result = dq_mxfp4_torch(x_fp4, x_scale, x.dtype)
    reference = qdq_mxfp4_torch(x, scale_calculation_mode="even")

    assert torch.equal(result, reference)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only tested on ROCm/CUDA."
)
@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_online_mxfp4_tp_weight_quant_matches_unsharded(tp_size: int):
    """TP-aligned MXFP4 shards match slices of unsharded quantization."""
    torch.manual_seed(3)

    device = current_platform.device_type
    hidden_size = 1024
    intermediate_size = 512

    dense_weight = torch.randn(
        hidden_size, intermediate_size, dtype=torch.bfloat16, device=device
    )
    w13_weight = torch.randn(
        2,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_weight = torch.randn(
        2,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )

    dense_quantized, dense_scales = mxfp4_quantize(dense_weight)
    w13_quantized, w13_scales = mxfp4_quantize(w13_weight)
    w2_quantized, w2_scales = mxfp4_quantize(w2_weight)

    for tp_rank in range(tp_size):
        output_per_partition = hidden_size // tp_size
        input_per_partition = intermediate_size // tp_size
        output_start = tp_rank * output_per_partition
        input_start = tp_rank * input_per_partition

        column_quantized, column_scales = mxfp4_quantize(
            dense_weight.narrow(0, output_start, output_per_partition)
        )
        assert torch.equal(
            column_quantized,
            dense_quantized.narrow(0, output_start, output_per_partition),
        )
        assert torch.equal(
            column_scales,
            dense_scales.narrow(0, output_start, output_per_partition),
        )

        row_quantized, row_scales = mxfp4_quantize(
            dense_weight.narrow(1, input_start, input_per_partition)
        )
        assert torch.equal(
            row_quantized,
            dense_quantized.narrow(1, input_start // 2, input_per_partition // 2),
        )
        assert torch.equal(
            row_scales,
            dense_scales.narrow(1, input_start // 32, input_per_partition // 32),
        )

        w13_quantized_shard, w13_scales_shard = mxfp4_quantize(
            w13_weight.narrow(1, 2 * input_start, 2 * input_per_partition)
        )
        assert torch.equal(
            w13_quantized_shard,
            w13_quantized.narrow(1, 2 * input_start, 2 * input_per_partition),
        )
        assert torch.equal(
            w13_scales_shard,
            w13_scales.narrow(1, 2 * input_start, 2 * input_per_partition),
        )

        w2_quantized_shard, w2_scales_shard = mxfp4_quantize(
            w2_weight.narrow(2, input_start, input_per_partition)
        )
        assert torch.equal(
            w2_quantized_shard,
            w2_quantized.narrow(2, input_start // 2, input_per_partition // 2),
        )
        assert torch.equal(
            w2_scales_shard,
            w2_scales.narrow(2, input_start // 32, input_per_partition // 32),
        )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only tested on ROCm/CUDA."
)
@pytest.mark.parametrize("moe_backend", ["aiter", "emulation"])
@pytest.mark.parametrize(
    "unpadded_hidden_size,unpadded_intermediate_size",
    [
        pytest.param(256, 256, id="no_padding"),
        # Not multiples of the MXFP4 block size (32), so that every backend --
        # including emulation, whose round-up granularity is 32 -- has to pad.
        pytest.param(240, 208, id="padding"),
    ],
)
def test_online_mxfp4_moe_matches_quark(
    moe_backend: str,
    unpadded_hidden_size: int,
    unpadded_intermediate_size: int,
    default_vllm_config,
    dist_init,
    monkeypatch,
):
    """Ensures `Mxfp4OnlineMoEMethod` (online quantization)
    and `QuarkOCP_MX_MoEMethod` (AMD Quark checkpoints) produce the same weights,
    with same MOE backend used.

    The `padding` case additionally covers `maybe_roundup_sizes`: the online
    path allocates its bf16 weights with `torch.empty_strided` and the loader
    only writes the unpadded slice, so the padding is filled with NaN here to
    reproduce that uninitialized state. It must be zeroed before quantization
    for the two paths to agree, since Quark allocates its padding with
    `torch.zeros`.
    """
    if moe_backend == "aiter":
        from vllm._aiter_ops import rocm_aiter_ops

        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
        monkeypatch.setattr("vllm.platforms.rocm.on_gfx950", lambda: True)
        rocm_aiter_ops.refresh_env_variables()

        if current_platform.is_cuda():
            pytest.skip(
                "mxfp4_backend == Mxfp4MoeBackend.AITER_MXFP4_MXFP4 requires "
                "rocm_aiter_ops in weight conversion, not compatible on cuda"
            )
    elif moe_backend == "emulation":
        # `OCP_MXQuantizationEmulationTritonExperts.is_supported_config` gates
        # on `has_quark()`, but its weight processing does not require it.
        monkeypatch.setattr(
            "vllm.model_executor.layers.fused_moe.experts."
            "ocp_mx_emulation_moe.has_quark",
            lambda: True,
        )

    default_vllm_config.model_config = ModelConfig()

    num_experts = 4
    device = current_platform.device_type

    def make_layer(prefix: str) -> RoutedExperts:
        runner = FusedMoEFactory(
            num_experts=num_experts,
            top_k=2,
            hidden_size=unpadded_hidden_size,
            intermediate_size=unpadded_intermediate_size,
            prefix=prefix,
        )
        layer = runner.routed_experts
        layer.moe_config.moe_backend = moe_backend
        return layer

    # `create_weights` implementations use plain `torch.zeros`/`torch.randn`
    # with no explicit device, so without this context they would default
    # to CPU and diverge from the (GPU) tensors produced by
    # `mxfp4_quantize`/`replace_parameter`.
    with torch.device(device):
        checkpoint_layer = make_layer("checkpoint_layer")
        online_layer = make_layer("online_layer")

        checkpoint_method = QuarkOCP_MX_MoEMethod(
            weight_config={"qscheme": "per_group", "dtype": "fp4"},
            input_config={"dtype": "fp4", "is_dynamic": True},
            moe=checkpoint_layer.moe_config,
        )
        online_method = Mxfp4OnlineMoEMethod(layer=online_layer)

        # `RoutedExperts.__init__` applies this round-up in production; these
        # layers are built without a quant config, so it is applied explicitly.
        # Both methods must agree on the padded sizes.
        roundup_arguments = dict(
            hidden_size=unpadded_hidden_size,
            intermediate_size_per_partition=unpadded_intermediate_size,
            act_dtype=torch.bfloat16,
            moe_parallel_config=online_layer.moe_config.moe_parallel_config,
        )
        hidden_size, intermediate_size = online_method.maybe_roundup_sizes(
            **roundup_arguments
        )
        assert (
            hidden_size,
            intermediate_size,
        ) == checkpoint_method.maybe_roundup_sizes(**roundup_arguments)

        if unpadded_hidden_size == 240:
            # Padded case.
            assert hidden_size > unpadded_hidden_size
            assert intermediate_size > unpadded_intermediate_size

        for layer in (checkpoint_layer, online_layer):
            assert layer.moe_config.hidden_dim_unpadded == unpadded_hidden_size
            assert (
                layer.moe_config.intermediate_size_per_partition_unpadded
                == unpadded_intermediate_size
            )
            layer.moe_config.hidden_dim = hidden_size
            layer.moe_config.intermediate_size_per_partition = intermediate_size

        gate_up_weight = torch.randn(
            num_experts,
            2 * unpadded_intermediate_size,
            unpadded_hidden_size,
            dtype=torch.bfloat16,
        )
        down_weight = torch.randn(
            num_experts,
            unpadded_hidden_size,
            unpadded_intermediate_size,
            dtype=torch.bfloat16,
        )

        scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
        for i, start in enumerate(range(0, unpadded_hidden_size, 32)):
            gate_up_weight[..., start : start + 32] *= scalings[i % len(scalings)]
        for i, start in enumerate(range(0, unpadded_intermediate_size, 32)):
            down_weight[..., start : start + 32] *= scalings[i % len(scalings)]

        def scatter_into_padded(
            gate_up_weight: torch.Tensor,
            down_weight: torch.Tensor,
            padding_value: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Place the source weights into buffers of the padded size, the
            way the weight loader writes only the unpadded slice of a larger
            allocation.
            """
            w13 = torch.full(
                (num_experts, 2 * intermediate_size, hidden_size),
                padding_value,
                dtype=torch.bfloat16,
            )
            w2 = torch.full(
                (num_experts, hidden_size, intermediate_size),
                padding_value,
                dtype=torch.bfloat16,
            )
            # w13 stacks gate and up, each padded to `intermediate_size`.
            w13[:, :unpadded_intermediate_size, :unpadded_hidden_size] = gate_up_weight[
                :, :unpadded_intermediate_size, :
            ]
            w13[
                :,
                intermediate_size : intermediate_size + unpadded_intermediate_size,
                :unpadded_hidden_size,
            ] = gate_up_weight[:, unpadded_intermediate_size:, :]
            w2[:, :unpadded_hidden_size, :unpadded_intermediate_size] = down_weight
            return w13, w2

        # Checkpoint path: pre-quantize the source weights to MXFP4 with the
        # ~same RTN recipe a real Quark checkpoint, then feed the
        # already-packed tensors into `QuarkOCP_MX_MoEMethod`. Quark's
        # `create_weights` allocates with `torch.zeros`, so its padding is zero.
        checkpoint_w13, checkpoint_w2 = scatter_into_padded(
            gate_up_weight, down_weight, padding_value=0.0
        )
        checkpoint_w13, checkpoint_w13_scale = mxfp4_quantize(checkpoint_w13)
        checkpoint_w2, checkpoint_w2_scale = mxfp4_quantize(checkpoint_w2)

        checkpoint_method.create_weights(
            layer=checkpoint_layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )
        checkpoint_layer.w13_weight.data.copy_(checkpoint_w13)
        checkpoint_layer.w2_weight.data.copy_(checkpoint_w2)
        checkpoint_layer.w13_weight_scale.data.copy_(checkpoint_w13_scale)
        checkpoint_layer.w2_weight_scale.data.copy_(checkpoint_w2_scale)
        checkpoint_method.process_weights_after_loading(checkpoint_layer)

        # Online path: feed the *same* raw bf16 source weights and let
        # `Mxfp4OnlineMoEMethod` quantize them during
        # `process_weights_after_loading`. The padding starts out
        # uninitialized, standing in for the materialized meta tensor.
        online_w13, online_w2 = scatter_into_padded(
            gate_up_weight, down_weight, padding_value=float("nan")
        )

        online_method.create_weights(
            layer=online_layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size,
            params_dtype=torch.bfloat16,
        )
        replace_parameter(online_layer, "w13_weight", online_w13)
        replace_parameter(online_layer, "w2_weight", online_w2)
        online_method.process_weights_after_loading(online_layer)

    assert checkpoint_method.mxfp4_backend == online_method.mxfp4_backend

    assert_quantized_weights_equal(
        checkpoint_layer,
        online_layer,
        keys=("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"),
        packed_weight_keys=("w13_weight", "w2_weight"),
    )


@pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="Only tested on ROCm/CUDA."
)
@pytest.mark.parametrize("linear_backend", ["emulation", "aiter", "marlin"])
def test_online_mxfp4_dense_matches_quark(
    linear_backend: str, default_vllm_config, dist_init, monkeypatch
):
    """Ensures `Mxfp4OnlineLinearMethod` (online quantization)
    and `QuarkOCP_MX` (AMD Quark checkpoints) produce the same weights,
    with same linear backend used.
    """
    if linear_backend == "marlin" and not current_platform.is_cuda():
        # `MarlinMxFp4LinearKernel.process_weights_after_loading` is CUDA-only
        pytest.skip(
            "MarlinMxFp4LinearKernel.process_weights_after_loading is CUDA-only."
        )

    if linear_backend == "aiter":
        # `AiterMxfp4LinearKernel` is registered only for ROCm in
        # `_POSSIBLE_MXFP4_KERNELS`, and `is_supported` gates on
        # `current_platform.supports_mx()`. Force it onto the CUDA
        # kernel list and bypass the platform gate so this backend can
        # be exercised on CUDA hosts too.

        monkeypatch.setattr(
            AiterMxfp4LinearKernel,
            "is_supported",
            classmethod(lambda cls, compute_capability=None: (True, None)),
        )
        monkeypatch.setitem(
            _POSSIBLE_MXFP4_KERNELS,
            PlatformEnum.CUDA,
            [
                AiterMxfp4LinearKernel,
                *_POSSIBLE_MXFP4_KERNELS.get(PlatformEnum.CUDA, []),
            ],
        )
    elif linear_backend == "emulation":
        # `EmulationMxfp4LinearKernel.can_implement` gates on `has_quark()`,
        # EmulationMxfp4LinearKernel.process_weights_after_loading does not require it.
        monkeypatch.setattr(
            "vllm.model_executor.kernels.linear.mxfp4.emulation.has_quark",
            lambda: True,
        )

    default_vllm_config.model_config = ModelConfig()
    default_vllm_config.kernel_config.linear_backend = linear_backend

    input_size = 256
    output_size = 128
    device = current_platform.device_type

    # `create_weights` implementations use plain `torch.empty` with no explicit
    # device, so without this context they would default to CPU and diverge
    # from the (GPU) tensors produced by `mxfp4_quantize`/`replace_parameter`.
    with torch.device(device):
        checkpoint_layer = torch.nn.Module()
        online_layer = torch.nn.Module()
        for layer in (checkpoint_layer, online_layer):
            # `LinearBase` subclasses normally set these attributes on the
            # layer before `create_weights` is called; the marlin kernel's
            # `process_weights_after_loading` reads them directly off the
            # layer, so a bare `torch.nn.Module` needs them set explicitly.
            layer.input_size = input_size
            layer.output_size = output_size
            layer.input_size_per_partition = input_size
            layer.output_size_per_partition = output_size
            layer.params_dtype = torch.bfloat16

        weight = torch.randn(output_size, input_size, dtype=torch.bfloat16)

        scalings = [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]
        for i in range(input_size // 32):
            weight[:, i * 32 : (i + 1) * 32] *= scalings[i % len(scalings)]

        # Checkpoint path: pre-quantize the source weight to MXFP4 with the
        # ~same RTN recipe as a real Quark checkpoint, then feed the
        # already-packed tensors into `QuarkOCP_MX`.
        checkpoint_weight, checkpoint_weight_scale = mxfp4_quantize(weight)

        checkpoint_scheme = QuarkOCP_MX(
            weight_quant_spec={"qscheme": "per_group", "dtype": "fp4"},
            input_quant_spec={"dtype": "fp4", "is_dynamic": True},
        )
        checkpoint_scheme.create_weights(
            layer=checkpoint_layer,
            output_partition_sizes=[output_size],
            input_size_per_partition=input_size,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        checkpoint_layer.weight.data.copy_(checkpoint_weight)
        checkpoint_layer.weight_scale.data.copy_(checkpoint_weight_scale)
        checkpoint_scheme.process_weights_after_loading(checkpoint_layer)

        # Online path: feed the *same* raw bf16 source weight and let
        # `Mxfp4OnlineLinearMethod` quantize it during
        # `process_weights_after_loading`.
        online_method = Mxfp4OnlineLinearMethod()
        online_method.create_weights(
            layer=online_layer,
            input_size_per_partition=input_size,
            output_partition_sizes=[output_size],
            input_size=input_size,
            output_size=output_size,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        replace_parameter(online_layer, "weight", weight.clone())
        online_method.process_weights_after_loading(online_layer)

    assert type(checkpoint_scheme.ocp_mx_linear) is type(online_method.kernel)

    assert_quantized_weights_equal(
        checkpoint_layer,
        online_layer,
        keys=("weight", "weight_scale"),
        packed_weight_keys=("weight",),
    )
