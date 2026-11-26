# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness and TP/EP parity for online quantization."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial

import pytest
import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

from tests.quantization.test_online import (
    DEVICE,
    _enable_aiter_mxfp4_moe,
    _fully_quantized_modelopt_mxfp8_config,
    _make_nvfp4_moe_source,
    _materialize_requantization_source,
    _naive_block_fp8_dequant,
    _nvfp4_moe_checkpoint_tensors,
    _serialize_nvfp4_experts,
    _write_minimal_llama_config,
    on_gfx950,
)
from tests.quantization.utils import named_expert_checkpoint_entries, quant_config_args
from tests.single_rank_group import SingleRankGroup, override_parallel_groups
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.kernel import MoEBackend
from vllm.config.model import ModelConfig
from vllm.config.vllm import (
    VllmConfig,
    get_current_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed import parallel_state
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    FusedMoEMethodBase,
    RoutedExperts,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_fp8 import (  # noqa: E501
    CompressedTensorsW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.online.base import (
    OnlineQuantizationConfig,
)
from vllm.model_executor.layers.quantization.online.fp8 import (
    Fp8PerBlockOnlineLinearMethod,
    Fp8PerBlockOnlineMoEMethod,
    Fp8PerTensorOnlineLinearMethod,
    Fp8PerTensorOnlineMoEMethod,
    Fp8PtpcOnlineLinearMethod,
    Fp8PtpcOnlineMoEMethod,
    OnlineLinearBase,
    _is_tp_sharded,
)
from vllm.model_executor.layers.quantization.online.int8 import Int8OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.lut_b import LutBOnlineMoEMethod
from vllm.model_executor.layers.quantization.online.moe_base import OnlineMoEMethodBase
from vllm.model_executor.layers.quantization.online.mxfp4 import (
    Mxfp4OnlineLinearMethod,
    Mxfp4OnlineMoEMethod,
)
from vllm.model_executor.layers.quantization.online.mxfp8 import Mxfp8OnlineMoEMethod
from vllm.model_executor.layers.quantization.online.nvfp4 import (
    Nvfp4OnlineMoEMethod,
    Nvfp4WeightOnlyOnlineLinearMethod,
    _quantize_moe_weight_to_nvfp4,
    _quantize_nvfp4,
)
from vllm.model_executor.layers.quantization.online.requantization import (
    skip_checkpoint_backend_selection,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
    QuarkW8A8Fp8MoEMethod,
    QuarkW8A8Int8MoEMethod,
    QuarkW8A8Mxfp8MoEMethod,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_ocp_mx import (
    QuarkOCP_MX,
)
from vllm.model_executor.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
    QuarkW8A8Fp8,
)
from vllm.model_executor.layers.quantization.utils.lut_b_utils import (
    dequantize_lut_b,
    quantize_lut_b,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    is_fp4_marlin_supported,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_SCALE_DTYPE,
    MXFP8_VALUE_DTYPE,
    mxfp8_e4m3_quantize,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    amax_for_tp_weight_quant,
    get_fp8_min_max,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
    kNvfp4DynamicToken,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import set_default_torch_dtype

# The parametrized cases restore their emulated groups without initializing
# torch.distributed. Run the expensive global cleanup once at module exit.
pytestmark = pytest.mark.skip_global_cleanup


_ReferencePack = tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
_ReferencePacker = Callable[[str, torch.Tensor], _ReferencePack]


LINEAR_SOURCE_QUANT_METHODS: dict[str, Callable[[], QuantizationConfig]] = {
    "mxfp8": _fully_quantized_modelopt_mxfp8_config,
    "nvfp4": partial(ModelOptNvFp4Config, is_checkpoint_nvfp4_serialized=True),
}
LINEAR_ONLINE_QUANT_METHODS: dict[str, type[OnlineLinearBase]] = {
    "mxfp4": Mxfp4OnlineLinearMethod,
    "fp8_per_tensor": Fp8PerTensorOnlineLinearMethod,
    "fp8_per_channel": Fp8PtpcOnlineLinearMethod,
    "fp8_per_block": Fp8PerBlockOnlineLinearMethod,
    "nvfp4_weight_only": Nvfp4WeightOnlyOnlineLinearMethod,
}
MOE_SOURCE_QUANT_METHODS: dict[
    str, Callable[[RoutedExperts], FusedMoEMethodBase] | None
] = {
    "bf16": None,
    "fp8/block": lambda layer: _make_fp8_moe_source(layer, blockwise=True),
    "fp8/tensor": lambda layer: _make_fp8_moe_source(layer, blockwise=False),
    "compressed_tensors/block": lambda layer: _make_ct_fp8_moe_source(layer, "block"),
    "compressed_tensors/tensor": lambda layer: _make_ct_fp8_moe_source(layer, "tensor"),
    "compressed_tensors/channel": lambda layer: _make_ct_fp8_moe_source(
        layer, "channel"
    ),
    "quark/block": lambda layer: _make_quark_fp8_moe_source(
        layer, kFp8Static128BlockSym, kFp8Dynamic128Sym
    ),
    "quark/tensor": lambda layer: _make_quark_fp8_moe_source(
        layer, kFp8StaticTensorSym, kFp8DynamicTensorSym
    ),
    "quark/channel": lambda layer: _make_quark_fp8_moe_source(
        layer, kFp8StaticChannelSym, kFp8DynamicTokenSym
    ),
    "compressed_tensors/nvfp4": lambda layer: _make_nvfp4_moe_source(
        "compressed_tensors", layer.moe_config
    ),
    "modelopt/nvfp4": lambda layer: _make_nvfp4_moe_source(
        "modelopt", layer.moe_config
    ),
    "quark/nvfp4": lambda layer: _make_nvfp4_moe_source("quark", layer.moe_config),
}
MOE_ONLINE_QUANT_METHODS: dict[str, type[OnlineMoEMethodBase]] = {
    "mxfp4": Mxfp4OnlineMoEMethod,
    "fp8_per_tensor": Fp8PerTensorOnlineMoEMethod,
    "fp8_per_channel": Fp8PtpcOnlineMoEMethod,
    "nvfp4_per_token": Nvfp4OnlineMoEMethod,
    "int8": Int8OnlineMoEMethod,
    "fp8_per_block": Fp8PerBlockOnlineMoEMethod,
    "mxfp8": Mxfp8OnlineMoEMethod,
    "lut_b": LutBOnlineMoEMethod,
}
_MOE_PADDING_TARGETS = (
    "mxfp4",
    "fp8_per_tensor",
    "fp8_per_channel",
    "nvfp4_per_token",
    "int8",
)


@pytest.mark.skipif(
    not current_platform.is_cuda_alike() or current_platform.is_fp8_fnuz(),
    reason="Offline FP8 comparison uses the E4M3FN checkpoint format.",
)
@pytest.mark.parametrize("source", tuple(LINEAR_SOURCE_QUANT_METHODS))
@pytest.mark.parametrize("target", tuple(LINEAR_ONLINE_QUANT_METHODS))
@pytest.mark.parametrize(
    "projection,shard_dim",
    [
        pytest.param(RowParallelLinear, 1, id="row-parallel"),
        pytest.param(ColumnParallelLinear, 0, id="column-parallel"),
        pytest.param(ReplicatedLinear, None, id="replicated"),
    ],
)
def test_online_linear_requantization_numerics(
    parallelism: str,
    numerics_model_config: ModelConfig,
    source: str,
    target: str,
    projection: type[RowParallelLinear | ColumnParallelLinear | ReplicatedLinear],
    shard_dim: int | None,
) -> None:
    """Check that sharding and requantizing linear weights agree in either order.

    - Loading a source checkpoint shard and requantizing it online produces
      the same weights and scales as quantizing the full weight first and
      loading the matching shard with the offline target method.
    - This holds for replicated, row-parallel, and column-parallel layers.
    - Before requantization, the dequantized source exactly matches the
      independently reconstructed reference rounded to BF16.
    - Scale reductions use the expected local and peer shard maxima.
    """
    # Check the portable NVFP4 packer even when the W4A16 GEMM is unavailable.
    pack_only = target == "nvfp4_weight_only" and not is_fp4_marlin_supported()
    effective_target = "fp8_per_tensor" if pack_only else target

    tp_size = 1 if parallelism == "tp1" else 2
    for rank in range(tp_size):
        with _emulate_rank(numerics_model_config, rank, tp_size):

            def shard(tensor: torch.Tensor, shard_rank: int = rank) -> torch.Tensor:
                if shard_dim is None:
                    return tensor
                return tensor.chunk(tp_size, dim=shard_dim)[shard_rank].contiguous()

            source_config = LINEAR_SOURCE_QUANT_METHODS[source]()
            source_config.online_quantization_config = OnlineQuantizationConfig(
                quant_config_args(linear=effective_target)
            )
            online_layer = projection(256, 256, bias=False, quant_config=source_config)
            method = online_layer.quant_method
            assert isinstance(method, LINEAR_ONLINE_QUANT_METHODS[effective_target])
            # Global scales span both axes; channel scales span only the input axis.
            assert _is_tp_sharded(online_layer) == (
                shard_dim is not None and tp_size > 1
            )
            assert _is_tp_sharded(online_layer, reduces_output_dim=False) == (
                shard_dim == 1 and tp_size > 1
            )

            reference = _load_linear_checkpoint(online_layer, source, shard)
            local_reference = shard(reference)
            reconstructed = method.get_weight_for_quantization(online_layer)
            # Check reconstruction separately from target quantization.
            torch.testing.assert_close(reconstructed, local_reference, rtol=0, atol=0)

            maxima = []
            if tp_size == 2 and shard_dim is not None:
                local_abs = local_reference.abs()
                peer_abs = shard(reference, 1 - rank).abs()
                if target in ("fp8_per_tensor", "nvfp4_weight_only"):
                    maxima.append(
                        (local_abs.amax().reshape(1), peer_abs.amax().reshape(1))
                    )
                elif target == "fp8_per_channel" and shard_dim == 1:
                    maxima.append(
                        (
                            local_abs.amax(dim=-1, keepdim=True),
                            peer_abs.amax(dim=-1, keepdim=True),
                        )
                    )

            with _emulate_max_reductions(
                parallel_state.get_tp_group().device_group, maxima
            ):
                if pack_only:
                    global_amax = amax_for_tp_weight_quant(
                        reconstructed.abs().amax().reshape(1),
                        _is_tp_sharded(online_layer),
                    )
                    packed, scales, global_scale = _quantize_nvfp4(
                        reconstructed, global_amax
                    )
                else:
                    method.process_weights_after_loading(online_layer)

            if pack_only:
                expected_packed, expected_scales, expected_global_scale = (
                    _pack_nvfp4_linear_reference(reference)
                )
                torch.testing.assert_close(
                    packed, shard(expected_packed), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales.float(), shard(expected_scales).float(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    global_scale, expected_global_scale, rtol=0, atol=0
                )
                continue

            if target == "fp8_per_block":
                # Requantization must replace the source scales.
                assert not hasattr(online_layer, "weight_scale")
                assert online_layer.weight_scale_inv.numel() > 0

            offline_layer = _prepare_offline_linear(
                projection,
                target,
                reference,
                shard,
                shard_dim,
                local_reference.shape[-1],
            )
            if target == "nvfp4_weight_only":
                torch.testing.assert_close(
                    online_layer.weight_global_scale,
                    offline_layer.weight_global_scale,
                    rtol=0,
                    atol=0,
                )
            _assert_requantized_weights(offline_layer, online_layer, target, "")


@pytest.mark.parametrize("source", tuple(MOE_SOURCE_QUANT_METHODS))
@pytest.mark.parametrize("target", tuple(MOE_ONLINE_QUANT_METHODS))
@pytest.mark.parametrize(
    "case",
    (
        "packing",
        "padding-gated",
        "padding-ungated",
        "nan-padding-aiter",
        "nan-padding-emulation",
    ),
)
@pytest.mark.parametrize("parallelism", ("tp1", "tp2", "ep2"), indirect=True)
def test_online_moe_requantization_numerics(
    parallelism: str,
    numerics_model_config: ModelConfig,
    source: str,
    target: str,
    case: str,
    unsharded_moe_reference: _ReferencePacker,
) -> None:
    """Check that online quantization gives each MoE rank the right weights.

    - Loading and quantizing a TP shard or an EP rank's whole experts online
      produces the same weights and scales as quantizing all the expert
      weights first and loading that rank's portion with the offline target
      method. For LUT-B, compare the dequantized target weights instead.
    - Serialized sources match the BF16 reference before requantization,
      including FP8 sources with a refined scale grid.
    - EP loads local experts and skips checkpoint entries owned by other ranks.
    - Gated and ungated experts have the expected TP padding. Biases and
      attempts to shrink weights are rejected, and NaNs in unloaded BF16
      padding do not change MXFP4 weights or scales.
    """
    from vllm.model_executor.layers.quantization.utils import fp8_utils
    from vllm.utils.flashinfer import has_flashinfer_trtllm_fused_moe
    from vllm.utils.import_utils import has_quark

    poison_padding = case.startswith("nan-padding")
    check_padding = case.startswith("padding-")
    if not current_platform.is_cuda_alike():
        pytest.skip("Requires CUDA or ROCm.")
    if (check_padding or poison_padding) and parallelism == "ep2":
        pytest.skip("Padding checks exercise TP layouts only.")
    if poison_padding and (source, target) != ("bf16", "mxfp4"):
        pytest.skip("Poisoned padding checks BF16-to-MXFP4 quantization.")
    if check_padding and (source == "bf16" or target not in _MOE_PADDING_TARGETS):
        pytest.skip(
            "This padding check requires a serialized source and a supported target."
        )
    if case == "nan-padding-aiter" and not current_platform.is_rocm():
        pytest.skip("AITER MXFP4 is ROCm-only.")
    if source != "bf16" and current_platform.is_fp8_fnuz():
        pytest.skip("Offline FP8 comparisons use the E4M3FN checkpoint format.")
    if case == "packing" and target == "mxfp4" and not on_gfx950() and not has_quark():
        pytest.skip("MXFP4 MoE emulation requires Quark.")
    if target == "nvfp4_per_token" and not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(100)
        and (case != "packing" or has_flashinfer_trtllm_fused_moe())
    ):
        pytest.skip("Per-token NVFP4 MoE requires Blackwell and FlashInfer TRTLLM.")

    use_aiter = case == "nan-padding-aiter" or (
        case == "packing" and target == "mxfp4" and on_gfx950()
    )

    world_size = 1 if parallelism == "tp1" else 2
    for rank in range(world_size):
        with (
            _emulate_rank(
                numerics_model_config,
                rank,
                world_size,
                expert_parallel=parallelism == "ep2",
            ),
            ExitStack() as cleanup,
        ):
            if use_aiter:
                # Restore the cached flags after monkeypatch restores the environment.
                cleanup.callback(rocm_aiter_ops.refresh_env_variables)
            patch = cleanup.enter_context(pytest.MonkeyPatch.context())
            if use_aiter:
                _enable_aiter_mxfp4_moe(patch)
            elif poison_padding:
                # Weight packing does not need Quark's inference runtime.
                patch.setattr(
                    "vllm.model_executor.layers.fused_moe.experts."
                    "ocp_mx_emulation_moe.has_quark",
                    lambda: True,
                )
            if world_size == 2 and case == "packing":
                patch.setattr(fp8_utils, "_DEQUANT_CHUNK_BYTES", 1)

            sizes = (208,) if poison_padding else (256,)
            if use_aiter and source == "fp8/block" and world_size == 1:
                # The second shape exercises a refined source block grid.
                sizes = (256, 192)
            for intermediate_size in sizes:
                online_rank = _create_online_moe(
                    source, target, rank, world_size, case, intermediate_size
                )
                online_layer = online_rank.layer
                checkpoint = _make_moe_checkpoint(
                    source, online_rank, vary_block_scales=poison_padding
                )
                if check_padding:
                    _copy_checkpoint_shard(online_rank, checkpoint, source)
                    _check_moe_padding(
                        online_rank.method,
                        online_layer,
                        _reference_shard(online_rank, checkpoint),
                    )
                    continue

                if intermediate_size == 192:
                    # The source refines its scale grid without configuring the target.
                    source_method = online_rank.method.requantization_source
                    assert isinstance(source_method, Fp8MoEMethod)
                    assert source_method.scale_block_size != [128, 128]
                    assert online_rank.method.uses_meta_device
                    assert online_layer.weight_block_size != [128, 128]
                    assert online_layer.w13_weight.dtype == torch.float8_e4m3fn

                entries = _named_moe_checkpoint(checkpoint, source, online_rank)
                padded_reference = None
                if poison_padding:
                    moe_config = online_layer.moe_config
                    assert moe_config.hidden_dim > online_rank.hidden_size
                    assert (
                        moe_config.intermediate_size_per_partition
                        > intermediate_size // world_size
                    )
                    padded_reference = _pad_local_moe_weights(online_rank, checkpoint)
                    poisoned_weights = _pad_local_moe_weights(
                        online_rank, checkpoint, value=float("nan")
                    )
                    for name, weight in zip(
                        ("w13_weight", "w2_weight"), poisoned_weights, strict=True
                    ):
                        replace_parameter(online_layer, name, weight)
                elif intermediate_size == 192:
                    # The loader currently expands the final partial block to
                    # one extra scale. Keep testing requantization on the
                    # already-refined grid without exercising that loader bug.
                    _copy_checkpoint_shard(online_rank, checkpoint, source)
                else:
                    local_entries = list(entries(online_rank.local_experts))
                    assert len(list(online_layer.load_weights(local_entries))) == len(
                        local_entries
                    )

                if target != "lut_b":
                    if not poison_padding:
                        torch.testing.assert_close(
                            online_rank.method.get_weights_for_quantization(
                                online_layer
                            ),
                            _reference_shard(online_rank, checkpoint),
                            rtol=0,
                            atol=0,
                        )
                    offline_layer, offline_method = _prepare_offline_moe(
                        online_rank,
                        checkpoint,
                        target,
                        unsharded_moe_reference,
                        padded_reference=padded_reference,
                    )

                if online_layer.moe_config.use_ep and source != "bf16":
                    nonlocal_experts = [
                        expert
                        for expert in range(online_rank.num_experts)
                        if expert not in online_rank.local_experts
                    ]
                    assert (
                        list(online_layer.load_weights(entries(nonlocal_experts))) == []
                    )
                _process_online_moe(online_rank, checkpoint, target)
                if intermediate_size == 192:
                    assert not hasattr(online_layer, "w13_weight_scale_inv")
                    assert not hasattr(online_layer, "w2_weight_scale_inv")

                if target == "lut_b":
                    # LUT-B emulation stores dequantized weights rather than bytes.
                    for projection, weight in zip(
                        ("w13", "w2"), checkpoint.reference, strict=True
                    ):
                        expected, _, _ = unsharded_moe_reference(projection, weight)
                        torch.testing.assert_close(
                            getattr(online_layer, f"{projection}_weight"),
                            online_rank.shard(expected, projection),
                            rtol=0,
                            atol=0,
                        )
                else:
                    offline_method.process_weights_after_loading(offline_layer)
                    _assert_requantized_weights(
                        offline_layer, online_layer, target, "w13_", "w2_"
                    )


@pytest.fixture(scope="module", autouse=True)
def _cleanup_numerics_module() -> Iterator[None]:
    yield
    parallel_state.cleanup_dist_env_and_memory()


@pytest.fixture(scope="module", autouse=True)
def dynamo_reset(reset_dynamo_caches: Callable[[], None]) -> Iterator[None]:
    """Reuse compiled FP8 casts across the source/target matrix."""
    try:
        # Cover the different rank shapes, dtypes, and strides without falling
        # back to eager when their combined specializations exceed the default.
        with torch._dynamo.config.patch(
            recompile_limit=32, fail_on_recompile_limit_hit=True
        ):
            yield
    finally:
        reset_dynamo_caches()


class _EmulatedRankGroup(SingleRankGroup):
    """Logical TP/EP membership for a rank running on device 0."""

    def __init__(self, rank: int, world_size: int) -> None:
        self.rank_in_group = rank
        self.world_size = world_size
        self.is_first_rank = rank == 0
        self.is_last_rank = rank == world_size - 1

    def rank(self) -> int:
        return self.rank_in_group

    def size(self) -> int:
        return self.world_size


@pytest.fixture(scope="module", params=["tp1", "tp2"])
def parallelism(request: pytest.FixtureRequest) -> str:
    if torch.accelerator.device_count() == 0:
        pytest.skip("Requires one accelerator.")
    return request.param


@pytest.fixture(scope="module")
def numerics_model_config(
    parallelism: str, tmp_path_factory: pytest.TempPathFactory
) -> ModelConfig:
    path = tmp_path_factory.mktemp(f"online_numerics_{parallelism}")
    _write_minimal_llama_config(path)
    return ModelConfig(model=str(path), dtype="bfloat16", skip_tokenizer_init=True)


@contextmanager
def _emulate_rank(
    model_config: ModelConfig, rank: int, world_size: int, expert_parallel: bool = False
) -> Iterator[None]:
    """Expose one logical rank on device 0 without real process groups."""
    torch.accelerator.set_device_index(0)
    config = VllmConfig(model_config=model_config)
    # The synthetic Llama config is dense; tests build the MoE after validation.
    config.parallel_config.enable_expert_parallel = expert_parallel
    with (
        override_parallel_groups(SingleRankGroup()),
        pytest.MonkeyPatch.context() as patch,
        set_current_vllm_config(config),
        set_default_torch_dtype(torch.bfloat16),
        torch.device(f"{DEVICE}:0"),
    ):
        patch.setattr(parallel_state, "_TP", _EmulatedRankGroup(rank, world_size))
        # MoE weight MAX uses the EP communicator, even under TP.
        patch.setattr(parallel_state, "_EP", _EmulatedRankGroup(rank, world_size))
        torch.manual_seed(11)
        yield


@contextmanager
def _emulate_max_reductions(
    expected_group: object, expected_maxima: Iterable[tuple[torch.Tensor, torch.Tensor]]
) -> Iterator[None]:
    """Check each local contribution and add the peer's shard maximum."""
    pending = iter(expected_maxima)

    def all_reduce(tensor: torch.Tensor, op: object, group: object) -> None:
        # Weight scales need a MAX on the intended TP/EP group, not a sum.
        assert op == torch.distributed.ReduceOp.MAX
        assert group is expected_group
        pair = next(pending, None)
        assert pair is not None, "Unexpected weight MAX reduction"
        local_maximum, peer_maximum = pair
        # Validate the actual local contribution before simulating the merge.
        torch.testing.assert_close(tensor, local_maximum, rtol=0, atol=0)
        tensor.copy_(torch.maximum(tensor, peer_maximum))

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.distributed, "all_reduce", all_reduce)
        yield
    # A bypassed collective must fail even if both ranks happen to pick the same max.
    assert next(pending, None) is None, "Missing weight MAX reduction"


def _offline_quantize(
    weight: torch.Tensor, target: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if target == "mxfp4":
        return mxfp4_quantize(weight)
    if target == "fp8_per_block":
        output_size, input_size = weight.shape
        tiles = weight.reshape(output_size // 128, 128, input_size // 128, 128)
        fp8_dtype = current_platform.fp8_dtype()
        scale = tiles.float().abs().amax((1, 3), keepdim=True).clamp_min(1e-4)
        _, fp8_max = get_fp8_min_max()
        scale = scale / fp8_max
        quantized = (tiles * scale.reciprocal()).to(fp8_dtype)
        return quantized.reshape(output_size, input_size), scale[:, 0, :, 0]
    # In contrast to the online path, the dynamic quantizer sees all TP shards.
    per_channel = target == "fp8_per_channel"
    if weight.ndim == 2:
        return ops.scaled_fp8_quant(weight, use_per_token_if_dynamic=per_channel)
    values, scales = zip(
        *(
            ops.scaled_fp8_quant(expert_weight, use_per_token_if_dynamic=per_channel)
            for expert_weight in weight
        ),
        strict=True,
    )
    scale_tensor = torch.stack(scales) if per_channel else torch.cat(scales)
    return torch.stack(values), scale_tensor


def _assert_requantized_weights(
    offline_layer: torch.nn.Module,
    online_layer: torch.nn.Module,
    target: str,
    *parameter_prefixes: str,
) -> None:
    from tests.quantization.test_online_mxfp4 import fix_negative_zeros

    for prefix in parameter_prefixes:
        expected = getattr(offline_layer, f"{prefix}weight")
        actual = getattr(online_layer, f"{prefix}weight")
        if target in ("mxfp4", "nvfp4_per_token", "nvfp4_weight_only"):
            actual = fix_negative_zeros(actual.view(torch.uint8))
            expected = fix_negative_zeros(expected.view(torch.uint8))
        elif expected.is_floating_point():
            actual, expected = actual.float(), expected.float()
        # Compare after kernel layout conversion, not just canonical packing.
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        scale_name = "weight_scale_inv" if target == "fp8_per_block" else "weight_scale"
        online_scale_name = "scale" if target == "int8" else scale_name
        actual_scale = getattr(online_layer, f"{prefix}{online_scale_name}")
        expected_scale = getattr(offline_layer, f"{prefix}{scale_name}")
        if target == "mxfp4":
            # Preserve exact E8M0 scale encodings as well as the packed FP4 values.
            actual_scale = actual_scale.view(torch.uint8)
            expected_scale = expected_scale.view(torch.uint8)
        else:
            actual_scale, expected_scale = actual_scale.float(), expected_scale.float()
        if target == "int8":
            expected_scale = expected_scale.squeeze(-1)
        if target == "fp8_per_tensor" and not prefix:
            # The checkpoint loader stores its single scale as a scalar.
            actual_scale = actual_scale.reshape(1)
            expected_scale = expected_scale.reshape(1)
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
        if target == "nvfp4_per_token":
            torch.testing.assert_close(
                getattr(online_layer, f"{prefix}weight_scale_2"),
                getattr(offline_layer, f"{prefix}weight_scale_2"),
                rtol=0,
                atol=0,
            )


def _load_linear_checkpoint(
    online_layer: LinearBase, source: str, shard: Callable[[torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Load the local source shard and return the full BF16 reference."""
    input_size, output_size = online_layer.input_size, online_layer.output_size
    _materialize_requantization_source(online_layer)
    if source == "mxfp8":
        weight = torch.randn(output_size, input_size).to(MXFP8_VALUE_DTYPE)
        scale = torch.randint(
            124, 131, (output_size, input_size // 32), dtype=MXFP8_SCALE_DTYPE
        )
        # Both row and column partitions must have distinct maxima.
        scale[:, input_size // 64 :] += 3
        scale[output_size // 2 :] += 3
        reference = weight.float() * (2.0 ** (scale.float() - 127)).repeat_interleave(
            32, dim=-1
        )
    else:
        global_scale = torch.tensor([97.0], dtype=torch.float32)
        weights = torch.randn(1, output_size, input_size)
        weights[..., input_size // 2 :] *= 2
        weights[:, output_size // 2 :] *= 2
        weight, scale, reference = (
            tensor[0] for tensor in _serialize_nvfp4_experts(weights, global_scale)
        )
        online_layer.weight_scale_2.data.copy_(global_scale.reciprocal())

    online_layer.weight.data.copy_(shard(weight))
    online_layer.weight_scale.data.copy_(shard(scale))
    return reference.to(torch.bfloat16)


def _pack_nvfp4_linear_reference(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_scale = (weight.abs().amax().float() / (6 * 448)).clamp_min(1e-12)
    packed, scales, _ = _serialize_nvfp4_experts(
        weight[None], global_scale.reciprocal()[None]
    )
    return packed[0], scales[0], global_scale


def _prepare_offline_linear(
    projection: type[RowParallelLinear | ColumnParallelLinear | ReplicatedLinear],
    target: str,
    reference: torch.Tensor,
    shard: Callable[[torch.Tensor], torch.Tensor],
    shard_dim: int | None,
    local_input_size: int,
) -> LinearBase:
    """Quantize the full reference, then prepare the matching offline shard."""
    output_size, input_size = reference.shape
    if target == "nvfp4_weight_only":
        packed, scales, global_scale = _pack_nvfp4_linear_reference(reference)
        offline_layer = projection(
            input_size,
            output_size,
            bias=False,
            quant_config=ModelOptNvFp4Config(
                is_checkpoint_nvfp4_serialized=True, quant_method="W4A16_NVFP4"
            ),
        )
        offline_layer.weight.data.copy_(shard(packed))
        offline_layer.weight_scale.data.copy_(shard(scales))
        offline_layer.weight_scale_2.data.fill_(global_scale)
        assert offline_layer.quant_method is not None
        offline_layer.quant_method.process_weights_after_loading(offline_layer)
        return offline_layer

    reference_weights, reference_scales = _offline_quantize(reference, target)
    offline_layer = projection(input_size, output_size, bias=False)
    if target == "mxfp4":
        checkpoint = QuarkOCP_MX(
            weight_quant_key=kMxfp4Static, activation_quant_key=kMxfp4Dynamic
        )
        reference_scales = shard(reference_scales)
    elif target == "fp8_per_channel":
        checkpoint = QuarkW8A8Fp8(kFp8StaticChannelSym, kFp8DynamicTokenSym)
        if shard_dim == 0:
            reference_scales = shard(reference_scales)
        reference_scales = reference_scales.squeeze(-1)
    else:
        block_size = [128, 128] if target == "fp8_per_block" else None
        checkpoint = Fp8LinearMethod(Fp8Config(weight_block_size=block_size))
        if block_size is not None:
            reference_scales = shard(reference_scales)

    local_reference_weights = shard(reference_weights)
    checkpoint.create_weights(
        layer=offline_layer,
        output_partition_sizes=[local_reference_weights.shape[0]],
        input_size_per_partition=local_input_size,
        params_dtype=torch.bfloat16,
        weight_loader=default_weight_loader,
        **(
            {"input_size": input_size, "output_size": output_size}
            if target != "mxfp4"
            else {}
        ),
    )
    offline_layer.weight.data.copy_(local_reference_weights)
    scale_name = "weight_scale_inv" if target == "fp8_per_block" else "weight_scale"
    getattr(offline_layer, scale_name).data.copy_(reference_scales)
    checkpoint.process_weights_after_loading(offline_layer)
    return offline_layer


def _make_fp8_moe_source(layer: RoutedExperts, blockwise: bool) -> Fp8MoEMethod:
    return Fp8MoEMethod(
        Fp8Config(weight_block_size=[128, 128] if blockwise else None), layer
    )


def _make_quark_fp8_moe_source(
    layer: RoutedExperts, weight_key: QuantKey, activation_key: QuantKey
) -> QuarkW8A8Fp8MoEMethod:
    return QuarkW8A8Fp8MoEMethod(layer.moe_config, weight_key, activation_key)


def _make_ct_fp8_moe_source(
    layer: RoutedExperts, scheme: str
) -> CompressedTensorsW8A8Fp8MoEMethod:
    return CompressedTensorsW8A8Fp8MoEMethod(
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=QuantizationStrategy(scheme),
            block_structure=[128, 128] if scheme == "block" else None,
            dynamic=False,
        ),
        QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            strategy=(
                QuantizationStrategy.TOKEN
                if scheme == "channel"
                else QuantizationStrategy.TENSOR
            ),
            dynamic=True,
        ),
        layer.moe_config,
    )


def _fp8_moe_checkpoint_tensors(
    source_format: str,
    scheme: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    block_size: list[int],
    num_gate_up_projections: int,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Build serialized expert weights and independently dequantize them to FP32."""
    checkpoint_tensors = {}
    reference_weights = []
    projection_shapes = (
        ("w13", num_gate_up_projections * intermediate_size, hidden_size),
        ("w2", hidden_size, intermediate_size),
    )
    for projection, output_size, input_size in projection_shapes:
        weight = torch.randn(num_experts, output_size, input_size).to(
            torch.float8_e4m3fn
        )
        if scheme == "block":
            block_rows, block_columns = block_size
            num_projections = num_gate_up_projections if projection == "w13" else 1
            # Gate and up have separate scale grids, including partial blocks.
            scale_rows = (
                cdiv(output_size // num_projections, block_rows) * num_projections
            )
            scales = (
                torch.rand(
                    num_experts,
                    scale_rows,
                    cdiv(input_size, block_columns),
                    dtype=torch.float32,
                )
                + 0.5
            )
            dequantized = torch.cat(
                [
                    _naive_block_fp8_dequant(weight_part, scale_part, block_size)
                    for weight_part, scale_part in zip(
                        weight.chunk(num_projections, dim=1),
                        scales.chunk(num_projections, dim=1),
                        strict=True,
                    )
                ],
                dim=1,
            )
        elif scheme == "channel":
            scales = torch.rand(num_experts, output_size, 1, dtype=torch.float32) + 0.5
            dequantized = weight.float() * scales
        elif projection == "w13":
            scales = (
                torch.rand(num_experts, num_gate_up_projections, dtype=torch.float32)
                + 0.5
            )
            dequantized = (
                weight.float().unflatten(
                    1, (num_gate_up_projections, intermediate_size)
                )
                * scales[:, :, None, None]
            ).flatten(1, 2)
        else:
            scales = torch.rand(num_experts, dtype=torch.float32) + 0.5
            dequantized = weight.float() * scales[:, None, None]

        scale_name = (
            "weight_scale_inv"
            if source_format == "fp8" and scheme == "block"
            else "weight_scale"
        )
        checkpoint_tensors[f"{projection}_weight"] = weight
        checkpoint_tensors[f"{projection}_{scale_name}"] = scales
        reference_weights.append(dequantized)
    return checkpoint_tensors, reference_weights


def _pack_moe_reference(weight: torch.Tensor, target: str) -> _ReferencePack:
    if target == "lut_b":
        packed, codebooks, scales = quantize_lut_b(weight)
        return dequantize_lut_b(packed, codebooks, scales), None, None
    if target == "nvfp4_per_token":
        # Compare native BF16 packing of shards against native packing of
        # full experts. The PyTorch packer can choose other codes on ties.
        return _quantize_moe_weight_to_nvfp4(weight)
    if target == "int8":
        scales = weight.abs().amax(dim=-1, keepdim=True) / 127
        packed_weights = (weight / scales).round().clamp(-127, 127).to(torch.int8)
    elif target in ("fp8_per_block", "mxfp8"):
        quantized = (
            _offline_quantize(expert_weight.float(), target)
            if target == "fp8_per_block"
            else mxfp8_e4m3_quantize(expert_weight, is_sf_swizzled_layout=False)
            for expert_weight in weight
        )
        packed_weights, scales = (
            torch.stack(parts) for parts in zip(*quantized, strict=True)
        )
    else:
        packed_weights, scales = _offline_quantize(weight, target)
    return packed_weights, scales, None


@pytest.fixture
def unsharded_moe_reference(target: str) -> _ReferencePacker:
    """Pack each full projection once per source, target, and shape."""
    packs: dict[tuple[str, torch.Size], _ReferencePack] = {}

    def pack(projection: str, weight: torch.Tensor) -> _ReferencePack:
        # Run lazily inside the first rank's device and backend contexts.
        key = (projection, weight.shape)
        if key not in packs:
            packs[key] = _pack_moe_reference(weight, target)
        return packs[key]

    return pack


def _moe_offline_method(target: str, layer: RoutedExperts) -> FusedMoEMethodBase:
    moe_config = layer.moe_config
    if target == "mxfp4":
        return QuarkOCP_MX_MoEMethod(moe_config, kMxfp4Static, kMxfp4Dynamic)
    if target == "mxfp8":
        return QuarkW8A8Mxfp8MoEMethod(moe_config, kMxfp8Static, kMxfp8Dynamic)
    if target == "fp8_per_block":
        return Fp8MoEMethod(Fp8Config(weight_block_size=[128, 128]), layer)
    if target == "fp8_per_channel":
        return QuarkW8A8Fp8MoEMethod(
            moe_config, kFp8StaticChannelSym, kFp8DynamicTokenSym
        )
    if target == "int8":
        return QuarkW8A8Int8MoEMethod(
            moe_config, kInt8StaticChannelSym, kInt8DynamicTokenSym
        )
    if target == "nvfp4_per_token":
        config = ModelOptNvFp4Config(is_checkpoint_nvfp4_serialized=True)
        config.moe_activation_override = kNvfp4DynamicToken
        return ModelOptNvFp4FusedMoE(config, moe_config)
    return Fp8MoEMethod(Fp8Config(), layer)


def _make_moe_layer(
    prefix: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    backend: MoEBackend,
    num_gate_up_projections: int = 2,
) -> RoutedExperts:
    gated = num_gate_up_projections == 2
    layer = FusedMoEFactory(
        num_experts=num_experts,
        top_k=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        activation="silu" if gated else "silu_no_mul",
        ckpt_names=("gate_proj", "down_proj", "up_proj" if gated else None),
        prefix=f"{prefix}_{intermediate_size}.experts",
    ).routed_experts
    layer.moe_config.moe_backend = backend
    return layer


def _pad_expert_reference(
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    intermediate_size: int,
    hidden_padding: int,
    intermediate_padding: int,
    value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Pad gate/up separately so the up projection lands after the new gate rows.
    return (
        torch.nn.functional.pad(
            gate_up_weights.unflatten(1, (-1, intermediate_size)),
            (0, hidden_padding, 0, intermediate_padding),
            value=value,
        ).flatten(1, 2),
        torch.nn.functional.pad(
            down_weights, (0, intermediate_padding, 0, hidden_padding), value=value
        ),
    )


def _check_moe_padding(
    method: OnlineMoEMethodBase,
    layer: RoutedExperts,
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    source_method = method.requantization_source
    assert source_method is not None
    dequantize = source_method.dequantize_weight
    source_chunks: list[tuple[slice, torch.Tensor, torch.Tensor]] = []

    def record_source_chunks(
        layer: RoutedExperts,
    ) -> Iterator[tuple[slice, torch.Tensor, torch.Tensor]]:
        chunks = dequantize(layer)
        assert not isinstance(chunks, torch.Tensor)
        for chunk in chunks:
            source_chunks.append(chunk)
            yield chunk

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(source_method, "dequantize_weight", record_source_chunks)
        gate_up_weights, down_weights = method.get_weights_for_quantization(layer)
    torch.testing.assert_close(
        (gate_up_weights, down_weights), expected, rtol=0, atol=0
    )

    # For one unpadded chunk, reuse the dequantized tensors without a copy.
    ((_, source_gate_up, source_down),) = source_chunks
    assert gate_up_weights is source_gate_up and down_weights is source_down

    moe_config = layer.moe_config
    hidden_size = moe_config.hidden_dim
    intermediate_size = moe_config.intermediate_size_per_partition
    for hidden_padding, intermediate_padding, bias, error, match in (
        (32, 32, None, None, ""),
        (32, 32, "w13_bias", NotImplementedError, "biases is unsupported"),
        (32, 32, "w2_bias", NotImplementedError, "biases is unsupported"),
        (-32, 0, None, ValueError, "cannot shrink weights"),
        (0, -32, None, ValueError, "cannot shrink weights"),
    ):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(moe_config, "hidden_dim", hidden_size + hidden_padding)
            patch.setattr(
                moe_config,
                "intermediate_size_per_partition",
                intermediate_size + intermediate_padding,
            )
            if bias is not None:
                patch.setattr(moe_config, "has_bias", True)
                weight = gate_up_weights if bias == "w13_bias" else down_weights
                patch.setattr(
                    layer,
                    bias,
                    torch.nn.Parameter(weight.new_zeros(weight.shape[:2])),
                    raising=False,
                )
            if error is not None:
                # Reject invalid layouts instead of silently dropping weights or biases.
                with pytest.raises(error, match=match):
                    method.get_weights_for_quantization(layer)
                continue
            torch.testing.assert_close(
                method.get_weights_for_quantization(layer),
                _pad_expert_reference(
                    *expected, intermediate_size, hidden_padding, intermediate_padding
                ),
                rtol=0,
                atol=0,
            )


@dataclass(frozen=True)
class _OnlineMoERank:
    layer: RoutedExperts
    method: OnlineMoEMethodBase
    hidden_size: int
    rank: int
    num_experts: int
    local_experts: range
    intermediate_size: int
    num_gate_up_projections: int

    def shard(
        self,
        tensor: torch.Tensor,
        projection: str,
        channel_scale: bool = False,
        shard_rank: int | None = None,
    ) -> torch.Tensor:
        local = tensor[self.local_experts.start : self.local_experts.stop]
        tp_size = self.layer.moe_config.tp_size
        if tp_size == 1:
            return local.contiguous()

        rank = self.rank if shard_rank is None else shard_rank
        if projection == "w13" and local.ndim == 3:
            # Split gate and up separately before restoring their packed order.
            gate_up = local.unflatten(1, (self.num_gate_up_projections, -1))
            return gate_up.chunk(tp_size, dim=2)[rank].flatten(1, 2)
        if projection == "w2" and local.ndim == 3 and not channel_scale:
            return local.chunk(tp_size, dim=-1)[rank].contiguous()
        return local


@dataclass(frozen=True)
class _MoECheckpoint:
    tensors: dict[str, torch.Tensor]
    reference: tuple[torch.Tensor, torch.Tensor]


def _local_expert_ids(layer: RoutedExperts, rank: int, world_size: int) -> range:
    moe_config = layer.moe_config
    assert (
        moe_config.use_ep
        == get_current_vllm_config().parallel_config.enable_expert_parallel
    )
    num_experts = moe_config.num_experts

    # EP owns whole experts; TP owns one weight shard of every expert.
    if not moe_config.use_ep:
        assert moe_config.tp_size == world_size
        assert layer.local_num_experts == num_experts
        return range(num_experts)

    assert moe_config.tp_size == 1 and moe_config.ep_size == world_size
    count = num_experts // world_size
    assert layer.local_num_experts == count
    first = rank * count
    expert_map = layer.expert_map
    assert expert_map is not None
    torch.testing.assert_close(
        expert_map[first : first + count],
        torch.arange(count, device=expert_map.device, dtype=expert_map.dtype),
    )
    return range(first, first + count)


def _create_online_moe(
    source: str,
    target: str,
    rank: int,
    world_size: int,
    case: str,
    intermediate_size: int = 256,
) -> _OnlineMoERank:
    check_padding = case.startswith("padding-")
    poison_padding = case.startswith("nan-padding")
    num_experts, hidden_size = (2, 128) if check_padding else (4, 256)
    if poison_padding:
        hidden_size = 240  # Deliberately not MXFP4-block aligned.
    num_gate_up_projections = 1 if case == "padding-ungated" else 2
    backends: dict[str, MoEBackend] = {
        "mxfp4": "aiter" if on_gfx950() else "emulation",
        "mxfp8": "auto",
        "lut_b": "emulation",
        "nvfp4_per_token": "auto",
    }
    backend = backends.get(target, "triton")
    if poison_padding:
        backend = "aiter" if case == "nan-padding-aiter" else "emulation"
    online_layer = _make_moe_layer(
        "online",
        num_experts,
        hidden_size,
        intermediate_size,
        backend,
        num_gate_up_projections,
    )
    moe_config = online_layer.moe_config
    local_experts = _local_expert_ids(online_layer, rank, world_size)
    source_factory = MOE_SOURCE_QUANT_METHODS[source]
    with skip_checkpoint_backend_selection():
        source_method = (
            source_factory(online_layer) if source_factory is not None else None
        )
    # Padding is checked before backend packing. In particular, an ungated
    # layout need not have a working GEMM backend to exercise padding.
    with skip_checkpoint_backend_selection() if check_padding else nullcontext():
        online_method = MOE_ONLINE_QUANT_METHODS[target](moe=moe_config)
    if source_method is not None:
        online_method.set_requantization_source(source_method)

    if not check_padding:
        # Replace the factory's unquantized method and its placeholder weights
        # so the real loader sees the online method and its checkpoint parameters.
        del online_layer.quant_method
        del online_layer.w13_weight
        del online_layer.w2_weight
        online_layer._replace_quant_method(online_method)

    storage_hidden_size = hidden_size
    storage_intermediate_size = intermediate_size // moe_config.tp_size
    if poison_padding:
        storage_hidden_size, storage_intermediate_size = _round_moe_storage(
            online_layer, online_method, storage_hidden_size, storage_intermediate_size
        )
    online_method.create_weights(
        layer=online_layer,
        num_experts=len(local_experts),
        hidden_size=storage_hidden_size,
        intermediate_size_per_partition=storage_intermediate_size,
        params_dtype=torch.bfloat16,
        global_num_experts=num_experts,
        weight_loader=online_layer.weight_loader,
    )
    if source_method is not None or not check_padding:
        _materialize_requantization_source(online_layer)
    return _OnlineMoERank(
        layer=online_layer,
        method=online_method,
        hidden_size=hidden_size,
        rank=rank,
        num_experts=num_experts,
        local_experts=local_experts,
        intermediate_size=intermediate_size,
        num_gate_up_projections=num_gate_up_projections,
    )


def _make_moe_checkpoint(
    source: str, online_rank: _OnlineMoERank, vary_block_scales: bool = False
) -> _MoECheckpoint:
    num_experts = online_rank.num_experts
    hidden_size = online_rank.hidden_size
    intermediate_size = online_rank.intermediate_size
    num_projections = online_rank.num_gate_up_projections
    source_format, _, scheme = source.partition("/")
    block_size = [128, 128]
    if source == "fp8/block" and intermediate_size % 128 != 0:
        # The partial-block case stages the refined scales directly.
        source_method = online_rank.method.requantization_source
        assert isinstance(source_method, Fp8MoEMethod)
        assert source_method.scale_block_size is not None
        block_size = source_method.scale_block_size

    # Build the same global checkpoint on every rank, independent of layer setup.
    torch.manual_seed(11)
    if source == "bf16":
        gate_up = torch.randn(
            num_experts,
            num_projections * intermediate_size,
            hidden_size,
            dtype=torch.bfloat16,
        )
        down = torch.randn(
            num_experts, hidden_size, intermediate_size, dtype=torch.bfloat16
        )
        if vary_block_scales:
            amplitudes = torch.tensor(
                [2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4], dtype=torch.bfloat16
            )
            for weight in (gate_up, down):
                blocks = torch.arange(weight.shape[-1]) // 32
                weight.mul_(amplitudes[blocks % len(amplitudes)])
        return _MoECheckpoint(
            {"w13_weight": gate_up, "w2_weight": down}, (gate_up, down)
        )
    if scheme == "nvfp4":
        gate = torch.randn(num_experts, intermediate_size, hidden_size)
        up = torch.randn_like(gate) if num_projections == 2 else None
        down = torch.randn(num_experts, hidden_size, intermediate_size)
        global_scales = torch.arange(num_experts, dtype=torch.float32) + 97
        tensors, reference_gate_up, reference_down = _nvfp4_moe_checkpoint_tensors(
            source_format,
            gate,
            up,
            down,
            global_scales,
            2 * global_scales,
            3 * global_scales,
        )
        return _MoECheckpoint(
            tensors,
            (reference_gate_up.to(torch.bfloat16), reference_down.to(torch.bfloat16)),
        )

    tensors, references = _fp8_moe_checkpoint_tensors(
        source_format,
        scheme,
        num_experts,
        hidden_size,
        intermediate_size,
        block_size,
        num_projections,
    )
    reference_gate_up, reference_down = references
    return _MoECheckpoint(
        tensors,
        (reference_gate_up.to(torch.bfloat16), reference_down.to(torch.bfloat16)),
    )


def _named_moe_checkpoint(
    checkpoint: _MoECheckpoint, source: str, online_rank: _OnlineMoERank
) -> Callable[[Sequence[int]], Iterator[tuple[str, torch.Tensor]]]:
    tensors = checkpoint.tensors
    if source == "quark/channel":
        # The checkpoint stores vectors; the reference uses broadcast scales.
        tensors = tensors.copy()
        for name in ("w13_weight_scale", "w2_weight_scale"):
            tensors[name] = tensors[name].squeeze(-1)
    return partial(
        named_expert_checkpoint_entries,
        tensors,
        gated=online_rank.num_gate_up_projections == 2,
    )


def _copy_checkpoint_shard(
    online_rank: _OnlineMoERank, checkpoint: _MoECheckpoint, source: str
) -> None:
    for name, tensor in checkpoint.tensors.items():
        parameter = getattr(online_rank.layer, name)
        local = online_rank.shard(
            tensor,
            name.partition("_")[0],
            channel_scale=source.endswith("/channel") and "scale" in name,
        )
        parameter.data.copy_(local.reshape_as(parameter))


def _reference_shard(
    online_rank: _OnlineMoERank, checkpoint: _MoECheckpoint
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up, down = checkpoint.reference
    return online_rank.shard(gate_up, "w13"), online_rank.shard(down, "w2")


def _moe_weight_maxima(
    online_rank: _OnlineMoERank, checkpoint: _MoECheckpoint, target: str
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if online_rank.layer.moe_config.tp_size != 2:
        return
    for projection, weight in zip(("w13", "w2"), checkpoint.reference, strict=True):
        if target in ("fp8_per_tensor", "nvfp4_per_token"):
            dimensions = (1, 2)
        elif projection == "w2" and target in ("fp8_per_channel", "int8"):
            # Channel scales only cross TP shards for the down projection.
            dimensions = (-1,)
        else:
            continue

        local = online_rank.shard(weight, projection)
        peer = online_rank.shard(weight, projection, shard_rank=1 - online_rank.rank)
        keepdim = target == "fp8_per_channel"
        yield (
            local.abs().amax(dim=dimensions, keepdim=keepdim),
            peer.abs().amax(dim=dimensions, keepdim=keepdim),
        )


def _process_online_moe(
    online_rank: _OnlineMoERank, checkpoint: _MoECheckpoint, target: str
) -> None:
    maxima = _moe_weight_maxima(online_rank, checkpoint, target)
    with _emulate_max_reductions(parallel_state.get_ep_group().device_group, maxima):
        online_rank.method.process_weights_after_loading(online_rank.layer)


def _prepare_offline_moe(
    online_rank: _OnlineMoERank,
    checkpoint: _MoECheckpoint,
    target: str,
    pack_reference: _ReferencePacker,
    padded_reference: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[RoutedExperts, FusedMoEMethodBase]:
    offline_layer = _make_moe_layer(
        "offline",
        online_rank.num_experts,
        online_rank.hidden_size,
        online_rank.intermediate_size,
        online_rank.layer.moe_config.moe_backend,
        online_rank.num_gate_up_projections,
    )
    offline_method = _moe_offline_method(target, offline_layer)
    if target == "mxfp4":
        # Compare packed bytes only after both paths select the same layout.
        assert isinstance(online_rank.method, Mxfp4OnlineMoEMethod)
        assert isinstance(offline_method, QuarkOCP_MX_MoEMethod)
        assert online_rank.method.mxfp4_backend == offline_method.mxfp4_backend
    hidden_size = online_rank.hidden_size
    local_intermediate_size = (
        online_rank.intermediate_size // offline_layer.moe_config.tp_size
    )
    if padded_reference is not None:
        hidden_size, local_intermediate_size = _round_moe_storage(
            offline_layer, offline_method, hidden_size, local_intermediate_size
        )
        online_config = online_rank.layer.moe_config
        assert hidden_size == online_config.hidden_dim
        assert local_intermediate_size == online_config.intermediate_size_per_partition
    offline_method.create_weights(
        layer=offline_layer,
        num_experts=len(online_rank.local_experts),
        hidden_size=hidden_size,
        intermediate_size_per_partition=local_intermediate_size,
        params_dtype=torch.bfloat16,
        global_num_experts=online_rank.num_experts,
    )

    def local_shard(
        tensor: torch.Tensor, projection: str, channel_scale: bool = False
    ) -> torch.Tensor:
        if padded_reference is not None:
            return tensor  # These weights already include this rank's padding.
        return online_rank.shard(tensor, projection, channel_scale=channel_scale)

    reference = checkpoint.reference if padded_reference is None else padded_reference
    for projection, weight in zip(("w13", "w2"), reference, strict=True):
        packed, scales, global_scale = (
            pack_reference(projection, weight)
            if padded_reference is None
            else _pack_moe_reference(weight, target)
        )
        assert scales is not None
        if global_scale is not None:
            if projection == "w13":
                global_scale = global_scale[:, None].expand(-1, 2)
            getattr(offline_layer, f"{projection}_weight_scale_2").data.copy_(
                local_shard(global_scale, projection)
            )
        if target == "fp8_per_tensor" and projection == "w13":
            scales = scales[:, None].expand(-1, 2)

        getattr(offline_layer, f"{projection}_weight").data.copy_(
            local_shard(packed, projection)
        )
        scale_name = "weight_scale_inv" if target == "fp8_per_block" else "weight_scale"
        scale_parameter = getattr(offline_layer, f"{projection}_{scale_name}")
        local_scales = local_shard(
            scales, projection, channel_scale=target in ("fp8_per_channel", "int8")
        )
        scale_parameter.data.copy_(local_scales.reshape_as(scale_parameter))
    return offline_layer, offline_method


def _round_moe_storage(
    layer: RoutedExperts,
    method: FusedMoEMethodBase,
    hidden_size: int,
    local_intermediate_size: int,
) -> tuple[int, int]:
    moe_config = layer.moe_config
    # Retain the logical sizes so quantization can identify unloaded cells.
    assert moe_config.hidden_dim_unpadded == hidden_size
    assert (
        moe_config.intermediate_size_per_partition_unpadded == local_intermediate_size
    )
    storage_sizes = method.maybe_roundup_sizes(
        hidden_size=hidden_size,
        intermediate_size_per_partition=local_intermediate_size,
        act_dtype=torch.bfloat16,
        moe_parallel_config=moe_config.moe_parallel_config,
    )
    moe_config.hidden_dim, moe_config.intermediate_size_per_partition = storage_sizes
    return storage_sizes


def _pad_local_moe_weights(
    online_rank: _OnlineMoERank, checkpoint: _MoECheckpoint, value: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_up, down = _reference_shard(online_rank, checkpoint)
    moe_config = online_rank.layer.moe_config
    hidden_padding = moe_config.hidden_dim - down.shape[1]
    intermediate_padding = moe_config.intermediate_size_per_partition - down.shape[2]
    return _pad_expert_reference(
        gate_up, down, down.shape[2], hidden_padding, intermediate_padding, value=value
    )
