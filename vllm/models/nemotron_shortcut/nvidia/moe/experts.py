# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron Shortcut routed experts and their quantization methods.

The experts are non-gated: `relu2(c * tanh(x / c))` between the two
projections, with `c` the checkpoint's `activation_clamp_value`. The MoE
plumbing sees the plain `RELU2_NO_MUL` activation; the clamp is carried on
`NemotronShortcutMoEConfig` and only the experts classes in this package read it.
`NemotronShortcutRoutedExperts` therefore selects every experts implementation
itself and never goes through the generic MoE oracles, which would pick a
backend that drops the clamp.
"""

import math
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptMixedPrecisionConfig,
    ModelOptMxFp8Config,
    ModelOptMxFp8FusedMoE,
    ModelOptNvFp4Config,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.models.nemotron_shortcut.nvidia.moe.config import (
    NemotronShortcutMoEConfig,
    activation_clamp,
)
from vllm.models.nemotron_shortcut.nvidia.moe.flashinfer import (
    NemotronShortcutCuteDslMxfp8Experts,
    NemotronShortcutCuteDslNvFp4Experts,
    NemotronShortcutFlashInferExperts,
)
from vllm.models.nemotron_shortcut.nvidia.moe.kernels import routed_experts
from vllm.platforms import current_platform

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )


@dataclass
class DispatchedTokens:
    """The tokens this rank's experts run, as the dispatch returns them."""

    hidden_states: torch.Tensor
    scale: torch.Tensor | None
    tokens_meta: mk.ExpertTokensMetadata | None
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    # The experts' output dtype and the combined output's shape on this rank.
    output_dtype: torch.dtype
    output_shape: tuple[int, ...]


def dispatch_tokens(
    kernel: mk.FusedMoEKernel,
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> DispatchedTokens:
    """`kernel`'s prepare: quantize `x` if the experts want it, then dispatch."""
    experts = kernel.fused_experts
    prepare_finalize = kernel.prepare_finalize
    assert isinstance(experts, mk.FusedMoEExpertsModular)
    assert isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular)
    a1q, a1q_scale, tokens_meta, dispatched_ids, dispatched_weights = (
        prepare_finalize.prepare(
            x,
            topk_weights,
            topk_ids,
            layer.global_num_experts,
            layer.expert_map,
            False,
            experts.quant_config,
            defer_input_quant=experts.expects_unquantized_inputs,
        )
    )
    return DispatchedTokens(
        a1q,
        a1q_scale,
        tokens_meta,
        topk_weights if dispatched_weights is None else dispatched_weights,
        topk_ids if dispatched_ids is None else dispatched_ids,
        experts.workspace_dtype(x.dtype),
        (*x.shape[:-1], experts.output_hidden_dim(x.size(-1))),
    )


def run_experts(
    kernel: mk.FusedMoEKernel,
    layer: RoutedExperts,
    tokens: DispatchedTokens,
    activation: MoEActivation,
) -> torch.Tensor:
    """The experts' output for the dispatched tokens, before the combine."""
    experts = kernel.fused_experts
    assert isinstance(experts, mk.FusedMoEExpertsModular)
    a1q, dtype = tokens.hidden_states, tokens.output_dtype
    w1, w2 = layer.w13_weight, layer.w2_weight
    _, num_tokens, n, k, top_k = experts.moe_problem_size(a1q, w1, w2, tokens.topk_ids)
    workspace13_shape, workspace2_shape, fused_out_shape = experts.workspace_shapes(
        num_tokens,
        n,
        k,
        top_k,
        layer.global_num_experts,
        w1.shape[0],
        tokens.tokens_meta,
        activation,
    )
    # The combine input the transport would like, as FusedMoEKernel picks it.
    fused_out = kernel.prepare_finalize.allocate_fused_expert_output(
        fused_out_shape, dtype, a1q.device, experts.finalize_weight_and_reduce_impl()
    )
    if fused_out is None:
        fused_out = a1q.new_empty(fused_out_shape, dtype=dtype)
    if num_tokens > 0:
        unfinalized = experts.apply(
            output=fused_out,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=tokens.topk_weights,
            topk_ids=tokens.topk_ids,
            activation=activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            a1q_scale=tokens.scale,
            a2_scale=experts.a2_scale,
            workspace13=a1q.new_empty(workspace13_shape, dtype=dtype),
            workspace2=a1q.new_empty(workspace2_shape, dtype=dtype),
            expert_tokens_meta=tokens.tokens_meta,
            apply_router_weight_on_input=False,
        )
        # Experts that leave the top-k reduction open write nothing to `fused_out`.
        if isinstance(unfinalized, UnfinalizedMoEOutput):
            raise NotImplementedError("staged MoE needs a finalized output")
    return fused_out


def combine_tokens(
    kernel: mk.FusedMoEKernel,
    tokens: DispatchedTokens,
    fused_out: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """`kernel`'s finalize: reduce `fused_out` into `output` on this rank."""
    experts = kernel.fused_experts
    assert isinstance(experts, mk.FusedMoEExpertsModular)
    prepare_finalize = kernel.prepare_finalize
    assert isinstance(prepare_finalize, mk.FusedMoEPrepareAndFinalizeModular)
    prepare_finalize.finalize(
        output,
        fused_out,
        tokens.topk_weights,
        tokens.topk_ids,
        False,
        experts.finalize_weight_and_reduce_impl(),
    )


def apply_moe_kernel(
    kernel: mk.FusedMoEKernel,
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: MoEActivation,
    shared_experts: "SharedExperts | None" = None,
    shared_experts_input: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a modular MoE kernel, keeping the experts' accumulation dtype.

    `FusedMoEKernel.apply` returns the finalized output in the activation
    dtype. Experts whose workspace dtype is wider (the FP32 top-k sum of the
    Triton experts) keep it here, so the runner reduces EP and TP partial sums
    before anything rounds them.
    """
    experts = kernel.fused_experts
    assert isinstance(experts, mk.FusedMoEExpertsModular)
    if experts.workspace_dtype(x.dtype) == x.dtype:
        result = kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=False,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
        assert isinstance(result, torch.Tensor)
        return result

    if shared_experts is not None or kernel.prepare_finalize.supports_async():
        raise NotImplementedError(
            "Nemotron Shortcut's FP32 expert output needs a synchronous dispatcher "
            "and shared experts run by the caller"
        )
    tokens = dispatch_tokens(kernel, layer, x, topk_weights, topk_ids)
    fused_out = run_experts(kernel, layer, tokens, activation)
    output = x.new_empty(tokens.output_shape, dtype=tokens.output_dtype)
    combine_tokens(kernel, tokens, fused_out, output)
    return output


class NemotronShortcutTritonExperts(mk.FusedMoEExpertsModular):
    """The reference experts: `kernels.routed_experts` on BF16 weights.

    Each EP rank runs its contiguous expert interval on every token, so the
    tokens must be replicated across EP ranks (no dispatcher, or the
    all-gather / reduce-scatter one).
    """

    def __init__(
        self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.clamp = activation_clamp(moe_config)
        parallel = moe_config.moe_parallel_config
        self.use_ep = parallel.use_ep
        # The linear expert placement: the first `remainder` ranks hold one more.
        base, remainder = divmod(moe_config.num_experts, parallel.ep_size)
        self.expert_begin = parallel.ep_rank * base + min(parallel.ep_rank, remainder)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda_alike()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return weight_key is None and activation_key is None

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.RELU2_NO_MUL

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_ag_rs_all2all_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # The kernels weight and sum the routes themselves.
        return TopKWeightAndReduceNoOP()

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return torch.float32

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        assert not apply_router_weight_on_input
        # The kernels take global expert ids and the local interval; vLLM's
        # expert_map only exists with EP, where it is the same interval.
        assert self.use_ep or expert_map is None
        output.copy_(
            routed_experts(
                hidden_states,
                w1,
                w2,
                topk_ids,
                topk_weights,
                self.expert_begin,
                self.clamp,
                self.quant_config.w1_scale,
                self.quant_config.w2_scale,
            )
        )


class NemotronShortcutMxfp8TritonExperts(NemotronShortcutTritonExperts):
    """The reference experts on MXFP8 weights (Triton `dot_scaled`).

    Activations are quantized in-kernel at the input of each projection;
    weights are E4M3 with E8M0 block-32 scales, as in the checkpoint.
    """

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_cuda() and current_platform.has_device_capability(
            100
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp8Static, kMxfp8Dynamic)


def _make_kernel(moe: FusedMoEConfig, experts: mk.FusedMoEExperts) -> mk.FusedMoEKernel:
    prepare_finalize = maybe_make_prepare_finalize(moe, None, allow_new_interface=True)
    assert prepare_finalize is not None
    return mk.FusedMoEKernel(prepare_finalize, experts)


class NemotronShortcutMoEMethod(FusedMoEMethodBase):
    """Unquantized experts.

    FlashInfer's CUTLASS fused MoE where it implements the clamp, else the
    Triton reference experts; `--moe-backend triton` forces the latter.
    """

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.moe_quant_config = FusedMoEQuantConfig.make()
        experts_cls: type[mk.FusedMoEExperts] = NemotronShortcutTritonExperts
        if (
            moe.moe_backend != "triton"
            and moe.in_dtype == torch.bfloat16
            and NemotronShortcutFlashInferExperts.is_supported_config(
                NemotronShortcutFlashInferExperts,
                moe,
                None,
                None,
                mk.FusedMoEActivationFormat.Standard,
            )[0]
        ):
            experts_cls = NemotronShortcutFlashInferExperts
        elif moe.moe_backend != "triton":
            logger.info_once(
                "Nemotron Shortcut BF16 experts: FlashInfer's CUTLASS MoE with the "
                "clamp is not available here; using the Triton experts"
            )
        self.moe_kernel = _make_kernel(moe, experts_cls(moe, self.moe_quant_config))

    create_weights = UnquantizedFusedMoEMethod.create_weights

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return self.moe_quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert self.moe_kernel is not None
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: "SharedExperts | None",
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return apply_moe_kernel(
            self.moe_kernel,
            layer,
            x,
            topk_weights,
            topk_ids,
            self.moe.activation,
            shared_experts,
            shared_experts_input,
        )


def select_mxfp8_experts(moe: FusedMoEConfig) -> type[mk.FusedMoEExperts]:
    """CuTe-DSL MXFP8 experts where supported, else the Triton reference.

    Only these two implement the clamp, so any other `--moe-backend` errors
    instead of silently running a plain ReLU².
    """
    backend = moe.moe_backend
    if backend in ("auto", "flashinfer_cutedsl"):
        supported, reason = NemotronShortcutCuteDslMxfp8Experts.is_supported_config(
            NemotronShortcutCuteDslMxfp8Experts,
            moe,
            kMxfp8Static,
            kMxfp8Dynamic,
            mk.FusedMoEActivationFormat.Standard,
        )
        if supported:
            return NemotronShortcutCuteDslMxfp8Experts
        if backend == "flashinfer_cutedsl":
            raise ValueError(f"CuTe-DSL MXFP8 experts are unavailable: {reason}")
    if backend in ("auto", "triton"):
        supported, reason = NemotronShortcutMxfp8TritonExperts.is_supported_config(
            NemotronShortcutMxfp8TritonExperts,
            moe,
            kMxfp8Static,
            kMxfp8Dynamic,
            mk.FusedMoEActivationFormat.Standard,
        )
        if supported:
            return NemotronShortcutMxfp8TritonExperts
        raise ValueError(f"Triton MXFP8 experts are unavailable: {reason}")
    raise ValueError(
        f"moe_backend={backend!r} does not implement Nemotron Shortcut's clamped ReLU² "
        "for MXFP8 experts; use auto, triton or flashinfer_cutedsl"
    )


class NemotronShortcutModelOptMxfp8MoEMethod(FusedMoEMethodBase):
    """ModelOpt MXFP8 experts.

    RoutedExperts.weight_loader picks the ModelOpt loading rules by the quant
    method's class name, so the name must contain "ModelOpt".
    """

    def __init__(self, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.moe_quant_config = FusedMoEQuantConfig.make()
        experts_cls = select_mxfp8_experts(moe)
        self.moe_kernel = _make_kernel(moe, experts_cls(moe, self.moe_quant_config))

    def create_weights(self, layer: RoutedExperts, *args, **kwargs) -> None:
        # The parameters are ModelOpt's MXFP8 ones; only the experts differ.
        ModelOptMxFp8FusedMoE.create_weights(
            cast(ModelOptMxFp8FusedMoE, self), layer, *args, **kwargs
        )
        self.moe_quant_config = FusedMoEQuantConfig.make(
            quant_dtype="mxfp8",
            block_shape=[1, MXFP8_BLOCK_SIZE],
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            is_scale_swizzled=False,
        )
        assert self.moe_kernel is not None
        self.moe_kernel.fused_experts.quant_config = self.moe_quant_config

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return self.moe_quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Backends that re-lay out the loaded scales (CuTe-DSL) do it here.
        assert self.moe_kernel is not None
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    apply = NemotronShortcutMoEMethod.apply


class NemotronShortcutModelOptNvFp4MoEMethod(ModelOptNvFp4FusedMoE):
    """ModelOpt NVFP4 experts on FlashInfer's CuTe-DSL fused MoE.

    The routing weight is applied before the down projection, as in training.
    RoutedExperts.weight_loader picks the ModelOpt loading rules by the quant
    method's class name, so the name must contain "ModelOpt".
    """

    def __init__(self, quant_config: ModelOptNvFp4Config, moe: FusedMoEConfig):
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            raise NotImplementedError("Nemotron Shortcut NVFP4 experts require SM10x")
        if moe.in_dtype != torch.bfloat16:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts require BF16 activations"
            )
        if quant_config.quant_method != "NVFP4":
            raise NotImplementedError("Nemotron Shortcut NVFP4 experts require W4A4")
        if getattr(quant_config, "group_size", None) != 16:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts require block-16"
            )
        if quant_config.moe_activation_override is not None:
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts do not support activation overrides"
            )
        if moe.moe_backend not in ("auto", "flashinfer_cutedsl"):
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts require moe_backend=flashinfer_cutedsl"
            )
        super().__init__(quant_config, replace(moe, moe_backend="flashinfer_cutedsl"))
        if self.nvfp4_backend != NvFp4MoeBackend.FLASHINFER_CUTEDSL:
            raise NotImplementedError(
                f"Nemotron Shortcut NVFP4 experts got the {self.nvfp4_backend} backend"
            )
        self.experts_cls = NemotronShortcutCuteDslNvFp4Experts

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Check every exported value before the shared-scale reduction can
        # hide a bad one.
        for scale in (
            layer.w13_input_scale,
            layer.w2_input_scale,
            layer.w13_weight_scale_2,
            layer.w2_weight_scale_2,
        ):
            if not bool((torch.isfinite(scale) & (scale > 0)).all()):
                raise ValueError("NVFP4 global scales must be positive and finite")
        super().process_weights_after_loading(layer)


def _resolve_mixed_precision(
    quant_config: ModelOptMixedPrecisionConfig, prefix: str
) -> QuantizationConfig | None:
    """The single per-layer config of a fused expert group."""
    candidates = quant_config._quantized_layer_prefix_candidates(prefix)
    # A fused group can only be excluded as a whole.
    if any(
        fnmatchcase(candidate, pattern) or pattern in candidate
        for candidate in candidates
        for pattern in quant_config.exclude_modules
    ):
        return None
    expert_prefixes = candidates + tuple(
        candidate.removesuffix(".experts") + "." + projection
        for candidate in candidates
        if candidate.endswith(".experts")
        for projection in ("up_proj", "down_proj")
    )
    entries = {
        key: info
        for key, info in quant_config.quantized_layers.items()
        if any(
            key == candidate or key.startswith(candidate + ".")
            for candidate in expert_prefixes
        )
    }
    algorithms = {info["quant_algo"].upper() for info in entries.values()}
    if len(algorithms) > 1:
        raise NotImplementedError(
            f"Expert group {prefix} must use one quantization algorithm, "
            f"got {sorted(algorithms)}"
        )
    if any(quant_config.is_layer_excluded(key) for key in entries):
        raise NotImplementedError(
            f"Expert group {prefix} cannot partially exclude experts"
        )
    algorithm = next(iter(algorithms), None)
    if algorithm is None:
        return None
    if algorithm == "NVFP4":
        if any(info.get("group_size", 16) != 16 for info in entries.values()):
            raise NotImplementedError(
                "Nemotron Shortcut NVFP4 experts require block-16"
            )
        return quant_config.nvfp4_config
    if algorithm == "MXFP8":
        return quant_config.mxfp8_config
    raise NotImplementedError(
        f"Nemotron Shortcut experts do not support ModelOpt algorithm {algorithm!r} "
        f"at {prefix}"
    )


class NemotronShortcutRoutedExperts(RoutedExperts):
    """Routed experts that select one of the clamp-aware implementations."""

    def __init__(
        self,
        layer_name: str,
        params_dtype: torch.dtype,
        moe_config: FusedMoEConfig,
        quant_config: QuantizationConfig | None,
        *,
        activation_clamp_value: float | None,
        **kwargs,
    ) -> None:
        if not isinstance(moe_config, NemotronShortcutMoEConfig):
            raise TypeError(
                "pass moe_config_cls=NemotronShortcutMoEConfig to the factory"
            )
        if activation_clamp_value is not None and not (
            math.isfinite(activation_clamp_value) and activation_clamp_value > 0
        ):
            raise ValueError(
                "activation_clamp_value must be positive and finite, got "
                f"{activation_clamp_value}"
            )
        # Set before RoutedExperts.__init__ selects the quantization method.
        moe_config.activation_clamp_value = activation_clamp_value
        super().__init__(layer_name, params_dtype, moe_config, quant_config, **kwargs)

    def _get_quant_method(
        self,
        prefix: str,
        quant_config: QuantizationConfig | None,
        moe_config: FusedMoEConfig,
    ) -> FusedMoEMethodBase:
        if isinstance(quant_config, ModelOptMixedPrecisionConfig):
            quant_config = _resolve_mixed_precision(quant_config, prefix)
        elif isinstance(
            quant_config, (ModelOptMxFp8Config, ModelOptNvFp4Config)
        ) and quant_config.is_layer_excluded(prefix):
            quant_config = None
        nvfp4 = isinstance(quant_config, ModelOptNvFp4Config)
        mxfp8 = isinstance(quant_config, ModelOptMxFp8Config)
        if quant_config is not None and not (nvfp4 or mxfp8):
            raise NotImplementedError(
                "Nemotron Shortcut quantized experts require ModelOpt MXFP8 or NVFP4"
            )
        if (nvfp4 or mxfp8) and not (
            current_platform.is_cuda() and current_platform.has_device_capability(100)
        ):
            raise NotImplementedError(
                "Nemotron Shortcut quantized experts require SM100 or newer"
            )
        parallel = moe_config.moe_parallel_config
        if parallel.use_ep and (
            parallel.pcp_size > 1
            or parallel.is_sequence_parallel
            or parallel.enable_eplb
            or self.expert_placement_strategy != "linear"
            or parallel.all2all_backend != "allgather_reducescatter"
        ):
            raise NotImplementedError(
                "Nemotron Shortcut expert parallelism needs linear placement, the "
                "allgather_reducescatter backend, and no PCP, SP or EPLB: "
                "every EP rank sees every token"
            )
        if nvfp4:
            return NemotronShortcutModelOptNvFp4MoEMethod(
                cast(ModelOptNvFp4Config, quant_config), moe_config
            )
        if mxfp8:
            return NemotronShortcutModelOptMxfp8MoEMethod(moe_config)
        return NemotronShortcutMoEMethod(moe_config)
