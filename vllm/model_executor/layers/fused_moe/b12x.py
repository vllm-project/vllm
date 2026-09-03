# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x modular tensor-parallel fused MoE backend."""

from collections.abc import Iterable
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
    kMxfp8Dynamic,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xWarmupUnit,
    get_b12x_fused_moe,
    reuse_packed_weight_storage,
)

logger = init_logger(__name__)

_B12X_MOE_MODES: dict[
    tuple[torch.dtype | str | None, torch.dtype | str | None],
    tuple[str, str, str],
] = {
    ("mxfp4", "mxfp8"): ("w4a8_mx", "fp4_e8m0_k32", "w31"),
    ("mxfp4", None): ("w4a16", "fp4_e8m0_k32", "w31"),
    ("nvfp4", "nvfp4"): ("nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", "mxfp8"): ("w4a8_nvfp4", "modelopt_nvfp4", "w31"),
    ("nvfp4", None): ("w4a16", "modelopt_nvfp4", "w13"),
}


def _require_b12x_fused_moe() -> Any:
    fused_moe = get_b12x_fused_moe()
    assert fused_moe is not None
    return fused_moe


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation == MoEActivation.SILU:
        return "silu"
    if activation in (MoEActivation.RELU2, MoEActivation.RELU2_NO_MUL):
        return "relu2"
    return activation.value


def _b12x_scratch_nbytes(plan: Any) -> int:
    specs = plan.scratch_specs()
    if len(specs) != 1:
        raise RuntimeError(f"expected one b12x MoE scratch buffer, got {len(specs)}")
    spec = specs[0]
    if spec.dtype != torch.uint8:
        raise TypeError(f"expected b12x MoE scratch dtype uint8, got {spec.dtype}")
    return int(spec.shape[0])


def _b12x_moe_execution_plan(
    *,
    tokens: int,
    topk: int,
    prepared: Any,
    quant_mode: str,
    apply_router_weight_on_input: bool,
    swiglu_limit: float | None,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
) -> Any:
    fused_moe = _require_b12x_fused_moe()

    return fused_moe.plan_execution(
        num_tokens=max(int(tokens), 1),
        num_topk=int(topk),
        device=prepared.w1_fp4.device,
        weight_plan=prepared.plan,
        quant_mode=quant_mode,
        apply_router_weight_on_input=apply_router_weight_on_input,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


def _run_b12x_moe_plan(
    *,
    plan: Any,
    scratch: torch.Tensor,
    hidden_states: torch.Tensor,
    prepared: Any,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    unit_scale_contract: bool,
) -> None:
    fused_moe = _require_b12x_fused_moe()

    binding = fused_moe.bind(
        plan,
        scratch=scratch,
        a=hidden_states,
        experts=prepared,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        output=output,
        input_scales_static=True,
        unit_scale_contract=unit_scale_contract,
    )
    fused_moe.run(binding=binding)


def _is_current_stream_capturing() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _normalize_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if topk_ids.dtype == torch.int32 and topk_ids.is_contiguous():
        return topk_ids
    if _is_current_stream_capturing():
        raise RuntimeError(
            "b12x MoE topk_ids normalization would allocate during CUDA capture"
        )
    return topk_ids.to(dtype=torch.int32).contiguous()


def _normalize_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype == torch.float32 and topk_weights.is_contiguous():
        return topk_weights
    if _is_current_stream_capturing():
        raise RuntimeError(
            "b12x MoE topk_weights normalization would allocate during CUDA capture"
        )
    return topk_weights.to(dtype=torch.float32).contiguous()


def _workspace_as_b12x_scratch(
    workspace: torch.Tensor | None,
    plan: Any,
) -> torch.Tensor:
    if workspace is None:
        raise RuntimeError("b12x MoE requires workspace2 scratch")
    if not workspace.is_contiguous():
        raise ValueError("b12x MoE workspace2 must be contiguous")
    scratch = workspace.view(-1).view(torch.uint8)
    required_nbytes = _b12x_scratch_nbytes(plan)
    if scratch.numel() < required_nbytes:
        raise ValueError(
            "b12x MoE workspace2 is too small: "
            f"have={scratch.numel()} bytes, need={required_nbytes} bytes"
        )
    return scratch


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    name: str,
) -> torch.Tensor | None:
    parameter = getattr(layer, name, None)
    if not isinstance(parameter, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=parameter.dtype, device=parameter.device)
    replace_parameter(layer, name, empty)
    return getattr(layer, name)


def _normalize_expert_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2:
        if scale.shape[1] not in (1, 2):
            raise ValueError(
                "expected an expert scale with one or two columns, got "
                f"{tuple(scale.shape)}"
            )
        scale = scale[:, 0]
    return scale.to(dtype=torch.float32).contiguous()


def _canonicalize_fp4_zero_signs_(packed: torch.Tensor) -> None:
    """Clear sign bits from packed FP4 zero values in place."""
    packed = packed.view(torch.uint8)
    magnitude = packed & 0x77
    nonzero = (magnitude | (magnitude >> 1) | (magnitude >> 2)) & 0x11
    packed.bitwise_and_(0x77 | (nonzero << 3))


class B12xExperts(mk.FusedMoEExpertsModular):
    """FP4 MoE experts backed by the b12x SM12x planned API."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)
        if quant_config.weight_quant_dtype not in ("mxfp4", "nvfp4"):
            raise ValueError(
                "b12x MoE requires MXFP4 or NVFP4 weights, got "
                f"{quant_config.weight_quant_dtype}"
            )
        scheme = (
            quant_config.weight_quant_dtype,
            quant_config.quant_dtype,
        )
        try:
            self._quant_mode, self._source_format, self._w13_layout = _B12X_MOE_MODES[
                scheme
            ]
        except KeyError as exc:
            raise ValueError(
                f"unsupported b12x MoE quantization scheme {scheme}"
            ) from exc
        self._prepared_experts: Any | None = None
        self._source_parameters_released = False
        self._unit_scales: dict[torch.device, torch.Tensor] = {}
        self._plans: dict[tuple[int, int, MoEActivation, bool], Any] = {}
        self._apply_router_weight_on_input = False

    def _unit_scale(self, device: torch.device, num_experts: int) -> torch.Tensor:
        scale = self._unit_scales.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scales[device] = scale
        return scale

    def _weight_global_scale(
        self,
        device: torch.device,
        num_experts: int,
        scale: torch.Tensor | None,
        name: str,
    ) -> torch.Tensor:
        if self._source_format != "modelopt_nvfp4":
            return self._unit_scale(device, num_experts)
        if scale is None:
            raise ValueError(f"b12x NVFP4 MoE requires {name}")
        scale = _normalize_expert_scale(scale)
        if scale.numel() != num_experts:
            raise ValueError(
                f"b12x NVFP4 MoE expected {num_experts} {name} values, "
                f"got {scale.numel()}"
            )
        return scale.to(device=device)

    def _swiglu_params(
        self,
        activation: MoEActivation,
    ) -> tuple[float | None, float | None, float | None]:
        if activation in (
            MoEActivation.SITU,
            MoEActivation.RELU2,
            MoEActivation.RELU2_NO_MUL,
        ):
            return None, None, None

        limit = self.quant_config.gemm1_clamp_limit
        if limit is None:
            limit = self.moe_config.swiglu_limit
        if activation != MoEActivation.SWIGLUOAI_UNINTERLEAVE:
            return limit, None, None

        alpha = self.quant_config.gemm1_alpha
        if alpha is None:
            alpha = self.moe_config.swiglu_alpha
        beta = self.quant_config.gemm1_beta
        if beta is None:
            beta = self.moe_config.swiglu_beta
        return limit, alpha, beta

    def _prepare_experts(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ) -> Any:
        quant_mode = self._quant_mode
        if _is_current_stream_capturing():
            raise RuntimeError(
                "b12x MoE weights must be prepared before CUDA graph capture"
            )
        if self.w1_scale is None or self.w2_scale is None:
            raise ValueError("b12x MoE requires w1 and w2 block scales")

        _canonicalize_fp4_zero_signs_(w1)
        _canonicalize_fp4_zero_signs_(w2)

        fused_moe = _require_b12x_fused_moe()

        num_experts = int(w1.shape[0])
        hidden_size = int(w2.shape[1])
        intermediate_size = int(w2.shape[2]) * 2
        unit_scale = self._unit_scale(w1.device, num_experts)
        w1_global_scale = self._weight_global_scale(
            w1.device, num_experts, self.g1_alphas, "w1 global scales"
        )
        w2_global_scale = self._weight_global_scale(
            w2.device, num_experts, self.g2_alphas, "w2 global scales"
        )

        if quant_mode in ("nvfp4", "w4a8_nvfp4"):
            if self.a1_gscale is None or self.a2_gscale is None:
                raise ValueError("b12x NVFP4 MoE requires activation global scales")
            a1_gscale = _normalize_expert_scale(self.a1_gscale).to(w1.device)
            a2_gscale = _normalize_expert_scale(self.a2_gscale).to(w2.device)
        else:
            a1_gscale = unit_scale
            a2_gscale = unit_scale

        weight_plan = fused_moe.plan_weights(
            quant_modes=quant_mode,
            source_format=self._source_format,
            activation=_b12x_activation_name(activation),
            params_dtype=params_dtype,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            w13_layout=self._w13_layout,
        )
        return fused_moe.prepare_weights(
            plan=weight_plan,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_global_scale=w1_global_scale,
            a1_gscale=a1_gscale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_global_scale=w2_global_scale,
            a2_gscale=a2_gscale,
            params_dtype=params_dtype,
        )

    def _refresh_quant_config(self, layer: torch.nn.Module) -> None:
        self.quant_config._w1.scale = layer.w13_weight_scale
        self.quant_config._w2.scale = layer.w2_weight_scale
        if self._source_format != "modelopt_nvfp4":
            return

        self.quant_config._w1.alpha_or_gscale = layer.w13_weight_scale_2
        self.quant_config._w2.alpha_or_gscale = layer.w2_weight_scale_2
        if self._quant_mode in ("nvfp4", "w4a8_nvfp4"):
            self.quant_config._a1.alpha_or_gscale = 1.0 / layer.w13_input_scale
            self.quant_config._a2.alpha_or_gscale = 1.0 / layer.w2_input_scale

    def _release_source_parameters(self, layer: torch.nn.Module) -> None:
        if self._source_parameters_released:
            return
        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            self.quant_config._w1.scale = w1_scale
        if w2_scale is not None:
            self.quant_config._w2.scale = w2_scale
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")
        self._source_parameters_released = True

    def _reuse_prepared_storage(self, layer: torch.nn.Module, prepared: Any) -> Any:
        previous = getattr(layer, "_b12x_prepared_experts", None)
        prepared = reuse_packed_weight_storage(previous, prepared)
        if prepared is not previous:
            self._plans.clear()
        self._prepared_experts = prepared
        layer._b12x_prepared_experts = prepared
        return prepared

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self._apply_router_weight_on_input = layer.apply_router_weight_on_input
        if self._apply_router_weight_on_input and self._quant_mode != "w4a16":
            raise ValueError(
                "b12x MoE supports apply_router_weight_on_input only with W4A16"
            )
        self._source_parameters_released = False
        self._refresh_quant_config(layer)
        prepared = self._prepare_experts(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=layer.activation,
            params_dtype=self.moe_config.in_dtype,
        )
        prepared = self._reuse_prepared_storage(layer, prepared)
        if prepared.plan.discards_source_parameters:
            self._release_source_parameters(layer)
        layer.b12x_warmup_provider = self

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: mk.FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        if moe_config.has_bias:
            return False, "kernel does not support expert biases"
        if moe_config.in_dtype not in (torch.float16, torch.bfloat16):
            return (
                False,
                f"kernel does not support {moe_config.in_dtype} input/output dtype",
            )
        if moe_config.activation == MoEActivation.SITU and (
            moe_config.activation_situ_beta != 4.0
            or moe_config.activation_situ_linear_beta != 25.0
        ):
            return False, "kernel supports only SiTU beta=4 and linear_beta=25"
        if (
            activation_key is not None
            and moe_config.activation == MoEActivation.SWIGLUOAI_UNINTERLEAVE
        ):
            return (
                False,
                "kernel does not support swigluoai_uninterleave with W4A8",
            )
        unpadded_intermediate_size = (
            moe_config.intermediate_size_per_partition_unpadded
            or moe_config.intermediate_size_per_partition
        )
        if weight_key == kMxfp4Static and unpadded_intermediate_size % 32 != 0:
            return (
                False,
                "MXFP4 requires the per-rank intermediate size to be divisible by 32",
            )
        if weight_key == kMxfp4Static and activation_key == kMxfp8Dynamic:
            if moe_config.activation not in (
                MoEActivation.SILU,
                MoEActivation.SITU,
            ):
                return False, "MXFP4 W4A8 supports only SiLU and SiTU"
            if (
                moe_config.hidden_dim % 256 != 0
                or moe_config.intermediate_size_per_partition % 32 != 0
            ):
                return (
                    False,
                    "MXFP4 W4A8 requires hidden size divisible by 256 and "
                    "per-rank intermediate size divisible by 32",
                )
        return mk.FusedMoEExperts.is_supported_config(
            cls, moe_config, weight_key, activation_key, activation_format
        )

    @staticmethod
    def _supports_current_device() -> bool:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        ):
            return False
        fused_moe = get_b12x_fused_moe()
        if fused_moe is None:
            return False
        return fused_moe.is_supported()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kMxfp4Static, kMxfp8Dynamic),
            (kMxfp4Static, None),
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, kMxfp8Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.SITU,
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size == 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _prepared(self) -> Any:
        if self._prepared_experts is None:
            raise RuntimeError(
                "b12x MoE weights must be prepared by process_weights_after_loading"
            )
        return self._prepared_experts

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() and w2.numel():
            return super().moe_problem_size(a1, w1, w2, topk_ids)
        prepared = self._prepared()
        tokens = int(a1.shape[0] if a1.ndim == 2 else a1.shape[1])
        return (
            int(prepared.num_experts),
            tokens,
            int(prepared.intermediate_size) * 2,
            int(a1.shape[-1]),
            int(topk_ids.shape[1]),
        )

    def _plan(
        self,
        *,
        tokens: int,
        topk: int,
        activation: MoEActivation,
        apply_router_weight_on_input: bool = False,
    ) -> Any:
        fused_moe = _require_b12x_fused_moe()

        key = (
            max(int(tokens), 1),
            int(topk),
            activation,
            bool(apply_router_weight_on_input),
        )
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        if _is_current_stream_capturing():
            raise RuntimeError("b12x MoE plans must be created before CUDA capture")

        limit, alpha, beta = self._swiglu_params(activation)
        prepared = self._prepared()
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=key[0],
                num_topk=key[1],
                device=prepared.w1_fp4.device,
                weight_plan=prepared.plan,
                core_token_counts=(key[0],),
                route_num_experts=0,
                quant_mode=self._quant_mode,
                apply_router_weight_on_input=key[3],
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
                frozen=True,
            )
        )
        self._plans[key] = plan
        return plan

    def get_b12x_warmup_unit(
        self,
        layer: torch.nn.Module,
        token_counts: tuple[int, ...],
        output_dtype: torch.dtype,
    ) -> B12xWarmupUnit:
        assert output_dtype == self.moe_config.in_dtype
        prepared = self._prepared()
        activation = layer.activation
        limit, alpha, beta = self._swiglu_params(activation)

        def compile() -> None:
            self.warmup_launches(layer, token_counts=token_counts)

        return B12xWarmupUnit(
            name="MoE",
            key=(
                type(self),
                prepared.w1_fp4.device,
                output_dtype,
                self._quant_mode,
                self._source_format,
                self._w13_layout,
                int(prepared.num_experts),
                int(prepared.hidden_size),
                int(prepared.intermediate_size),
                int(self.moe_config.experts_per_token),
                _b12x_activation_name(activation),
                bool(layer.apply_router_weight_on_input),
                limit,
                alpha,
                beta,
            ),
            compile=compile,
        )

    @torch.inference_mode()
    def warmup_launches(
        self,
        layer: torch.nn.Module,
        *,
        token_counts: Iterable[int],
    ) -> int:
        """Compile one representative launch for every planned regime."""
        activation = layer.activation
        dtype = self.moe_config.in_dtype
        topk = int(self.moe_config.experts_per_token)
        apply_router_weight_on_input = bool(layer.apply_router_weight_on_input)
        limit, alpha, beta = self._swiglu_params(activation)
        prepared = self._prepared()
        device = prepared.w1_fp4.device
        launch_tokens: dict[tuple[Any, ...], int] = {}
        for tokens in sorted({int(count) for count in token_counts if int(count) > 0}):
            execution_plan = _b12x_moe_execution_plan(
                tokens=tokens,
                topk=topk,
                prepared=prepared,
                quant_mode=self._quant_mode,
                apply_router_weight_on_input=apply_router_weight_on_input,
                swiglu_limit=limit,
                swiglu_alpha=alpha,
                swiglu_beta=beta,
            )
            signature = (execution_plan.implementation, execution_plan.execution)
            launch_tokens.setdefault(signature, tokens)

        for tokens in launch_tokens.values():
            hidden_states = torch.zeros(
                (tokens, int(prepared.hidden_size)),
                dtype=dtype,
                device=device,
            )
            output = torch.empty_like(hidden_states)
            topk_ids = (
                torch.arange(topk, device=device, dtype=torch.int32)
                .unsqueeze(0)
                .expand(tokens, -1)
                .contiguous()
            )
            topk_ids.remainder_(int(prepared.num_experts))
            topk_weights = torch.full(
                (tokens, topk),
                1.0 / topk,
                dtype=torch.float32,
                device=device,
            )
            plan = self._plan(
                tokens=tokens,
                topk=topk,
                activation=activation,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
            scratch = torch.empty(
                (_b12x_scratch_nbytes(plan),),
                dtype=torch.uint8,
                device=device,
            )
            _run_b12x_moe_plan(
                plan=plan,
                scratch=scratch,
                hidden_states=hidden_states,
                prepared=prepared,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                output=output,
                unit_scale_contract=self._quant_mode == "w4a16",
            )
        return len(launch_tokens)

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
        del N, global_num_experts, local_num_experts, expert_tokens_meta
        plan = self._plan(
            tokens=M,
            topk=topk,
            activation=activation,
            apply_router_weight_on_input=self._apply_router_weight_on_input,
        )
        itemsize = self.moe_config.in_dtype.itemsize
        scratch_elements = max(
            1, (_b12x_scratch_nbytes(plan) + itemsize - 1) // itemsize
        )
        return (0,), (scratch_elements,), (M, K)

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        del w1, w2, global_num_experts
        del a1q_scale, a2_scale, workspace13, expert_tokens_meta
        if expert_map is not None:
            raise ValueError("b12x TP MoE does not support expert maps")
        if bool(apply_router_weight_on_input) != self._apply_router_weight_on_input:
            raise ValueError(
                "apply_router_weight_on_input does not match the prepared b12x MoE plan"
            )
        prepared = self._prepared()
        topk_ids = _normalize_topk_ids(topk_ids)
        topk_weights = _normalize_topk_weights(topk_weights)
        plan = self._plan(
            tokens=int(hidden_states.shape[0]),
            topk=int(topk_ids.shape[1]),
            activation=activation,
            apply_router_weight_on_input=bool(apply_router_weight_on_input),
        )
        scratch = _workspace_as_b12x_scratch(workspace2, plan)

        _run_b12x_moe_plan(
            plan=plan,
            scratch=scratch,
            hidden_states=hidden_states,
            prepared=prepared,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            output=output,
            unit_scale_contract=self._quant_mode == "w4a16",
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
