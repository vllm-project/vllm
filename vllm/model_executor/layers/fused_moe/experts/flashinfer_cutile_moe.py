# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 routed experts that pick W4A4 or W4A16 per forward on SM12x.

Both precisions run through FlashInfer's unified MoE runners and read one
prepared weight view, so switching precision costs no second weight copy.
The choice is made from the token count before any activation quantization:
W4A16 (BF16 activations) at or below the cutoff, W4A4 above it.
"""

from dataclasses import dataclass
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform

# FlashInfer backend config class names. Every candidate must consume the view
# built by ``CuTileNvfp4Config.prepare_weights`` unchanged.
W4A4_BACKENDS = {
    "cutile": "CuTileNvfp4Config",
    "sm12x": "SM12xNvfp4Config",
}
W4A16_BACKENDS = {
    "cutile": "CuTileNvfp4Bf16Config",
    "sm12x": "SM12xNvfp4Bf16Config",
}

# Token-bucket scratch of SM12x W4A4 runners, shared by layers of one geometry.
_SHARED_W4A4_WORKSPACES: dict[tuple, dict[int, list[torch.Tensor]]] = {}

_ACTIVATIONS = {
    MoEActivation.SILU: "SwiGLU",
    MoEActivation.RELU2_NO_MUL: "ReLU2",
}


def _fused_moe():
    import flashinfer.fused_moe as fused_moe

    return fused_moe


def has_flashinfer_cutile_nvfp4(
    w4a16_backend: str | None = None, w4a4_backend: str = "cutile"
) -> bool:
    """Whether FlashInfer exposes the requested unified-MoE NVFP4 backends."""
    try:
        fused_moe = _fused_moe()
        from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
    except ImportError:
        return False
    names = [W4A4_BACKENDS.get(w4a4_backend, "")]
    if w4a16_backend is not None:
        names.append(W4A16_BACKENDS.get(w4a16_backend, ""))
    classes = [getattr(fused_moe, name, None) for name in names]
    return all(c is not None and c in _BACKEND_RUNNERS for c in classes)


def prepare_cutile_nvfp4_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    activation: MoEActivation,
) -> dict[str, torch.Tensor]:
    """Build the single weight view shared by the W4A4 and W4A16 runners.

    vLLM holds gated w13 as ``[gate; up]`` with linear block scales. FlashInfer
    takes ``[up; gate]`` and returns ``[gate; up]`` with SM12x-swizzled scales.
    """
    num_experts, w13_rows, packed_k = w13.shape
    intermediate = w2.shape[2] * 2
    if activation == MoEActivation.SILU:
        w13 = torch.cat((w13[:, intermediate:], w13[:, :intermediate]), dim=1)
        w13_scale = torch.cat(
            (w13_scale[:, intermediate:], w13_scale[:, :intermediate]), dim=1
        )
    fused_moe = _fused_moe()
    return fused_moe.CuTileNvfp4Config.prepare_weights(
        w13,
        w13_scale,
        w13_scale_2.float().reshape(num_experts),
        w2,
        w2_scale,
        w2_scale_2.float().reshape(num_experts),
        num_local_experts=num_experts,
        hidden_size=packed_k * 2,
        intermediate_size=intermediate,
        activation=getattr(fused_moe, _ACTIVATIONS[activation])(),
        device=w13.device,
    )


@dataclass
class _Path:
    runner: Any
    op_name: str


# (op name, geometry) pairs whose SM12x kernels were compiled for every token
# bucket in this process; the compiled-kernel cache is shared by all layers.
_WARMED_BUCKETS: set[tuple] = set()


class CuTileNvfp4DynamicMoE:
    """Token-count dispatch between two unified-MoE runners over one view."""

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        activation: MoEActivation,
        max_num_tokens: int,
        a16_max_tokens: int,
        w4a16_backend: str,
        device: torch.device,
        w4a4_backend: str = "cutile",
    ):
        fused_moe = _fused_moe()
        from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

        if activation not in _ACTIVATIONS:
            raise ValueError(f"unsupported activation {activation!r}")
        for knob, name, choices in (
            ("w4a4", w4a4_backend, W4A4_BACKENDS),
            ("w4a16", w4a16_backend, W4A16_BACKENDS),
        ):
            if name not in choices:
                raise ValueError(
                    f"nvfp4_moe_dynamic_{knob}_backend={name!r}; "
                    f"expected one of {sorted(choices)}"
                )
        self.a16_max_tokens = a16_max_tokens
        self.max_num_tokens = max_num_tokens
        self._geometry = (num_experts, top_k, intermediate_size, activation)
        self._weights: Any = None

        def build(config_name: str, act_format: Any) -> _Path:
            config_cls = getattr(fused_moe, config_name)
            config = fused_moe.MoEConfig(
                routing=fused_moe.RoutingConfig(num_experts=num_experts, top_k=top_k),
                quant=fused_moe.QuantConfig(
                    weight=fused_moe.QuantFormat.NVFP4, activation=act_format
                ),
                experts=fused_moe.ExpertConfig(intermediate_size=intermediate_size),
                activation=getattr(fused_moe, _ACTIVATIONS[activation])(),
                backend=fused_moe.BackendOptions((config_cls(),)),
                execution=fused_moe.ExecutionConfig(
                    enable_pdl=False, tune_max_num_tokens=max_num_tokens
                ),
            )
            runner = _BACKEND_RUNNERS[config_cls](config, device)
            runner.check_support()
            runner.build()
            return _Path(runner, f"vllm_nvfp4_moe_{config_name}")

        self.w4a4 = build(W4A4_BACKENDS[w4a4_backend], fused_moe.QuantFormat.NVFP4)
        self.w4a16 = (
            build(W4A16_BACKENDS[w4a16_backend], fused_moe.QuantFormat.BF16)
            if a16_max_tokens > 0
            else None
        )
        # The cuTile expert sort has no sentinel for padding rows (expert id
        # -1); the SM12x backends skip ids outside [0, E) themselves.
        self._remap_padding = w4a4_backend == "cutile" or (
            self.w4a16 is not None and w4a16_backend == "cutile"
        )

    def set_weights(self, view: dict[str, torch.Tensor]) -> None:
        pack = _fused_moe().MoEWeightPack()
        for path in (self.w4a4, self.w4a16):
            if path is not None:
                for key in getattr(
                    path.runner, "weight_view_keys", (path.runner.backend_key,)
                ):
                    pack.prepare_for(key, view)
        self._weights = pack
        # SM12x W4A4 rewrites its whole scratch from the inputs on every call
        # and MoE layers run one after another, so layers with the same
        # geometry share one workspace per token bucket instead of keeping
        # num_layers copies (about 5 KB per token per layer).
        runner = self.w4a4.runner
        if runner.backend_key == "sm12x_nvfp4" and hasattr(runner, "_workspaces"):
            key = (str(runner.device), *self._geometry, view["w2"].shape[1])
            runner._workspaces = _SHARED_W4A4_WORKSPACES.setdefault(key, {})

    def select(self, num_tokens: int) -> _Path:
        if self.w4a16 is not None and num_tokens <= self.a16_max_tokens:
            return self.w4a16
        return self.w4a4

    def _inputs(self, path: _Path, output, hidden_states, topk_ids, topk_weights):
        act = _fused_moe().MoEActivationPack(
            hidden_states_q=hidden_states,
            hidden_states_scale=None,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        inputs = path.runner.pack_inputs(act, self._weights)
        # Write in place when the runner's output buffer has the call's shape;
        # runners that pad to a token bucket keep their own buffer.
        if inputs[0].shape == output.shape:
            inputs[0] = output
        return inputs

    def run(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        from flashinfer.autotuner import AutoTuner

        assert self._weights is not None, "set_weights must run before run()"
        if self._remap_padding:
            invalid = topk_ids < 0
            topk_ids = topk_ids.masked_fill(invalid, 0)
            topk_weights = topk_weights.masked_fill(invalid, 0.0)
        tuner = AutoTuner.get()
        if tuner.is_tuning_mode:
            self._warm_buckets(hidden_states, topk_ids, topk_weights)
        chosen = self.select(hidden_states.shape[0])
        paths = [chosen]
        # Tuning passes run at one token count; tune both precisions there so
        # decode-sized buckets of the W4A16 path are not left on fallbacks.
        if tuner.is_tuning_mode:
            paths = [p for p in (self.w4a4, self.w4a16) if p is not None]
        tactic: Any = -1
        for path in paths:
            inputs = self._inputs(path, output, hidden_states, topk_ids, topk_weights)
            _, best = tuner.choose_one(
                path.op_name, [path.runner], path.runner.tuning_config, inputs
            )
            if path is chosen:
                tactic, chosen_inputs = best, inputs
        result = chosen.runner.forward(chosen_inputs, tactic=tactic)
        if result.data_ptr() != output.data_ptr():
            output.copy_(result)
        return output

    def _warm_buckets(self, hidden_states, topk_ids, topk_weights) -> None:
        """Compile the SM12x kernels for every token bucket a path can serve.

        They are specialized per token bucket, so an unseen bucket would
        otherwise compile on its first serving step and stall it.
        """
        from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets

        if torch.cuda.is_current_stream_capturing():
            return
        for path, limit in (
            (self.w4a4, self.max_num_tokens),
            (self.w4a16, self.a16_max_tokens),
        ):
            if path is None or not path.runner.backend_key.startswith("sm12x"):
                continue
            key = (path.op_name, self._geometry, hidden_states.shape[1])
            if key in _WARMED_BUCKETS:
                continue
            _WARMED_BUCKETS.add(key)
            for bucket in get_hybrid_num_tokens_buckets(
                min(limit, self.max_num_tokens)
            ):
                x = hidden_states[:1].expand(bucket, -1).contiguous()
                ids = topk_ids[:1].expand(bucket, -1).contiguous()
                weights = topk_weights[:1].expand(bucket, -1).contiguous()
                inputs = self._inputs(path, x.new_empty(x.shape), x, ids, weights)
                path.runner.forward(inputs, tactic=-1)


class FlashInferCuTileNvfp4Experts(mk.FusedMoEExpertsModular):
    """FlashInfer unified-MoE NVFP4 experts with per-forward precision choice.

    ``kernel_config.nvfp4_moe_dynamic_max_tokens`` > 0 runs W4A16 for forwards
    with at most that many tokens and W4A4 otherwise; 0 always runs W4A4.
    BF16 hidden states reach the experts unquantized, and both W4A4 backends
    quantize activations dynamically in-kernel, so calibrated input scales
    are not used.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        kernel_config = get_current_vllm_config().kernel_config
        self.dispatch = CuTileNvfp4DynamicMoE(
            num_experts=moe_config.num_local_experts,
            top_k=moe_config.experts_per_token,
            intermediate_size=moe_config.intermediate_size_per_partition,
            activation=moe_config.activation,
            max_num_tokens=moe_config.max_num_tokens,
            a16_max_tokens=kernel_config.nvfp4_moe_dynamic_max_tokens,
            w4a16_backend=kernel_config.nvfp4_moe_dynamic_w4a16_backend,
            device=torch.device(moe_config.device),
            w4a4_backend=kernel_config.nvfp4_moe_dynamic_w4a4_backend,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The oracle already stored the prepared view on the layer.
        view = {
            "w1": layer.w13_weight,
            "w1_scale": layer.w13_weight_scale,
            "w1_global_scale": layer.w13_weight_scale_2,
            "w2": layer.w2_weight,
            "w2_scale": layer.w2_weight_scale,
            "w2_global_scale": layer.w2_weight_scale_2,
        }
        input_scales = getattr(layer, "cutile_input_global_scales", None)
        if input_scales is not None:
            view["w1_input_global_scale"], view["w2_input_global_scale"] = input_scales
        elif self.dispatch.w4a16 is not None:
            # No calibrated activation scales (e.g. a W4A16 checkpoint such as
            # an MTP drafter): W4A4 would quantize with a unit global scale,
            # so run every forward as W4A16.
            self.dispatch.a16_max_tokens = self.dispatch.max_num_tokens
        self.dispatch.set_weights(view)

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
            and has_flashinfer_cutile_nvfp4()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in _ACTIVATIONS

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

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
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        assert expert_map is None and not apply_router_weight_on_input
        self.dispatch.run(
            output,
            hidden_states.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            topk_weights.float().contiguous(),
        )
