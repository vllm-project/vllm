# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import weakref
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed import get_ep_group
from vllm.model_executor.warmup.cutedsl_warmup import (
    CuTeDSLCompileUnit,
    register_cutedsl_warmup_provider,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class _FlashInferMoeEpApi:
    def __init__(self) -> None:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                FleetParams,
                MegaConfig,
                MoEEpMegaLayer,
                MoEEpTensors,
                MoEWeightPack,
                Nvfp4CutedslMegaMoeConfig,
            )
        except ImportError as err:
            raise RuntimeError(
                "flashinfer_moe_ep_cutedsl requires the public FlashInfer "
                "MoE EP API shipped with FlashInfer 0.6.17 or newer"
            ) from err

        self.BootstrapConfig = BootstrapConfig
        self.FleetParams = FleetParams
        self.MegaConfig = MegaConfig
        self.MoEEpMegaLayer = MoEEpMegaLayer
        self.MoEEpTensors = MoEEpTensors
        self.MoEWeightPack = MoEWeightPack
        self.Nvfp4CutedslMegaMoeConfig = Nvfp4CutedslMegaMoeConfig


def _load_flashinfer_moe_ep_api() -> _FlashInferMoeEpApi:
    return _FlashInferMoeEpApi()


class FlashInferMoeEpCutedsl:
    """Own one public FlashInfer CuTeDSL MoE EP layer."""

    def __init__(
        self,
        layer: RoutedExperts,
        moe: FusedMoEConfig,
        *,
        input_norm_const: float,
        fc1_alpha: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc1_norm_const: torch.Tensor,
    ) -> None:
        api = _load_flashinfer_moe_ep_api()
        ep_group = get_ep_group()
        bootstrap = api.BootstrapConfig(
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            process_group=ep_group.device_group,
            device=torch.accelerator.current_device_index(),
        )
        fleet_params = api.FleetParams(
            num_experts=moe.num_experts,
            max_tokens_per_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
        )
        weights = api.MoEWeightPack(
            w13=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )
        megakernel = api.Nvfp4CutedslMegaMoeConfig(
            intermediate_size=moe.intermediate_size,
            top_k=moe.experts_per_token,
            gate_up_clamp=moe.swiglu_limit,
            apply_topk_in_fc1=layer.apply_router_weight_on_input,
            in_kernel_fc2_reduce=False,
            combine_dtype="bf16",
            input_norm_const=input_norm_const,
            fc1_alpha=None,
            fc2_alpha=None,
            fc1_norm_const=None,
            knobs=None,
        )
        backend = api.MegaConfig(
            megakernel=megakernel,
            quantize_input=True,
            preprocess_weights=True,
        )

        self._moe_ep_tensors_cls = api.MoEEpTensors
        self._mega_layer: Any | None = api.MoEEpMegaLayer(
            bootstrap,
            fleet_params,
            weights,
            backend,
        )
        self._hidden_size = moe.hidden_dim
        self._num_experts = moe.num_experts
        self._top_k = moe.experts_per_token
        self._fc1_alpha: torch.Tensor | None = fc1_alpha
        self._fc2_alpha: torch.Tensor | None = fc2_alpha
        self._fc1_norm_const: torch.Tensor | None = fc1_norm_const
        _register_flashinfer_moe_ep_cutedsl(self)

    @property
    def destroyed(self) -> bool:
        return self._mega_layer is None

    def _tensors(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Any:
        return self._moe_ep_tensors_cls(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            fc1_alpha=cast(torch.Tensor, self._fc1_alpha),
            fc2_alpha=cast(torch.Tensor, self._fc2_alpha),
            fc1_norm_const=cast(torch.Tensor, self._fc1_norm_const),
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._mega_layer is None:
            raise RuntimeError("FlashInfer MoE EP layer was destroyed")
        return self._mega_layer(self._tensors(hidden_states, topk_ids, topk_weights))

    def warmup(self) -> None:
        if self._mega_layer is None:
            return
        device = cast(torch.Tensor, self._fc1_alpha).device
        hidden_states = torch.zeros(
            1,
            self._hidden_size,
            dtype=torch.bfloat16,
            device=device,
        )
        topk_ids = (
            torch.arange(
                self._top_k,
                dtype=torch.int32,
                device=device,
            )
            .mul_(self._num_experts // self._top_k)
            .view(1, self._top_k)
        )
        topk_weights = torch.full(
            (1, self._top_k),
            1.0 / self._top_k,
            dtype=torch.float32,
            device=device,
        )
        self._mega_layer.warmup(self._tensors(hidden_states, topk_ids, topk_weights))

    def destroy(self) -> None:
        mega_layer = self._mega_layer
        if mega_layer is None:
            return
        mega_layer.destroy()
        self._mega_layer = None
        self._fc1_alpha = None
        self._fc2_alpha = None
        self._fc1_norm_const = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.destroy()


_FLASHINFER_MOE_EP_CUTEDSL: list[weakref.ReferenceType[FlashInferMoeEpCutedsl]] = []


def _live_flashinfer_moe_ep_cutedsl() -> list[FlashInferMoeEpCutedsl]:
    live: list[FlashInferMoeEpCutedsl] = []
    refs: list[weakref.ReferenceType[FlashInferMoeEpCutedsl]] = []
    for ref in _FLASHINFER_MOE_EP_CUTEDSL:
        adapter = ref()
        if adapter is not None:
            refs.append(ref)
            if not adapter.destroyed:
                live.append(adapter)
    _FLASHINFER_MOE_EP_CUTEDSL[:] = refs
    return live


def _register_flashinfer_moe_ep_cutedsl(
    adapter: FlashInferMoeEpCutedsl,
) -> None:
    _FLASHINFER_MOE_EP_CUTEDSL.append(weakref.ref(adapter))


def has_flashinfer_moe_ep_cutedsl() -> bool:
    return bool(_live_flashinfer_moe_ep_cutedsl())


def warmup_flashinfer_moe_ep_cutedsl() -> None:
    for adapter in _live_flashinfer_moe_ep_cutedsl():
        adapter.warmup()


def destroy_flashinfer_moe_ep_cutedsl() -> None:
    for adapter in _live_flashinfer_moe_ep_cutedsl():
        adapter.destroy()
    _FLASHINFER_MOE_EP_CUTEDSL.clear()


class _FlashInferMoeEpCutedslWarmupProvider:
    def get_cutedsl_warmup_compile_units(self) -> list[CuTeDSLCompileUnit]:
        if not has_flashinfer_moe_ep_cutedsl():
            return []
        return [
            CuTeDSLCompileUnit(
                name="FlashInfer MoE EP CuTeDSL",
                key="flashinfer_moe_ep_cutedsl",
                compile=warmup_flashinfer_moe_ep_cutedsl,
            )
        ]


_WARMUP_PROVIDER = _FlashInferMoeEpCutedslWarmupProvider()
register_cutedsl_warmup_provider(_WARMUP_PROVIDER)
