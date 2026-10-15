# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Temporary MiniMax-local MonoKernel lifecycle adapter.

This stays detached from the model until the FFN kernel passes its viability
gate. It mirrors the proposed common runtime's attach and per-step interfaces
without routing production forwards through a fake kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.platforms import current_platform

from .routing import StepDecision, decide_step, tensor_identity
from .spec import MONO_SPEC, MiniMaxM3MonoSpec
from .weights import MonoLayerWeights

if TYPE_CHECKING:
    from vllm.models.minimax_m3.amd.model import (
        MiniMaxM3DecoderLayer,
        MiniMaxM3Model,
    )

logger = init_logger(__name__)

TensorIdentity = tuple[int, int, torch.dtype, tuple[int, ...], int]


@dataclass(frozen=True)
class CacheBinding:
    main: torch.Tensor
    key_pages: torch.Tensor
    value_pages: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    index_key: torch.Tensor
    sparse_table: torch.Tensor
    sparse_context: torch.Tensor

    def identity(self) -> dict[str, TensorIdentity]:
        return {name: tensor_identity(value) for name, value in vars(self).items()}


class CacheBinder(Protocol):
    def bind(self, layer_id: int, binding: CacheBinding) -> None: ...


def _runtime_refusals(spec: MiniMaxM3MonoSpec) -> list[str]:
    reasons: list[str] = []
    if not current_platform.is_rocm():
        reasons.append("requires ROCm")
        return reasons
    try:
        import flydsl  # noqa: F401
        from aiter.ops.flydsl.kernels import buffer_ops  # noqa: F401

        from vllm.platforms.rocm import get_cdna_version
    except ImportError as err:
        reasons.append(f"requires FlyDSL and AITER FlyDSL helpers ({err})")
        return reasons
    if get_cdna_version() != 4:
        reasons.append("requires CDNA4/gfx950")
    try:
        cus = current_platform.num_compute_units(
            torch.accelerator.current_device_index()
        )
    except RuntimeError as err:
        reasons.append(f"cannot determine compute-unit count ({err})")
    else:
        if cus != spec.blocks:
            reasons.append(f"requires {spec.blocks} compute units, found {cus}")
    return reasons


class MiniMaxM3MonoLayer:
    """Owns one rank's mono lifecycle and a single decision for each forward."""

    def __init__(
        self, model: MiniMaxM3Model, config: VllmConfig, binder: CacheBinder
    ) -> None:
        self.model = model
        self.spec = MONO_SPEC
        reasons = self.spec.refusal_reasons(config) + _runtime_refusals(self.spec)
        if reasons:
            detail = "; ".join(reasons)
            raise ValueError(f"--monokernel unsupported: {detail}.")
        self.layers: list[MiniMaxM3DecoderLayer] = []
        self.binder = binder
        self.active = False
        self.last_decision = StepDecision(False, "not started")
        self._cache_identities: dict[int, dict[str, TensorIdentity]] = {}
        self.weights: dict[int, MonoLayerWeights] = {}
        self._weights_attached = False

    def attach(self) -> None:
        expected = list(range(self.spec.first_layer, self.spec.last_layer + 1))
        layers = [
            layer
            for layer in self.model.layers
            if getattr(layer, "layer_id", -1) in expected
        ]
        actual = [getattr(layer, "layer_id", -1) for layer in layers]
        if actual != expected or any(
            not getattr(layer, "is_moe_layer", False)
            or not hasattr(layer.self_attn, "indexer")
            for layer in layers
        ):
            raise ValueError("--monokernel requires exactly sparse MoE layers 3-59")
        self.layers = layers

    def pack_after_load(self) -> None:
        if self._weights_attached:
            return
        failures: list[str] = []
        for layer in self.layers:
            try:
                self.weights[layer.layer_id] = MonoLayerWeights.from_layer(layer)
            except ValueError as err:
                failures.append(str(err))
        if failures:
            raise ValueError("; ".join(failures))
        self._weights_attached = True

    @staticmethod
    def _cache_binding(layer: MiniMaxM3DecoderLayer) -> CacheBinding:
        attn = layer.self_attn
        main = attn.kv_cache
        key_pages, value_pages = attn.get_aiter_sparse_pa_kv_cache()
        binding = CacheBinding(
            main=main,
            key_pages=key_pages,
            value_pages=value_pages,
            k_scale=attn._k_scale,
            v_scale=attn._v_scale,
            index_key=attn.indexer.index_cache.kv_cache,
            sparse_table=attn.sparse_bt_buffer,
            sparse_context=attn.sparse_ctx_buffer,
        )
        if (
            main.numel() == 0
            or main.device.type != "cuda"
            or main.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or main.shape[2] != 128
        ):
            raise ValueError("invalid main FP8 KV cache binding")
        if (
            key_pages.device != main.device
            or value_pages.device != main.device
            or key_pages.stride(-1) != 1
            or value_pages.stride(-1) != 1
        ):
            raise ValueError("invalid AITER page-16 K/V views")
        for name, scale in (("K", binding.k_scale), ("V", binding.v_scale)):
            if (
                not isinstance(scale, torch.Tensor)
                or scale.device != main.device
                or scale.dtype != torch.float32
                or scale.numel() != 1
            ):
                raise ValueError(f"invalid {name} cache scale binding")
        if (
            binding.index_key.device != main.device
            or binding.index_key.dtype not in (torch.uint8, torch.float8_e4m3fn)
            or binding.index_key.shape[:2] != main.shape[:1] + main.shape[2:3]
        ):
            raise ValueError("invalid index-key cache binding")
        for scratch in (binding.sparse_table, binding.sparse_context):
            if (
                not isinstance(scratch, torch.Tensor)
                or scratch.device != main.device
                or scratch.dtype != torch.int32
                or not scratch.is_contiguous()
            ):
                raise ValueError("invalid model-owned sparse scratch binding")
        return binding

    def _bind_caches(self) -> None:
        changed: list[tuple[int, CacheBinding, dict[str, TensorIdentity]]] = []
        for layer in self.layers:
            binding = self._cache_binding(layer)
            identity = binding.identity()
            if self._cache_identities.get(layer.layer_id) != identity:
                changed.append((layer.layer_id, binding, identity))
        if not changed:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "--monokernel cache first-bind/rebind reached inside graph capture"
            )
        failure: Exception | None = None
        try:
            for layer_id, binding, _ in changed:
                self.binder.bind(layer_id, binding)
        except Exception as err:
            failure = err
        from vllm.models.deepseek_v41.amd.mono.common.peer_memory import tp_agree

        if not tp_agree(failure is None, get_tp_group().cpu_group):
            raise RuntimeError(
                "rank-uniform MonoKernel cache rebind failed"
            ) from failure
        for layer_id, _, identity in changed:
            self._cache_identities[layer_id] = identity

    def begin_step(self, rows: int) -> None:
        self.active = False
        if not is_forward_context_available() or not self.layers:
            self.last_decision = StepDecision(False, "forward context missing")
            return
        context = get_forward_context()
        all_metadata = context.attn_metadata
        if not isinstance(all_metadata, dict):
            self.last_decision = StepDecision(False, "attention metadata missing")
            return
        layer_name = self.layers[0].self_attn.layer_name
        self.last_decision = decide_step(
            self.spec,
            rows,
            context.cudagraph_runtime_mode,
            all_metadata.get(layer_name),
        )
        if not self.last_decision.use_mono:
            return
        if not self._weights_attached:
            raise RuntimeError("--monokernel weights were not packed after load")
        self._bind_caches()
        self.active = True
        logger.info_once(
            "MiniMax-M3 MonoKernel lifecycle active for widths %s.",
            self.spec.widths,
        )
