# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout-neutral target-state transport for KDA RecoverSSM."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from math import prod

import torch
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

KDA_TARGET_STATE_TRANSPORT = "target_state_v1"
KDA_TARGET_CONV_REGION = "kda_target_conv_v1"
KDA_BASE_RECURRENT_REGION = "kda_base_recurrent_v1"


@dataclass(frozen=True)
class KDATransportRegion:
    kind: str
    tensor: torch.Tensor
    block_stride_bytes: int
    content_len_bytes: int


def _split_page_states(
    cache: torch.Tensor,
    spec: MambaSpec,
) -> tuple[torch.Tensor, ...]:
    if cache.ndim != 4 or tuple(cache.shape[1:3]) != (1, 1):
        raise ValueError(
            "KDA target-state transport requires a [blocks, 1, 1, bytes] cache"
        )
    if cache.element_size() != 1:
        raise ValueError("KDA target-state transport requires a byte cache view")

    pages = cache[:, 0, 0]
    if pages.shape[1] < spec.state_content_size_bytes:
        raise ValueError("KDA cache page is shorter than its declared state fields")
    states: list[torch.Tensor] = []
    offset = 0
    for shape, dtype in zip(spec.shapes, spec.dtypes, strict=True):
        num_bytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
        state = pages[:, offset : offset + num_bytes].view(dtype)
        states.append(state.view(cache.shape[0], *shape))
        offset += num_bytes
    if offset != spec.state_content_size_bytes:
        raise ValueError("KDA state fields do not cover the declared page content")
    return tuple(states)


class KDATargetStateLayerTransport:
    """Expose only target state shared by base KDA and RecoverSSM pages."""

    def __init__(
        self,
        layer_name: str,
        group_index: int,
        cache: torch.Tensor,
        spec: MambaSpec,
        *,
        conv_state_dim_first: bool | None = None,
    ) -> None:
        if spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN:
            raise ValueError("KDA transport requires a GDN_ATTN MambaSpec")
        if len(spec.shapes) not in (2, 4):
            raise ValueError("KDA transport requires base or RecoverSSM state fields")
        if spec.num_speculative_blocks:
            raise ValueError(
                "ordinary speculative KDA pages are not target-state transportable"
            )

        self.layer_name = layer_name
        self.group_index = group_index
        self.cache = cache
        self.states = _split_page_states(cache, spec)
        dim_first = (
            is_conv_state_dim_first()
            if conv_state_dim_first is None
            else conv_state_dim_first
        )
        conv_state, self.recurrent_state = self.states[:2]
        self.local_conv = conv_state if dim_first else conv_state.transpose(-1, -2)
        if len(self.states) == 4:
            verify_len = spec.shapes[2][1]
            self.conv_history_len = self.local_conv.shape[-1] - verify_len + 1
        else:
            self.conv_history_len = self.local_conv.shape[-1]
        if self.conv_history_len <= 0:
            raise ValueError("RecoverSSM conv state has no target history")

        self.target_conv = torch.empty(
            (
                cache.shape[0],
                self.local_conv.shape[-2],
                self.conv_history_len,
            ),
            dtype=conv_state.dtype,
            device=conv_state.device,
        )

    @property
    def regions(self) -> tuple[KDATransportRegion, KDATransportRegion]:
        return (
            KDATransportRegion(
                kind=KDA_TARGET_CONV_REGION,
                tensor=self.target_conv,
                block_stride_bytes=(
                    self.target_conv.stride(0) * self.target_conv.element_size()
                ),
                content_len_bytes=(
                    self.target_conv[0].numel() * self.target_conv.element_size()
                ),
            ),
            KDATransportRegion(
                kind=KDA_BASE_RECURRENT_REGION,
                tensor=self.recurrent_state,
                block_stride_bytes=(
                    self.recurrent_state.stride(0) * self.recurrent_state.element_size()
                ),
                content_len_bytes=(
                    self.recurrent_state[0].numel()
                    * self.recurrent_state.element_size()
                ),
            ),
        )

    def stage_blocks(self, block_ids: list[int]) -> None:
        indices = self._indices(block_ids)
        if indices is None:
            return
        source = self.local_conv.index_select(0, indices)[..., : self.conv_history_len]
        self.target_conv.index_copy_(0, indices, source)

    def materialize_blocks(self, block_ids: list[int]) -> None:
        indices = self._indices(block_ids)
        if indices is None:
            return
        conv = torch.zeros(
            (indices.numel(), *self.local_conv.shape[1:]),
            dtype=self.local_conv.dtype,
            device=self.local_conv.device,
        )
        conv[..., : self.conv_history_len].copy_(
            self.target_conv.index_select(0, indices)
        )
        self.local_conv.index_copy_(0, indices, conv)
        for record in self.states[2:]:
            record.index_fill_(0, indices, 0)

    def _indices(self, block_ids: list[int]) -> torch.Tensor | None:
        valid = list(
            dict.fromkeys(
                block_id
                for block_id in block_ids
                if block_id != NULL_BLOCK_ID and block_id >= 0
            )
        )
        if not valid:
            return None
        if max(valid) >= self.cache.shape[0]:
            raise ValueError(
                f"KDA block id {max(valid)} exceeds {self.cache.shape[0]} blocks"
            )
        return torch.tensor(valid, dtype=torch.long, device=self.cache.device)


class KDATargetStateTransport:
    def __init__(self, layers: dict[str, KDATargetStateLayerTransport]) -> None:
        self.layers = layers
        self._layers_by_group: dict[int, list[KDATargetStateLayerTransport]] = {}
        for layer in layers.values():
            self._layers_by_group.setdefault(layer.group_index, []).append(layer)
        self._lock = threading.Lock()

    @classmethod
    def create(
        cls,
        kv_caches: dict[str, torch.Tensor],
        kv_cache_config: KVCacheConfig,
        *,
        conv_state_dim_first: bool | None = None,
    ) -> KDATargetStateTransport:
        layers: dict[str, KDATargetStateLayerTransport] = {}
        for group_index, group in enumerate(kv_cache_config.transfer_groups):
            group_spec = group.kv_cache_spec
            specs_by_layer = getattr(group_spec, "kv_cache_specs", {})
            for layer_name in group.layer_names:
                spec = specs_by_layer.get(layer_name, group_spec)
                cache = kv_caches.get(layer_name)
                if cache is not None and isinstance(spec, MambaSpec):
                    layers[layer_name] = KDATargetStateLayerTransport(
                        layer_name,
                        group_index,
                        cache,
                        spec,
                        conv_state_dim_first=conv_state_dim_first,
                    )
        return cls(layers)

    def regions_for_layer(self, layer_name: str) -> tuple[KDATransportRegion, ...]:
        layer = self.layers.get(layer_name)
        return () if layer is None else layer.regions

    def stage_groups(self, block_ids_by_group: list[list[int]]) -> None:
        self._apply_groups(block_ids_by_group, materialize=False)

    def materialize_groups(self, block_ids_by_group: list[list[int]]) -> None:
        self._apply_groups(block_ids_by_group, materialize=True)

    def _apply_groups(
        self,
        block_ids_by_group: list[list[int]],
        *,
        materialize: bool,
    ) -> None:
        if not self.layers:
            return
        with self._lock:
            devices = {layer.cache.device for layer in self.layers.values()}
            if len(devices) != 1:
                raise ValueError("KDA transport layers must share one device")
            device = next(iter(devices))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for group_index, block_ids in enumerate(block_ids_by_group):
                for layer in self._layers_by_group.get(group_index, ()):
                    if materialize:
                        layer.materialize_blocks(block_ids)
                    else:
                        layer.stage_blocks(block_ids)
            if device.type == "cuda":
                torch.cuda.synchronize(device)


def kda_target_state_transport_enabled(extra_config: dict[str, object]) -> bool:
    return extra_config.get("kda_transport_policy") == KDA_TARGET_STATE_TRANSPORT
