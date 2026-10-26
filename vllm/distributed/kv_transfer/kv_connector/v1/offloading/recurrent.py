# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restricted GDN state geometry and native canonical storage contract."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.utils.torch_utils import get_kv_cache_torch_dtype
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVQuantMode,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)
from vllm.v1.kv_offload.base import CanonicalKVCaches, CanonicalPageMapping, CopyRun
from vllm.v1.kv_offload.config import (
    CanonicalStorageLayout,
    GDNLayerSchema,
    RecurrentSchema,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class GDNGeometry:
    projection_dims: tuple[int, ...]
    conv_rows: int
    temporal_shape: tuple[int, int, int]
    component_bytes: tuple[int, ...]
    conv_dtype: torch.dtype
    temporal_dtype: torch.dtype

    def schema(self, layer_name: str, tp_size: int) -> GDNLayerSchema:
        q, k, v = self.projection_dims
        heads, vd, kd = self.temporal_shape
        return GDNLayerSchema(
            layer_name=layer_name,
            q_shape_global=(q * tp_size, self.conv_rows),
            k_shape_global=(k * tp_size, self.conv_rows),
            v_shape_global=(v * tp_size, self.conv_rows),
            temporal_shape_global=(heads * tp_size, vd, kd),
            conv_dtype=str(self.conv_dtype).removeprefix("torch."),
            temporal_dtype=str(self.temporal_dtype).removeprefix("torch."),
        )


def derive_gdn_geometry(spec: KVCacheSpec, tp_size: int) -> GDNGeometry | None:
    """Parse only the two-component, fully sharded DS GDN contract."""
    if (
        not isinstance(spec, MambaSpec)
        or spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN
        or spec.tp_replicated
        or not is_conv_state_dim_first()
        or tp_size < 1
        or len(spec.shapes) != 2
        or len(spec.dtypes) != 2
        or tuple(map(len, spec.shapes)) != (2, 3)
        or any(
            type(dim) is not int or dim <= 0 for shape in spec.shapes for dim in shape
        )
        or spec.dtypes
        not in ((torch.bfloat16, torch.float32), (torch.float16, torch.float32))
        or spec.mamba_cache_mode != "align"
        or spec.num_speculative_blocks != 0
        or spec.num_prefill_checkpoint_blocks != 0
        or spec.prefill_checkpoint_alignment is not None
    ):
        return None
    conv_dim, rows = spec.shapes[0]
    heads, vd, kd = spec.shapes[1]
    remainder = conv_dim - heads * vd
    if remainder <= 0 or remainder % 2:
        return None
    q = remainder // 2
    if q % kd or heads % (q // kd):
        return None
    split = derive_mamba_conv_split(spec, tp_size)
    component_bytes = (*split.proj_bytes, split.ssm_sizes[1])
    if (
        spec.page_size_padded is not None
        and spec.page_size_padded < sum(component_bytes)
    ) or split.ssm_sizes[0] % 4:
        return None
    return GDNGeometry(
        split.local_proj_dims,
        rows,
        (heads, vd, kd),
        component_bytes,
        spec.dtypes[0],
        spec.dtypes[1],
    )


def gdn_mapping(
    spec: KVCacheSpec,
    raw: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    tp_size: int,
    rank: int,
) -> CanonicalPageMapping:
    """Validate the live raw page and derive uncoalesced projection-major runs."""
    geometry = derive_gdn_geometry(spec, tp_size)
    if geometry is None or not isinstance(raw, torch.Tensor):
        raise ValueError("Unsupported GDN canonical state geometry")
    payload = sum(geometry.component_bytes)
    if (
        raw.dtype != torch.int8
        or tuple(raw.shape) != (num_blocks, 1, 1, payload)
        or raw.stride(0) != spec.page_size_bytes
        or raw.stride(-1) != 1
        or raw.storage_offset() % 4
        or not 0 <= rank < tp_size
    ):
        raise ValueError("Unsupported GDN raw page shape, stride or alignment")
    runs = []
    local = canonical = 0
    for size in geometry.component_bytes:
        runs.append(CopyRun(local, canonical + rank * size, size, 1, size, size))
        local += size
        canonical += tp_size * size
    return CanonicalPageMapping(canonical, payload, tuple(runs), 1, 0, True)


def resolve_gdn_format(
    config: "VllmConfig",
    cache: KVCacheConfig,
) -> tuple[RecurrentSchema, CanonicalStorageLayout] | None:
    """Certify the whole hybrid and regular allocation, never one layer alone."""
    p = config.parallel_config
    tp = p.tensor_parallel_size
    groups = cache.kv_cache_groups
    if not any(
        isinstance(spec, MambaSpec)
        for group in groups
        for spec in iter_layer_specs(group.kv_cache_spec)
    ):
        return None
    if (
        config.use_v2_model_runner
        or p.pipeline_parallel_size != 1
        or p.prefill_context_parallel_size != 1
        or p.decode_context_parallel_size != 1
        or p.world_size != tp
        or p.data_parallel_size != 1
        or p.nnodes_within_dp != 1
        or p.distributed_executor_backend != "mp"
        or config.speculative_config is not None
        or config.cache_config.use_replayssm
        or not config.cache_config.enable_prefix_caching
        or cache.hisparse_host_num_blocks is not None
        or cache.num_blocks <= 0
        or not groups
    ):
        return None
    count = len(groups[0].layer_names)
    if not count or len(cache.kv_cache_tensors) != len(groups):
        return None
    names = [name for group in groups for name in group.layer_names]
    if len(set(names)) != len(names):
        return None
    schemas: list[GDNLayerSchema] = []
    canonical_sizes = []
    physical_pages = []
    total_heads = config.model_config.get_total_num_kv_heads()
    attention_dtype = get_kv_cache_torch_dtype(
        config.cache_config.cache_dtype, config.model_config.dtype
    )
    attention_seen = False
    for group, allocation in zip(groups, cache.kv_cache_tensors):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            if set(spec.kv_cache_specs) != set(group.layer_names) or not all(
                value == spec.first_spec for value in spec.kv_cache_specs.values()
            ):
                return None
            spec = spec.first_spec
        if group.host_resident or not group.enable_kv_transfer or group.is_eagle_group:
            return None
        if isinstance(spec, MambaSpec):
            geometry = derive_gdn_geometry(spec, tp)
            if geometry is None:
                return None
            hf = config.model_config.hf_text_config
            declared = (
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_key_head_dim",
                "linear_value_head_dim",
                "linear_conv_kernel_dim",
            )
            if all(type(getattr(hf, field, None)) is int for field in declared):
                q, _, v = geometry.projection_dims
                if (
                    hf.linear_num_key_heads * hf.linear_key_head_dim != q * tp
                    or hf.linear_num_value_heads * hf.linear_value_head_dim != v * tp
                    or hf.linear_num_value_heads != geometry.temporal_shape[0] * tp
                    or (hf.linear_value_head_dim, hf.linear_key_head_dim)
                    != geometry.temporal_shape[1:]
                    or hf.linear_conv_kernel_dim - 1 != geometry.conv_rows
                ):
                    return None
            schemas.extend(geometry.schema(name, tp) for name in group.layer_names)
            canonical_size = sum(geometry.component_bytes) * tp
        elif (
            type(spec) is FullAttentionSpec
            and spec.attention_chunk_size is None
            and spec.sliding_window is None
            and not spec.non_causal
            and spec.kv_quant_mode is KVQuantMode.NONE
            and spec.dtype in (torch.bfloat16, torch.float16)
            and spec.dtype == attention_dtype
            and spec.head_size_v == spec.head_size
            and spec.num_heads == spec.num_kv_heads
            and spec.tokens_per_state == 1
            and spec.state_content_size_bytes
            == 2 * spec.head_size * torch.empty((), dtype=spec.dtype).element_size()
            and total_heads % tp == 0
            and spec.num_kv_heads == total_heads // tp
        ):
            attention_seen = True
            canonical_size = spec.real_page_size_bytes * tp
        else:
            return None
        page = spec.page_size_bytes
        if (
            len(group.layer_names) != count
            or allocation.layers != group.layer_names
            or allocation.offset != 0
            or allocation.host_resident
            or allocation.block_stride != page
            or allocation.layer_stride != cache.num_blocks * page
            or allocation.size != count * cache.num_blocks * page
        ):
            return None
        canonical_sizes.append(canonical_size)
        physical_pages.append(page)
    if not attention_seen or len(set(physical_pages)) != 1:
        return None
    return RecurrentSchema(tuple(schemas)), CanonicalStorageLayout(
        segment_page_sizes=(max(canonical_sizes),) * count,
        group_refs=(tuple(range(count)),) * len(groups),
    )


def validate_storage_layout(
    caches: CanonicalKVCaches, layout: CanonicalStorageLayout
) -> None:
    """Reject live aliases or canonical segments that differ from the namespace."""
    sizes = [0] * len(caches.tensors)
    refs = []
    for group in caches.group_data_refs:
        refs.append(tuple(ref.tensor_idx for ref in group))
        for ref in group:
            if ref.mapping is None or not ref.mapping.parallelism_agnostic:
                raise ValueError("GDN canonical storage has an uncertified layer")
            sizes[ref.tensor_idx] = max(
                sizes[ref.tensor_idx], ref.mapping.canonical_page_size_bytes
            )
    if tuple(sizes) != layout.segment_page_sizes or tuple(refs) != layout.group_refs:
        raise ValueError("GDN canonical storage layout differs from certified plan")
