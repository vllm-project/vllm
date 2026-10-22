# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper and B300 sparse attention using FlashInfer and Triton."""

import functools
from dataclasses import replace
from typing import Any, ClassVar, NamedTuple, cast

import numpy as np
import torch

from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType, MultipleOf
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache
from vllm.v1.kv_cache_layout import KVCacheLayout

from .ops.attention_gate import headwise_sigmoid_gate
from .ops.sparse_prefill import make_prefill_chunks
from .ops.sparse_prefill_pages import (
    kv_page_stride,
    plan_staged_prefill_pages,
    stage_prefill_pages,
)
from .ops.summary_update import build_prefill_groups

# Transient dense BF16 copy of a prefill batch's live pages for one KV group.
_PREFILL_STAGING_BYTES = 512 << 20
_PRIMS_TS_CAPABILITIES = (
    DeviceCapability(10, 0),
    DeviceCapability(10, 3),
    DeviceCapability(10, 7),
)


class _PrefillLayout(NamedTuple):
    heads_per_group: int
    head_size: int
    groups: int
    dtype: torch.dtype
    quantized: bool
    block_size: int
    # Eight-token pages between KV blocks, which may be strided.
    page_stride: int


def _upload(
    values: np.ndarray, device: torch.device, *, dtype=np.int64
) -> torch.Tensor:
    return (
        torch.from_numpy(values.astype(dtype))
        .pin_memory()
        .to(device, non_blocking=True)
    )


@functools.cache
def _num_sms(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


@functools.cache
def _shared_adapter(device: torch.device):
    from .flashinfer_sparse import Step5FlashInferSparse

    return Step5FlashInferSparse(device)


@functools.cache
def _shared_prims_ts_adapter(device: torch.device, max_model_len: int, topk: int):
    from .prims_ts import Step5PrimsTSSparse

    return Step5PrimsTSSparse(device, max_model_len, topk)


class Step5SplitKVFlashAttentionImpl(FlashAttentionImpl):
    """FlashAttention using the canonical packed-content KV cache."""

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        *args,
        **kwargs,
    ):
        config = get_current_vllm_config_or_none()
        self.use_prims_ts = (
            config is not None
            and config.kernel_config.step5_attention_backend == "prims_ts"
        )
        # FA2 validates the BF16 path; PrimTS handles all E4M3 attention.
        parent_cache_dtype = (
            "auto"
            if self.use_prims_ts and kv_cache_dtype in ("fp8", "fp8_e4m3")
            else kv_cache_dtype
        )
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            parent_cache_dtype,
            *args,
            **kwargs,
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.num_speculative_tokens = (
            config.speculative_config.num_speculative_tokens
            if config is not None and config.speculative_config is not None
            else 0
        )
        if self.use_prims_ts:
            from .prims_ts import Step5PrimsTSSWA

            self.prims_ts_swa = Step5PrimsTSSWA(
                cast(VllmConfig, config).model_config.max_model_len,
                self.sliding_window[0] + 1,
            )
            self.supports_quant_query_input = False

    def should_use_prims_ts_for_batch(self, metadata) -> bool:
        if not self.use_prims_ts:
            return False
        from .prims_ts import uses_prims_ts_attention

        return uses_prims_ts_attention(
            metadata, self.num_speculative_tokens, self.kv_cache_dtype
        )

    def _split_kv_cache(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

    def supports_fused_swa_decode(self) -> bool:
        return self.vllm_flash_attn_version == 2 and self.head_size == 192

    def forward_with_headwise_gate(
        self, layer, query, key, value, kv_cache, attn_metadata, output, gate
    ):
        if self.should_use_prims_ts_for_batch(attn_metadata):
            from .prims_ts import get_kv_cache_and_scales

            cache, qk_scale, v_scale = get_kv_cache_and_scales(query, kv_cache, layer)
            return self.prims_ts_swa.run(
                query,
                cache,
                attn_metadata,
                output,
                gate,
                self.scale * qk_scale,
                v_scale,
            )
        if gate is None:
            return self.forward(
                layer, query, key, value, kv_cache, attn_metadata, output
            )
        if (
            attn_metadata is None
            or not attn_metadata.num_actual_tokens
            or self.dcp_world_size != 1
            or self.attn_type != AttentionType.DECODER
            or attn_metadata.causal is not True
            or query.dtype != torch.bfloat16
            or gate.dtype != torch.bfloat16
            or attn_metadata.max_num_splits > 1
            or attn_metadata.max_seq_len <= 0
            or is_quantized_kv_cache(self.kv_cache_dtype)
            or self.alibi_slopes is not None
            or self.logits_soft_cap != 0
            or self.sliding_window is None
            or self.sliding_window[0] < 0
            or self.sliding_window[1] != 0
            or self.head_size != 192
            or self.num_heads // self.num_kv_heads > 16
            or attn_metadata.max_query_len > 4
            or attn_metadata.num_actual_tokens
            != attn_metadata.seq_lens.shape[0] * attn_metadata.max_query_len
        ):
            self.forward(layer, query, key, value, kv_cache, attn_metadata, output)
            gated = headwise_sigmoid_gate(output.flatten(1), gate)
            return output.copy_(gated.view_as(output))

        n = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        requests = attn_metadata.seq_lens.shape[0]
        query_len = attn_metadata.max_query_len
        # Uniform decode and MTP verify: GQA-packed split-window kernel.
        from .ops.swa_decode import swa_decode_gated, swa_decode_plan

        window = self.sliding_window[0] + 1
        pack, splits = swa_decode_plan(
            n,
            self.num_kv_heads,
            self.num_heads // self.num_kv_heads,
            query_len,
            _num_sms(query.device),
            window,
        )
        swa_decode_gated(
            query[:n],
            key_cache,
            value_cache,
            attn_metadata.block_table[:requests],
            attn_metadata.seq_lens,
            gate[:n],
            output[:n],
            scale=self.scale,
            window=window,
            query_len=query_len,
            splits=splits,
            pack=pack,
        )
        return output


class Step5PrimsTSSWAImpl(Step5SplitKVFlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.sliding_window[0] < 0 or self.sliding_window[1] != 0:
            raise NotImplementedError("Step5 PrimTS SWA requires a causal window")


def _validate_step5_dsa_attention_contract(
    *,
    dtype: torch.dtype,
    head_size: int,
    q_heads_per_kv: int,
    alibi_slopes: list[float] | torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    attn_type: str,
    kv_sharing_target_layer_name: str | None,
    sinks: torch.Tensor | None,
) -> None:
    if dtype != torch.bfloat16:
        raise NotImplementedError(
            f"Step5 DSA kernels require bfloat16 activations, got {dtype}."
        )
    if head_size != 192:
        raise NotImplementedError(
            f"Step5 DSA kernels require head_size 192, got {head_size}."
        )
    if q_heads_per_kv not in (4, 8, 16):
        raise ValueError(
            "Step5 DSA kernels require q_heads_per_kv in {4, 8, 16}, got "
            f"{q_heads_per_kv}."
        )
    if alibi_slopes is not None:
        raise NotImplementedError("Step5 DSA does not support ALiBi.")
    if sliding_window != (-1, -1):
        raise NotImplementedError(
            "Step5 DSA does not support sliding-window attention."
        )
    if logits_soft_cap != 0:
        raise NotImplementedError("Step5 DSA does not support logits soft cap.")
    if attn_type != AttentionType.DECODER:
        raise NotImplementedError(
            "Step5 DSA supports causal decoder self-attention only, got "
            f"attn_type={attn_type!r}."
        )
    if kv_sharing_target_layer_name is not None:
        raise NotImplementedError("Step5 DSA does not support KV cache sharing.")
    if sinks is not None:
        raise NotImplementedError("Step5 DSA does not support attention sinks.")


class Step5FlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        step5_metadata = cast(Any, metadata)
        step5_metadata.num_actual_reqs = common_attn_metadata.num_reqs
        step5_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        return metadata


class Step5DSAMetadataBuilder(Step5FlashAttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    supports_update_block_table = False

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        # MTP verify batches take the multi-token decode path.
        if vllm_config.speculative_config is not None:
            return AttentionCGSupport.UNIFORM_BATCH
        return cls._cudagraph_support

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)
        step5_metadata = cast(Any, metadata)
        step5_metadata.step5_runtime = None
        self._plan_prefill(step5_metadata, common_attn_metadata)
        return metadata

    def _layer(self):
        if not self.layer_names:
            return None
        context = self.vllm_config.compilation_config.static_forward_context
        return context.get(self.layer_names[0])

    def _plan_prefill(self, metadata, common) -> None:
        """Plan prefill steps here, before the forward, so the forward has no
        host sync: host lengths come from the scheduler and exact positions
        stay on device. Uniform (decode and MTP verify) batches need no plan.
        """
        layer = self._layer()
        impl = getattr(layer, "impl", None)
        n, reqs, query_len = (
            common.num_actual_tokens,
            common.num_reqs,
            common.max_query_len,
        )
        if (
            impl is None
            or impl.adapter is None
            or impl.summary_cache is None
            or not n
            or common.seq_lens_cpu_upper_bound is None
            or (n == reqs * query_len and query_len <= impl.num_speculative_tokens + 1)
        ):
            return
        # Upper bounds are exact without speculative decoding.
        positions = common.positions[:n] if impl.num_speculative_tokens else None
        if impl.num_speculative_tokens and common.positions is None:
            return
        spec = cast(AttentionSpec, self.kv_cache_spec)
        metadata.step5_runtime = impl.plan_prefill_step(
            metadata,
            n,
            common.seq_lens_cpu_upper_bound[:reqs].numpy(),
            _PrefillLayout(
                impl.num_heads // spec.num_kv_heads,
                spec.head_size,
                spec.num_kv_heads,
                cast(torch.dtype, self.vllm_config.model_config.dtype),
                is_quantized_kv_cache(impl.kv_cache_dtype),
                spec.block_size,
                kv_page_stride(layer.kv_cache),
            ),
            positions,
        )


class Step5DSAAttentionImpl(Step5SplitKVFlashAttentionImpl):
    num_speculative_tokens = 0
    # Real sparse-indexer heads per provider group; 0 disables row pairing.
    indexer_heads = 0
    can_return_lse_for_decode = False
    supports_dcp = False

    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        *args,
        sparse_config=None,
        max_model_len=None,
        max_num_seqs=None,
        num_speculative_tokens=0,
        **kwargs,
    ):
        # Sparse attention consumes model-dtype KV gathered from the cache.
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            "auto",
            *args,
            **kwargs,
        )
        self.kv_cache_dtype = kv_cache_dtype
        config = get_current_vllm_config_or_none()
        capability = current_platform.get_device_capability()
        if (
            capability is None
            or not Step5DSAAttentionBackend.supports_compute_capability(capability)
        ):
            raise NotImplementedError(
                f"Step5 {'PrimTS' if self.use_prims_ts else 'native'} DSA "
                f"does not support GPU capability {capability}"
            )
        if config is not None:
            # FlashInfer builds host-side ragged plans before CUDA graph capture.
            config.compilation_config.cudagraph_num_of_warmups = max(
                1, config.compilation_config.cudagraph_num_of_warmups
            )
            if config.parallel_config.decode_context_parallel_size > 1:
                raise NotImplementedError(
                    "Step5 does not support decode context parallelism"
                )
            if config.parallel_config.use_ubatching:
                raise NotImplementedError("Step5 does not support DBO/ubatching")
        self.num_speculative_tokens = (
            config.speculative_config.num_speculative_tokens
            if config is not None and config.speculative_config is not None
            else num_speculative_tokens
        )
        if is_quantized_kv_cache(self.kv_cache_dtype) and self.kv_cache_dtype not in (
            "fp8",
            "fp8_e4m3",
        ):
            raise NotImplementedError("Step5 DSA supports E4M3 FP8 KV cache only")
        if self.kv_cache_dtype == "float16":
            raise NotImplementedError("Step5 DSA does not support a float16 KV cache")
        _validate_step5_dsa_attention_contract(
            dtype=torch.get_default_dtype(),
            head_size=self.head_size,
            q_heads_per_kv=self.num_queries_per_kv,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            attn_type=self.attn_type,
            kv_sharing_target_layer_name=self.kv_sharing_target_layer_name,
            sinks=self.sinks,
        )
        self.sparse_topk = sparse_config.topk
        self.indexer_heads = (
            sparse_config.sparse_indexer_num_heads // sparse_config.index_tp_size
        )
        self.max_num_seqs = max_num_seqs or 1
        self.max_num_regions = ((max_model_len or 1048576) + 7) // 8
        # Created with the model so memory profiling counts its workspace.
        device = torch.device(
            current_platform.device_type, torch.accelerator.current_device_index()
        )
        self.adapter = (
            _shared_prims_ts_adapter(device, max_model_len or 1048576, self.sparse_topk)
            if self.use_prims_ts
            else _shared_adapter(device)
        )
        self.summary_cache = None
        self.csa_ring = None

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        dsa_proxy_query=None,
        dsa_proxy_key=None,
        dsa_proxy_weights=None,
        dsa_proxy_z=None,
        dsa_summary_updated=False,
        defer_inactive_clear=False,
        dsa_gate=None,
    ):
        """``dsa_gate`` ([tokens, heads]) returns headwise-gated output."""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Step5 does not support output quantization")
        if output is None:
            output = torch.empty_like(query, dtype=torch.get_default_dtype())
        if attn_metadata is None:
            return output.zero_()
        assert self.summary_cache is not None and self.adapter is not None
        n = attn_metadata.num_actual_tokens
        if n < output.shape[0]:
            output[n:].zero_()
        if not n:
            return output
        # Uniform batches take the decode kernels, including MTP verify where
        # each request scores k + 1 tokens as independent rows. That path has
        # no host metadata, so it is FULL CUDA graph capturable.
        query_len = attn_metadata.max_query_len
        is_decode = n == attn_metadata.num_actual_reqs * query_len and (
            query_len <= self.num_speculative_tokens + 1
        )
        prims_ts = self.should_use_prims_ts_for_batch(attn_metadata)
        if (dsa_summary_updated or defer_inactive_clear) and not is_decode:
            raise ValueError("Decode attention fusions cannot be used for prefill")
        # Both prefill paths gate the finished output. Native decode fuses
        # the gate; the PrimTS adapter applies it after attention.
        if dsa_gate is not None and not is_decode:
            self.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                dsa_proxy_query=dsa_proxy_query,
                dsa_proxy_key=dsa_proxy_key,
                dsa_proxy_weights=dsa_proxy_weights,
                dsa_proxy_z=dsa_proxy_z,
                dsa_summary_updated=dsa_summary_updated,
                defer_inactive_clear=defer_inactive_clear,
            )
            gated = headwise_sigmoid_gate(output[:n].flatten(1), dsa_gate[:n])
            output[:n].copy_(gated.view_as(output[:n]))
            return output
        if prims_ts:
            from .prims_ts import get_kv_cache_and_scales

            kv_cache, qk_scale, v_scale = get_kv_cache_and_scales(
                query, kv_cache, layer
            )
            kv_scales = (qk_scale, v_scale)
        else:
            kv_scales = (
                (layer._k_scale, layer._v_scale)
                if kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)
                else (None, None)
            )
        groups = kv_cache.shape[1]
        if not is_decode and attn_metadata.step5_runtime is None:
            # Fallback when the builder could not plan ahead: one host sync.
            count = attn_metadata.num_actual_reqs
            attn_metadata.step5_runtime = self.plan_prefill_step(
                attn_metadata,
                n,
                attn_metadata.seq_lens[:count].cpu().numpy(),
                _PrefillLayout(
                    query.shape[1] // groups,
                    query.shape[2],
                    groups,
                    query.dtype,
                    kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn),
                    kv_cache.shape[2],
                    kv_page_stride(kv_cache),
                ),
            )
        if prims_ts and not is_decode:
            plan = attn_metadata.step5_runtime[5]
            plan.bind(
                query[:n],
                kv_cache,
                attn_metadata.block_table[: attn_metadata.num_actual_reqs],
                output[:n],
                owners=attn_metadata.step5_runtime[0],
                positions=attn_metadata.step5_runtime[1],
                qk_scale=kv_scales[0],
                v_scale=kv_scales[1],
            )
        if groups > 1 and is_decode:
            # Fold (token, KV group) into rows: one launch chain for all groups.
            rows, heads, dim = n * groups, query.shape[1] // groups, query.shape[2]
            self._forward_decode(
                query[:n].view(rows, heads, dim),
                kv_cache,
                attn_metadata,
                output[:n].view(rows, heads, dim),
                # Indexer outputs may be group-major; kernels read their strides.
                dsa_proxy_query[:n],
                dsa_proxy_key[:n],
                dsa_proxy_weights[:n],
                dsa_proxy_z[:n],
                summary_updated=dsa_summary_updated,
                defer_inactive_clear=defer_inactive_clear,
                kv_scales=kv_scales,
                query_len=query_len,
                gate=dsa_gate,
            )
            return output
        if groups > 1:
            heads = query.shape[1] // groups
            # Native FA2 needs contiguous per-group output. PrimTS collects
            # all groups' selections, then writes directly into the bound
            # full output, avoiding temporary group buffers and the final copy.
            grouped_output = (
                None
                if prims_ts
                else torch.empty(
                    (groups, n, heads, query.shape[2]),
                    device=output.device,
                    dtype=output.dtype,
                )
            )
            for group in range(groups):
                group_query = query[:n, group * heads : (group + 1) * heads]
                group_output = (
                    output[:n, group * heads : (group + 1) * heads]
                    if prims_ts
                    else cast(torch.Tensor, grouped_output)[group]
                )
                args = (
                    group_query,
                    kv_cache[:, group : group + 1],
                    attn_metadata,
                    group_output,
                    dsa_proxy_query[:n, group : group + 1],
                    dsa_proxy_key[:n],
                    dsa_proxy_weights[:n, group : group + 1],
                    dsa_proxy_z[:n],
                )
                self._forward_prefill(
                    *args, summary_updated=group > 0, kv_scales=kv_scales
                )
            if grouped_output is not None:
                output[:n].view(n, groups, heads, query.shape[2]).copy_(
                    grouped_output.transpose(0, 1)
                )
            return output
        if is_decode:
            return self._forward_decode(
                query[:n],
                kv_cache,
                attn_metadata,
                output[:n],
                dsa_proxy_query[:n],
                dsa_proxy_key[:n],
                dsa_proxy_weights[:n],
                dsa_proxy_z[:n],
                summary_updated=dsa_summary_updated,
                defer_inactive_clear=defer_inactive_clear,
                kv_scales=kv_scales,
                query_len=query_len,
                gate=dsa_gate,
            )
        return self._forward_prefill(
            query[:n],
            kv_cache,
            attn_metadata,
            output,
            dsa_proxy_query[:n],
            dsa_proxy_key[:n],
            dsa_proxy_weights[:n],
            dsa_proxy_z[:n],
            kv_scales=kv_scales,
        )

    def plan_prefill_step(self, attn_metadata, n, lengths, layout, positions=None):
        """Layer-invariant prefill metadata, built once per step.

        ``lengths`` are host sequence lengths. Without ``positions`` they must
        be exact; with exact device ``positions`` they may be upper bounds
        (MRv2 speculative decoding), which is what lets the metadata builder
        plan before the forward without a host sync. Returns None if only the
        ragged fallback applies and the lengths are not exact. ``layout`` is a
        ``_PrefillLayout`` of the full (all-group) query and KV cache.
        """
        assert self.summary_cache is not None and self.adapter is not None
        exact = positions is None
        device = attn_metadata.slot_mapping.device
        count = attn_metadata.num_actual_reqs
        starts = attn_metadata.query_start_loc_cpu[: count + 1].numpy()
        query_lens = np.diff(starts)
        host_positions = np.arange(n) + np.repeat(
            lengths - query_lens - starts[:-1], query_lens
        )
        owners = _upload(
            np.repeat(np.arange(count), query_lens),
            device,
            dtype=np.int32 if self.use_prims_ts else np.int64,
        )
        if positions is None:
            positions = _upload(host_positions, device)
        table = attn_metadata.block_table[:count]
        num_regions = (int(lengths.max()) + 7) // 8
        groups, block_size = layout.groups, layout.block_size
        # FP8 and multi-group (strided) caches cannot feed the paged FA2
        # kernels directly; stage live pages into dense BF16 per layer
        # instead of gathering ~4K tokens of KV for every query row.
        staging = (layout.quantized or groups > 1) and not self.use_prims_ts
        staged_pages = int(((lengths + block_size - 1) // block_size).sum())
        paged_plan = None
        if not staging or (
            staged_pages * block_size * layout.head_size * 4 <= _PREFILL_STAGING_BYTES
        ):
            query_dtype = (
                torch.float8_e4m3fn
                if self.use_prims_ts and layout.quantized
                else layout.dtype
            )
            paged_plan = self.adapter.prepare_prefill_attention(
                (n, layout.heads_per_group, layout.head_size),
                query_dtype,
                device,
                block_size,
                host_positions,
                min(num_regions, self.sparse_topk) + 1,
                self.scale,
                contiguous_query=groups == 1,
                request_offsets=starts,
                num_kv_groups=groups,
            )
        if paged_plan is None and not exact:
            return None
        if paged_plan is not None and not exact:
            paged_plan.fixup_positions(positions)
        staged = (
            plan_staged_prefill_pages(table, lengths.tolist(), block_size)
            if paged_plan is not None and staging
            else None
        )
        attn_table = table if staged is None else staged.table
        # Staged pages are dense.
        page_stride = layout.page_stride if staged is None else block_size // 8
        chunks = (
            make_prefill_chunks(host_positions.tolist(), self.sparse_topk, device)
            if paged_plan is None
            else None
        )
        summary_plan = build_prefill_groups(
            attn_metadata.slot_mapping[:n],
            positions,
            owners,
            attn_metadata.query_start_loc[: count + 1],
        )
        indexer = self.adapter.prefill_metadata(
            owners,
            positions,
            num_regions,
            self.sparse_topk,
            paged_prefill=(
                paged_plan.page_metadata(attn_table, owners, page_stride)
                if paged_plan is not None and num_regions <= self.sparse_topk
                else None
            ),
            out=cast(Any, paged_plan).ids if self.use_prims_ts else None,
            history_only=self.use_prims_ts,
        )
        # Pairs of consecutive rows of one request fill the logits MMA with
        # their real heads instead of zero-padding each row to eight heads.
        paired = (
            self.adapter.prefill_pair_inputs(indexer)
            if num_regions > self.sparse_topk
            and self.indexer_heads * 2 == 8
            and not (query_lens % 2).any()
            else None
        )
        return (
            owners,
            positions,
            num_regions,
            chunks,
            indexer,
            paged_plan,
            summary_plan,
            staged,
            paired,
        )

    def _forward_prefill(
        self,
        query,
        kv_cache,
        attn_metadata,
        output,
        dsa_proxy_query,
        dsa_proxy_key,
        dsa_proxy_weights,
        dsa_proxy_z,
        *,
        summary_updated=False,
        kv_scales=(None, None),
    ):
        assert self.summary_cache is not None and self.adapter is not None
        n = query.shape[0]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Step5 prefill requires a breakable CUDA graph")
        table = attn_metadata.block_table[: attn_metadata.num_actual_reqs]
        (
            owners,
            positions,
            num_regions,
            chunks,
            indexer,
            paged_plan,
            summary_plan,
            staged,
            paired,
        ) = attn_metadata.step5_runtime
        attn_cache, attn_table = kv_cache, table
        if staged is not None:
            attn_cache = stage_prefill_pages(kv_cache, staged, query.dtype, *kv_scales)
            attn_table = staged.table
        if not summary_updated:
            self.summary_cache.update_prefill(
                dsa_proxy_key[:n],
                dsa_proxy_z[:n],
                summary_plan,
                positions,
                self.csa_ring,
            )
        packed_summaries = None
        if num_regions > self.sparse_topk:
            packed_summaries = self.adapter.pack_summaries(
                self.summary_cache.mean, table, num_regions
            )
        # Bound logits and gathered KV scratch independently of prefill length.
        start = 0
        for chunk, indexer_meta in enumerate(indexer):
            end = start + indexer_meta.current.numel()
            current = indexer_meta.current
            if packed_summaries is None:
                selected = indexer_meta.selected
            else:
                packed, pages = packed_summaries
                scores = self.adapter.logits(
                    dsa_proxy_query[start:end],
                    dsa_proxy_weights[start:end],
                    packed,
                    pages,
                    owners[start:end],
                    indexer_meta.lengths,
                    num_regions,
                    block_table=indexer_meta.block_table,
                    schedule_meta=indexer_meta.schedule,
                    pairs=paired[chunk] if paired is not None else None,
                    paired_heads=self.indexer_heads,
                )
                selected = self.adapter.select_regions(
                    scores,
                    current,
                    self.sparse_topk,
                    topk_lengths=indexer_meta.topk_lengths,
                    paged_prefill=(
                        paged_plan.page_metadata(
                            attn_table, owners, kv_page_stride(attn_cache), start, end
                        )
                        if paged_plan is not None
                        else None
                    ),
                    # Attention is order-independent; skipping the 1K sort
                    # makes selection ~2x cheaper.
                    sort=False,
                    out=paged_plan.selection_output(start, end)
                    if self.use_prims_ts
                    else None,
                    history_only=self.use_prims_ts,
                )
            if paged_plan is not None:
                if paged_plan.query_buffer is not None and not self.use_prims_ts:
                    paged_plan.prepare_chunk(
                        start,
                        query[start:end],
                        attn_cache,
                        selected,
                        attn_table,
                        owners[start:end],
                    )
                start = end
                continue
            self.adapter.run_attention(
                query[start:end],
                kv_cache,
                selected,
                table,
                owners[start:end],
                positions[start:end],
                output[start:end],
                self.scale,
                prefill_chunks=chunks[start // 32 : (end + 31) // 32],
                k_scale=kv_scales[0],
                v_scale=kv_scales[1],
            )
            start = end
        if paged_plan is not None:
            paged_plan.finish(query[:n], attn_cache, output[:n])
        return output

    def _forward_decode(
        self,
        query,
        kv_cache,
        metadata,
        output,
        proxy_query,
        proxy_key,
        proxy_weights,
        proxy_z,
        *,
        summary_updated=False,
        defer_inactive_clear=False,
        kv_scales=(None, None),
        query_len=1,
        gate=None,
    ):
        assert self.summary_cache is not None and self.adapter is not None
        n = proxy_key.shape[0]
        requests = n // query_len
        slots = metadata.slot_mapping[:n]
        table = metadata.block_table[:requests]
        seq_lens = metadata.seq_lens[:requests]
        if self.num_speculative_tokens:
            # Tokens of one request fill their open region in order; a region
            # closed by a rejected draft is summarized again when it recloses.
            assert not summary_updated
            self.summary_cache.update_verify(
                proxy_key,
                proxy_z,
                slots,
                metadata.query_start_loc[: requests + 1],
                seq_lens,
                self.csa_ring,
            )
        elif not summary_updated:
            self.summary_cache.update_decode(
                proxy_key, proxy_z, slots, seq_lens - 1, self.csa_ring
            )
        num_regions = min(self.max_num_regions, table.shape[1] * kv_cache.shape[2] // 8)
        # Rows are (token, KV group): row r is group r % G of token r // G.
        scores, decode = self.adapter.decode_logits(
            proxy_query,
            proxy_weights,
            self.summary_cache.mean,
            table,
            num_regions,
            slots,
            seq_lens,
            row_groups=kv_cache.shape[1],
            query_len=query_len,
        )
        run_decode = self.adapter.select_and_run_decode_attention
        if self.use_prims_ts and not self.should_use_prims_ts_for_batch(metadata):
            run_decode = self.adapter.native_decode_attention
        run_decode(
            query,
            kv_cache,
            scores,
            decode.current,
            self.sparse_topk,
            table,
            decode.positions,
            decode.active,
            output,
            self.scale,
            seq_lens=seq_lens,
            topk_lengths=decode.topk_lengths,
            clear_inactive=not defer_inactive_clear,
            k_scale=kv_scales[0],
            v_scale=kv_scales[1],
            query_len=query_len,
            gate=gate,
        )
        return output


class Step5SplitKVFlashAttentionBackend(FlashAttentionBackend):
    """``FLASH_ATTN`` with Step5's fused SWA decode for the non-DSA layers.

    Keeps the inherited ``get_name()``: only the impl differs from the parent,
    so these layers resolve to the registered ``FLASH_ATTN`` entry.
    """

    @staticmethod
    def get_impl_cls() -> type[FlashAttentionImpl]:
        return Step5SplitKVFlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        config = get_current_vllm_config_or_none()
        if (
            config is not None
            and config.kernel_config.step5_attention_backend == "prims_ts"
        ):
            return Step5FlashAttentionMetadataBuilder
        return FlashAttentionMetadataBuilder


class Step5PrimsTSSWABackend(Step5SplitKVFlashAttentionBackend):
    @staticmethod
    def get_impl_cls():
        return Step5PrimsTSSWAImpl

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return kv_cache_dtype in (None, "auto", "bfloat16", "fp8", "fp8_e4m3")

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        if (
            head_size != 192
            or dtype != torch.bfloat16
            or device_capability not in _PRIMS_TS_CAPABILITIES
            or use_mla
            or has_sink
            or use_mm_prefix
        ):
            return (
                "Step5 PrimTS SWA requires BF16 D192 on SM100/103/107 "
                "without sinks or MLA"
            )
        return super().supports_combination(
            head_size,
            dtype,
            "auto",
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )


class Step5DSAAttentionBackend(Step5SplitKVFlashAttentionBackend):
    """Backend entry for Step5 GQA + DSA attention."""

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # DSA layers pack their region summaries beside the KV inside each
        # block, so the layer dim must sit inside the block dim.
        return (KVCacheLayout.BLNHC,)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == 192

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        return super().supports_combination(
            head_size,
            dtype,
            "auto",
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return False

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        # Sparse prefill reads eight-token pages, which must tile every block
        # stride.
        return replace(spec, block_stride_alignment=MultipleOf(8))

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        config = get_current_vllm_config_or_none()
        if (
            config is not None
            and config.kernel_config.step5_attention_backend == "prims_ts"
        ):
            return capability in _PRIMS_TS_CAPABILITIES
        return capability in (DeviceCapability(9, 0), DeviceCapability(10, 3))

    @staticmethod
    def get_name() -> str:
        return "STEP5_DSA"

    @staticmethod
    def get_impl_cls() -> type[Step5DSAAttentionImpl]:
        return Step5DSAAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[FlashAttentionMetadataBuilder]:
        return Step5DSAMetadataBuilder
