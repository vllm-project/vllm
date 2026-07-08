# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composite attention backend (standard family).

Pairs a prefill backend and a decode backend behind one attention layer. The
decode backend is authoritative for the KV-cache layout and cudagraph support
(decode-first); the prefill backend must be KV-layout-compatible with it (see
`kv_layouts_compatible`). The KV cache is written once via the layer's external
`unified_kv_cache_update` path, delegated to the decode impl; both sub-impls'
`forward` then only read the shared cache.
"""

from dataclasses import dataclass
from typing import Any, cast

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    QueryLenSupport,
)
from vllm.v1.attention.backends.utils import (
    split_common_attn_metadata,
    split_decodes_and_prefills,
)


@dataclass
class CompositeMetadata:
    """Metadata produced by `CompositeAttentionMetadataBuilder`.

    `decode_metadata` / `prefill_metadata` are the per-role metadata objects
    built by the decode / prefill sub-builders over the corresponding row slice
    (or None when that slice is empty).
    """

    num_decodes: int
    num_prefills: int
    num_decode_tokens: int
    num_prefill_tokens: int
    decode_metadata: Any | None
    prefill_metadata: Any | None


class CompositeAttentionMetadataBuilder(AttentionMetadataBuilder[CompositeMetadata]):
    """Builds `CompositeMetadata` by splitting the batch and delegating each
    role slice to the corresponding sub-builder.

    Subclasses created by `make_composite_backend` set the `prefill_backend`
    and `decode_backend` class attributes.
    """

    prefill_backend: type[AttentionBackend]
    decode_backend: type[AttentionBackend]

    def __init__(
        self,
        kv_cache_spec,
        layer_names,
        vllm_config,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.decode_builder = self.decode_backend.get_builder_cls()(
            kv_cache_spec, layer_names, vllm_config, device
        )
        self.prefill_builder = self.prefill_backend.get_builder_cls()(
            kv_cache_spec, layer_names, vllm_config, device
        )
        # Decode defines what counts as a decode: reuse its resolved threshold
        # (decode-first). `query_len_support` stays sourced from the decode
        # sub-builder directly (see `_split_counts`) since it is a ClassVar.
        self.reorder_batch_threshold = self.decode_builder.reorder_batch_threshold
        self.supports_update_block_table = (
            self.decode_builder.supports_update_block_table
            and self.prefill_builder.supports_update_block_table
        )

    @classmethod
    def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
        # Cudagraph support follows the decode backend; prefill stays eager.
        return cls.decode_backend.get_builder_cls().get_cudagraph_support(
            vllm_config, kv_cache_spec
        )

    def _split_counts(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> tuple[int, int, int, int]:
        return split_decodes_and_prefills(
            common_attn_metadata,
            decode_threshold=self.reorder_batch_threshold or 1,
            require_uniform=(
                self.decode_builder.query_len_support != QueryLenSupport.VARLEN
            ),
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompositeMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            self._split_counts(common_attn_metadata)
        )
        decode_cm, prefill_cm = split_common_attn_metadata(
            common_attn_metadata, num_decodes, num_decode_tokens
        )
        # common_prefix_len (cascade attention) is not propagated: the composite
        # disables cascade (see use_cascade_attention).
        decode_metadata = (
            self.decode_builder.build(0, decode_cm, fast_build)
            if num_decodes > 0
            else None
        )
        prefill_metadata = (
            self.prefill_builder.build(0, prefill_cm, fast_build)
            if num_prefills > 0
            else None
        )
        return CompositeMetadata(
            num_decodes=num_decodes,
            num_prefills=num_prefills,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            decode_metadata=decode_metadata,
            prefill_metadata=prefill_metadata,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> CompositeMetadata:
        # Full cudagraph capture is decode-only (mirrors MLA). Assert no prefills
        # then build only the decode metadata via the decode sub-builder.
        _, num_prefills, _, num_prefill_tokens = self._split_counts(
            common_attn_metadata
        )
        assert num_prefills == 0, (
            "Composite backend only supports decode-only full CUDAGraph capture."
        )
        decode_metadata = self.decode_builder.build_for_cudagraph_capture(
            common_attn_metadata
        )
        return CompositeMetadata(
            num_decodes=common_attn_metadata.num_reqs,
            num_prefills=0,
            num_decode_tokens=common_attn_metadata.num_actual_tokens,
            num_prefill_tokens=0,
            decode_metadata=decode_metadata,
            prefill_metadata=None,
        )

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        # Cascade attention spans the whole batch and is not modeled per-role.
        return False


class CompositeAttentionImpl(AttentionImpl):
    """Runs the decode rows through the decode impl and the prefill rows through
    the prefill impl, writing into the shared output. KV is written once by the
    decode impl via `do_kv_cache_update`.

    Subclasses created by `make_composite_backend` set the `prefill_backend`
    and `decode_backend` class attributes.
    """

    prefill_backend: type[AttentionBackend]
    decode_backend: type[AttentionBackend]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = "decoder",
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.decode_impl = cast(
            AttentionImpl,
            self.decode_backend.get_impl_cls()(
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
                **kwargs,
            ),
        )
        self.prefill_impl = cast(
            AttentionImpl,
            self.prefill_backend.get_impl_cls()(
                num_heads,
                head_size,
                scale,
                num_kv_heads,
                alibi_slopes,
                sliding_window,
                kv_cache_dtype,
                logits_soft_cap,
                attn_type,
                kv_sharing_target_layer_name,
                **kwargs,
            ),
        )
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.kv_cache_dtype = kv_cache_dtype
        # Only fuse when both sub-impls agree.
        self.supports_quant_query_input = (
            self.decode_impl.supports_quant_query_input
            and self.prefill_impl.supports_quant_query_input
        )

    def fused_output_quant_supported(self, quant_key) -> bool:
        return self.decode_impl.fused_output_quant_supported(
            quant_key
        ) and self.prefill_impl.fused_output_quant_supported(quant_key)

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Decode owns the KV layout; it is the canonical writer for the whole
        # batch (decodes + prefills share one physical cache).
        self.decode_impl.do_kv_cache_update(  # type: ignore[attr-defined]
            layer, key, value, kv_cache, slot_mapping
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CompositeMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return output

        d = attn_metadata.num_decode_tokens

        def _slice(x, start, end):
            return x[start:end] if x is not None else x

        if attn_metadata.num_decode_tokens > 0:
            self.decode_impl.forward(
                layer,
                query[:d],
                _slice(key, 0, d),
                _slice(value, 0, d),
                kv_cache,
                attn_metadata.decode_metadata,
                output[:d],
                output_scale=_slice(output_scale, 0, d)
                if isinstance(output_scale, torch.Tensor)
                else output_scale,
                output_block_scale=output_block_scale,
            )
        if attn_metadata.num_prefill_tokens > 0:
            self.prefill_impl.forward(
                layer,
                query[d:],
                _slice(key, d, None),
                _slice(value, d, None),
                kv_cache,
                attn_metadata.prefill_metadata,
                output[d:],
                output_scale=_slice(output_scale, d, None)
                if isinstance(output_scale, torch.Tensor)
                else output_scale,
                output_block_scale=output_block_scale,
            )
        return output


class _CompositeBackendMixin:
    """Mixin providing composite overrides. Mixed ahead of the decode backend so
    the composite inherits the decode backend's KV-layout / capability methods
    (decode-first) while overriding identity, dispatch, and prefill-sourced
    capabilities."""

    prefill_backend: type[AttentionBackend]
    decode_backend: type[AttentionBackend]
    _composite_builder_cls: type[CompositeAttentionMetadataBuilder]
    _composite_impl_cls: type[CompositeAttentionImpl]

    # Composite writes KV once via the layer's external update path.
    forward_includes_kv_cache_update = False

    @classmethod
    def is_composite(cls) -> bool:
        return True

    @classmethod
    def get_name(cls) -> str:
        return (
            f"COMPOSITE[{cls.prefill_backend.get_name()}"
            f"|{cls.decode_backend.get_name()}]"
        )

    @classmethod
    def get_builder_cls(cls):
        return cls._composite_builder_cls

    @classmethod
    def get_impl_cls(cls):
        return cls._composite_impl_cls

    # Prefill-sourced capabilities: the composite exists precisely so the prefill
    # backend can offer a capability the decode backend may lack (the feature is
    # only needed during prefill).
    @classmethod
    def supports_non_causal(cls) -> bool:
        return cls.prefill_backend.supports_non_causal()

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return cls.prefill_backend.supports_mm_prefix()

    # Cross-role capabilities: require agreement from both sub-backends.
    @classmethod
    def supports_kv_connector(cls) -> bool:
        return (
            cls.prefill_backend.supports_kv_connector()
            and cls.decode_backend.supports_kv_connector()
        )

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return cls.prefill_backend.supports_head_size(
            head_size
        ) and cls.decode_backend.supports_head_size(head_size)


def make_composite_backend(
    prefill_backend: type[AttentionBackend],
    decode_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    """Create a composite backend pairing `prefill_backend` with `decode_backend`.

    The returned class subclasses the decode backend (decode-first: it owns the
    KV-cache shape/stride/layout/block-size methods) and overrides identity and
    dispatch to route prefill and decode rows to their respective sub-impls.
    """
    assert prefill_backend.is_mla() == decode_backend.is_mla(), (
        "Composite backends must pair within one attention family"
    )
    role_attrs = {
        "prefill_backend": prefill_backend,
        "decode_backend": decode_backend,
    }
    builder_cls = type(
        f"Composite{prefill_backend.__name__}{decode_backend.__name__}Builder",
        (CompositeAttentionMetadataBuilder,),
        dict(role_attrs),
    )
    impl_cls = type(
        f"Composite{prefill_backend.__name__}{decode_backend.__name__}Impl",
        (CompositeAttentionImpl,),
        dict(role_attrs),
    )
    backend_namespace = dict(role_attrs)
    backend_namespace["_composite_builder_cls"] = builder_cls
    backend_namespace["_composite_impl_cls"] = impl_cls
    backend_cls = type(
        f"Composite{prefill_backend.__name__}{decode_backend.__name__}Backend",
        (_CompositeBackendMixin, decode_backend),
        backend_namespace,
    )
    return backend_cls
