# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reusable attention backend composition and routing policies."""

import functools
import math
from typing import ClassVar, Protocol, cast

import torch

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
from vllm.v1.kv_cache_interface import KVCacheSpec


def requires_mm_prefix(
    metadata: CommonAttentionMetadata, *, unclamped_window: bool = False
) -> bool:
    ranges = metadata.mm_req_doc_ranges
    # With one query per request there are no future keys to unmask. This
    # also keeps single-token prefills on the captured FlashInfer path.
    if not ranges or (metadata.max_query_len <= 1 and not unclamped_window):
        return False
    seq_lens = metadata.seq_lens_cpu_upper_bound
    assert seq_lens is not None
    starts = metadata.query_start_loc_cpu
    for req_idx, spans in ranges.items():
        query_len = int(starts[req_idx + 1] - starts[req_idx])
        if query_len == 0 or (query_len == 1 and not unclamped_window):
            continue
        end = int(seq_lens[req_idx])
        begin = end - query_len
        # An unclamped image can also expose keys behind the sliding window.
        if unclamped_window and any(
            start < stop and start < end and stop >= begin for start, stop in spans
        ):
            return True
        # Prefill lengths are exact. Historical image ranges must not divert
        # later text/decode queries to the image backend.
        if any(
            start < stop and start < end - 1 and stop > begin for start, stop in spans
        ):
            return True
    return False


def _has_unclamped_window(layers) -> bool:
    return any(
        getattr(layer, "sliding_window", None) is not None
        and not getattr(layer, "mm_prefix_clamp_sliding_window", False)
        for layer in layers
    )


class CompositeAttentionRouting(Protocol):
    capture_variant: ClassVar[int]

    def __init__(self, layer_names: list[str], vllm_config: VllmConfig) -> None: ...

    def select(self, metadata: CommonAttentionMetadata) -> int: ...

    @staticmethod
    def get_cudagraph_support(
        vllm_config: VllmConfig, kv_cache_spec: KVCacheSpec
    ) -> AttentionCGSupport: ...

    @staticmethod
    def variant_uses_mm_prefix(variant: int) -> bool: ...


class MMPrefixAttentionRouting:
    """Route image masks to variant zero and causal queries to variant one."""

    def __init__(self, layer_names, vllm_config):
        layers = vllm_config.compilation_config.static_forward_context
        self.unclamped_window = _has_unclamped_window(
            layers[name] for name in layer_names
        )

    def select(self, metadata):
        return int(
            not requires_mm_prefix(metadata, unclamped_window=self.unclamped_window)
        )

    capture_variant: ClassVar[int] = 1

    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec):
        if _has_unclamped_window(
            vllm_config.compilation_config.static_forward_context.values()
        ):
            return AttentionCGSupport.NEVER
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    @staticmethod
    def variant_uses_mm_prefix(variant):
        return variant == 0


def _intersect_kernel_block_sizes(
    first: type[AttentionBackend], second: type[AttentionBackend]
):
    from vllm.v1.attention.backend import MultipleOf

    result: list[int | MultipleOf] = []
    seen: set[tuple[type, int]] = set()
    for lhs in first.get_supported_kernel_block_sizes():
        for rhs in second.get_supported_kernel_block_sizes():
            candidate: int | MultipleOf | None = None
            if isinstance(lhs, MultipleOf) and isinstance(rhs, MultipleOf):
                candidate = MultipleOf(math.lcm(lhs.base, rhs.base))
            elif isinstance(lhs, MultipleOf) and isinstance(rhs, int):
                candidate = rhs if rhs % lhs.base == 0 else None
            elif isinstance(lhs, int) and isinstance(rhs, MultipleOf):
                candidate = lhs if lhs % rhs.base == 0 else None
            elif lhs == rhs:
                candidate = lhs
            if candidate is None:
                continue
            value = candidate.base if isinstance(candidate, MultipleOf) else candidate
            key = (type(candidate), value)
            if key not in seen:
                result.append(candidate)
                seen.add(key)
    return result


@functools.cache
def create_composite_attention_backend(
    general_backend: type[AttentionBackend],
    causal_backend: type[AttentionBackend],
    *,
    name: str,
    module: str,
    routing_policy: type[CompositeAttentionRouting],
    head_sizes: tuple[int, ...] = (),
    kernel_block_sizes: tuple[int, ...] = (),
    device_major: int | None = None,
) -> type[AttentionBackend]:
    """Compose two backends with shared storage and an explicit routing policy."""
    if general_backend.full_cls_name() == causal_backend.full_cls_name():
        return general_backend

    general_impl_cls = general_backend.get_impl_cls()
    causal_impl_cls = causal_backend.get_impl_cls()
    general_builder_cls = general_backend.get_builder_cls()
    causal_builder_cls = causal_backend.get_builder_cls()
    if (
        general_backend.forward_includes_kv_cache_update
        or causal_backend.forward_includes_kv_cache_update
    ):
        raise ValueError("Composite backends require separate KV cache updates")

    class CompositeAttentionImpl(AttentionImpl):
        def __init__(self, *args, **kwargs) -> None:
            self.general_impl = cast(AttentionImpl, general_impl_cls(*args, **kwargs))
            try:
                self.causal_impl = cast(AttentionImpl, causal_impl_cls(*args, **kwargs))
            except TypeError as e:
                raise ValueError(
                    f"Causal attention backend {causal_backend.get_name()} "
                    f"does not accept the attention arguments required by "
                    f"the general backend {general_backend.get_name()}: {e}"
                ) from e
            for name in (
                "num_heads",
                "num_kv_heads",
                "head_size",
                "scale",
                "kv_cache_dtype",
            ):
                if hasattr(self.general_impl, name):
                    setattr(self, name, getattr(self.general_impl, name))
            self.supports_quant_query_input = (
                self.general_impl.supports_quant_query_input
                and self.causal_impl.supports_quant_query_input
            )
            self.supports_pcp = self.general_impl.supports_pcp
            self.supports_dcp = self.causal_impl.supports_dcp
            self.can_return_lse_for_decode = self.causal_impl.can_return_lse_for_decode
            self.lse_base_on_e = self.causal_impl.lse_base_on_e
            self.need_to_return_lse_for_decode = (
                self.causal_impl.need_to_return_lse_for_decode
            )
            self.supports_mtp_with_cp_non_trivial_interleave_size = (
                self.causal_impl.supports_mtp_with_cp_non_trivial_interleave_size
            )

        def get_impl_for_metadata(self, attn_metadata):
            return (
                self.causal_impl
                if getattr(attn_metadata, "_attention_backend_variant", 0)
                else self.general_impl
            )

        def get_impl_variants(self):
            return self.general_impl, self.causal_impl

        def forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=None,
            output_block_scale=None,
        ):
            impl = self.get_impl_for_metadata(attn_metadata)
            return impl.forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        def process_weights_after_loading(self, act_dtype: torch.dtype):
            for impl in self.get_impl_variants():
                impl.process_weights_after_loading(act_dtype)

        def fused_output_quant_supported(self, quant_key):
            return all(
                impl.fused_output_quant_supported(quant_key)
                for impl in self.get_impl_variants()
            )

        def fused_qk_norm_rope_kvcache_supported(self):
            return self.general_impl.fused_qk_norm_rope_kvcache_supported()

        def do_qk_norm_rope_kvcache_update(self, *args, **kwargs):
            method = self.general_impl.do_qk_norm_rope_kvcache_update
            return method(*args, **kwargs)

        def fused_rope_kvcache_supported(self):
            return self.general_impl.fused_rope_kvcache_supported()

        def do_rope_and_kv_cache_update(self, *args, **kwargs):
            method = self.general_impl.do_rope_and_kv_cache_update
            return method(*args, **kwargs)

        def do_kv_cache_update(self, *args, **kwargs):
            method = self.general_impl.do_kv_cache_update  # type: ignore[attr-defined]
            return method(*args, **kwargs)

    class CompositeAttentionMetadataBuilder(AttentionMetadataBuilder):
        requires_block_table_width = (
            general_builder_cls.requires_block_table_width
            or causal_builder_cls.requires_block_table_width
        )

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kwargs):
            super().__init__(kv_cache_spec, layer_names, vllm_config, device)
            args = (kv_cache_spec, layer_names, vllm_config, device)
            # Only the variant that asked for the extra ctor kwargs receives them.
            self.general_builder = general_builder_cls(
                *args,
                **(kwargs if general_builder_cls.requires_block_table_width else {}),
            )
            self.causal_builder = causal_builder_cls(
                *args,
                **(kwargs if causal_builder_cls.requires_block_table_width else {}),
            )
            self.reorder_batch_threshold = min(
                (
                    builder.reorder_batch_threshold
                    for builder in self._builders
                    if builder.reorder_batch_threshold is not None
                ),
                default=None,
            )
            self.routing = routing_policy(layer_names, vllm_config)

        def set_kernel_block_size(self, kernel_block_size):
            super().set_kernel_block_size(kernel_block_size)
            for builder in self._builders:
                builder.set_kernel_block_size(kernel_block_size)

        @classmethod
        def get_cudagraph_support(cls, vllm_config, kv_cache_spec):
            return min(
                general_builder_cls.get_cudagraph_support(vllm_config, kv_cache_spec),
                causal_builder_cls.get_cudagraph_support(vllm_config, kv_cache_spec),
                routing_policy.get_cudagraph_support(vllm_config, kv_cache_spec),
                key=lambda support: support.value,
            )

        def _builder(self, common_attn_metadata):
            backend_variant = self.routing.select(common_attn_metadata)
            builder = self.causal_builder if backend_variant else self.general_builder
            return builder, backend_variant

        def _build(self, method, common_attn_metadata, **kwargs):
            builder, backend_variant = self._builder(common_attn_metadata)
            metadata = getattr(builder, method)(
                common_attn_metadata=common_attn_metadata, **kwargs
            )
            metadata._attention_backend_variant = backend_variant
            return metadata

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return self._build(
                "build",
                common_attn_metadata,
                common_prefix_len=common_prefix_len,
                fast_build=fast_build,
            )

        def build_for_cudagraph_capture(self, common_attn_metadata):
            variant = routing_policy.capture_variant
            metadata = self._builders[variant].build_for_cudagraph_capture(
                common_attn_metadata
            )
            metadata._attention_backend_variant = variant
            return metadata

        @property
        def _builders(self):
            return self.general_builder, self.causal_builder

        # Forward the optional workspace protocol so that a wrapped builder
        # which allocates one (FlashInfer) still joins the runner's cross-group
        # sharing instead of allocating a second buffer behind the composite.
        def _get_workspace_buffer(self):
            for builder in self._builders:
                if hasattr(builder, "_get_workspace_buffer"):
                    workspace = builder._get_workspace_buffer()
                    # The runner only calls the getter for the group that
                    # provides the buffer, so hand it to the sibling here.
                    self.set_workspace_buffer(workspace)
                    return workspace
            return None

        def set_workspace_buffer(self, workspace_buffer):
            for builder in self._builders:
                if hasattr(builder, "set_workspace_buffer"):
                    builder.set_workspace_buffer(workspace_buffer)

    class CompositeAttentionBackend(AttentionBackend):
        forward_includes_kv_cache_update = False
        general_backend_cls = general_backend
        causal_backend_cls = causal_backend

        @staticmethod
        def get_name():
            return name

        @staticmethod
        def get_impl_cls():
            return CompositeAttentionImpl

        @staticmethod
        def get_builder_cls():
            return CompositeAttentionMetadataBuilder

        @staticmethod
        def get_supported_kernel_block_sizes():
            if kernel_block_sizes:
                return [
                    size
                    for size in kernel_block_sizes
                    if all(
                        backend.supports_block_size(size)
                        for backend in (general_backend, causal_backend)
                    )
                ]
            return _intersect_kernel_block_sizes(general_backend, causal_backend)

        @classmethod
        def supports_block_size(cls, block_size):
            return bool(
                cls.get_supported_kernel_block_sizes()
            ) and super().supports_block_size(block_size)

        @classmethod
        def get_supported_head_sizes(cls):
            return list(head_sizes)

        @classmethod
        def supports_head_size(cls, head_size):
            return (not head_sizes or head_size in head_sizes) and all(
                backend.supports_head_size(head_size)
                for backend in (general_backend, causal_backend)
            )

        @classmethod
        def supports_compute_capability(cls, capability):
            return (device_major is None or capability.major == device_major) and all(
                backend.supports_compute_capability(capability)
                for backend in (general_backend, causal_backend)
            )

        @classmethod
        def supports_dtype(cls, dtype):
            return all(
                backend.supports_dtype(dtype)
                for backend in (general_backend, causal_backend)
            )

        @classmethod
        def supports_kv_cache_dtype(cls, dtype):
            return super().supports_kv_cache_dtype(dtype) and all(
                backend.supports_kv_cache_dtype(dtype)
                for backend in (general_backend, causal_backend)
            )

        @classmethod
        def supports_mm_prefix(cls):
            return all(
                not routing_policy.variant_uses_mm_prefix(variant)
                or backend.supports_mm_prefix()
                for variant, backend in enumerate((general_backend, causal_backend))
            )

        @classmethod
        def supports_sliding_window(cls):
            return all(
                backend.supports_sliding_window()
                for backend in (general_backend, causal_backend)
            )

        @classmethod
        def supports_pcp(cls):
            return False

        @classmethod
        def supports_dcp(cls):
            return False

        @classmethod
        def supports_device_cpu_query_lens_mismatch(cls):
            return False

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
            for variant, backend in enumerate((general_backend, causal_backend)):
                reason = backend.supports_combination(
                    head_size,
                    dtype,
                    kv_cache_dtype,
                    block_size,
                    use_mla,
                    has_sink,
                    use_sparse,
                    use_mm_prefix and routing_policy.variant_uses_mm_prefix(variant),
                    device_capability,
                )
                if reason:
                    return f"{backend.get_name()}: {reason}"
            return None

        @classmethod
        def customize_spec(cls, spec):
            general_spec = general_backend.customize_spec(spec)
            if general_spec != causal_backend.customize_spec(spec):
                raise ValueError("Composite backends require the same KV cache spec")
            return general_spec

        @classmethod
        def supported_kv_cache_layouts(cls):
            return tuple(
                get_supported_kv_cache_layouts((general_backend, causal_backend))
            )

    CompositeAttentionBackend.__name__ = name
    CompositeAttentionBackend.__qualname__ = name
    CompositeAttentionBackend.__module__ = module
    return CompositeAttentionBackend
