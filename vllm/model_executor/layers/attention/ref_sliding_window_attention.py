# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
from typing import Any

import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    subclass_attention_backend_with_overrides,
)
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

_USE_DECODE_SLIDING_WINDOW = "_ref_sliding_window_attention_use_window"
_SPLIT_METADATA = "_ref_sliding_window_attention_split_metadata"


def _full_attention_window(window: Any) -> Any:
    if isinstance(window, tuple):
        return (-1, -1)
    if isinstance(window, int):
        return -1
    return None


def _slice_optional_tensor(
    tensor: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor[start:end]


def _slice_output_scale(
    tensor: torch.Tensor | None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if tensor is None or tensor.ndim == 0 or tensor.shape[0] < end:
        return tensor
    return tensor[start:end]


def _slice_causal(
    causal: bool | torch.Tensor,
    start: int,
    end: int,
) -> bool | torch.Tensor:
    if isinstance(causal, torch.Tensor):
        return causal[start:end]
    return causal


def _slice_mm_req_doc_ranges(
    ranges: dict[int, list[tuple[int, int]]] | None,
    req_start: int,
    req_end: int,
) -> dict[int, list[tuple[int, int]]] | None:
    if ranges is None:
        return None
    sliced = {
        req_idx - req_start: req_ranges
        for req_idx, req_ranges in ranges.items()
        if req_start <= req_idx < req_end
    }
    return sliced or None


def _get_seq_lens_cpu(
    common_attn_metadata: CommonAttentionMetadata,
) -> torch.Tensor:
    if common_attn_metadata.seq_lens_cpu_upper_bound is not None:
        return common_attn_metadata.seq_lens_cpu_upper_bound
    if common_attn_metadata._seq_lens_cpu is not None:
        return common_attn_metadata._seq_lens_cpu
    return common_attn_metadata.seq_lens.cpu()


def _get_decode_prefix(
    common_attn_metadata: CommonAttentionMetadata,
    decode_threshold: int,
) -> tuple[int, int]:
    query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    seq_lens_cpu = _get_seq_lens_cpu(common_attn_metadata)
    context_lens_cpu = seq_lens_cpu - query_lens_cpu

    is_decode = (context_lens_cpu > 0) & (query_lens_cpu <= decode_threshold)
    if common_attn_metadata.is_prefilling is not None:
        is_prefilling = common_attn_metadata.is_prefilling.cpu()
        is_decode &= ~is_prefilling

    num_decode_reqs = int(is_decode.sum().item())
    if num_decode_reqs == 0:
        return 0, 0

    if not bool(is_decode[:num_decode_reqs].all().item()) or bool(
        is_decode[num_decode_reqs:].any().item()
    ):
        raise ValueError(
            "RefSlidingWindowAttention mixed prefill/decode support expects "
            "decode requests to be placed before prefill/extend requests. "
            "The metadata builder requests this ordering via reorder_batch."
        )

    num_decode_tokens = int(query_start_loc_cpu[num_decode_reqs].item())
    return num_decode_reqs, num_decode_tokens


def _slice_common_attn_metadata(
    common_attn_metadata: CommonAttentionMetadata,
    req_start: int,
    req_end: int,
    token_start: int,
    token_end: int,
    seq_lens_cpu: torch.Tensor,
) -> CommonAttentionMetadata:
    query_start_loc = (
        common_attn_metadata.query_start_loc[req_start : req_end + 1] - token_start
    )
    query_start_loc_cpu = (
        common_attn_metadata.query_start_loc_cpu[req_start : req_end + 1] - token_start
    )
    req_seq_lens_cpu = seq_lens_cpu[req_start:req_end]
    query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
    num_computed_tokens_cpu = req_seq_lens_cpu - query_lens_cpu

    return common_attn_metadata.replace(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=common_attn_metadata.seq_lens[req_start:req_end],
        num_reqs=req_end - req_start,
        num_actual_tokens=token_end - token_start,
        max_query_len=int(query_lens_cpu.max().item()),
        max_seq_len=int(req_seq_lens_cpu.max().item()),
        block_table_tensor=common_attn_metadata.block_table_tensor[req_start:req_end],
        slot_mapping=common_attn_metadata.slot_mapping[token_start:token_end],
        causal=_slice_causal(common_attn_metadata.causal, req_start, req_end),
        dcp_local_seq_lens=_slice_optional_tensor(
            common_attn_metadata.dcp_local_seq_lens, req_start, req_end
        ),
        dcp_local_seq_lens_cpu=_slice_optional_tensor(
            common_attn_metadata.dcp_local_seq_lens_cpu, req_start, req_end
        ),
        positions=_slice_optional_tensor(
            common_attn_metadata.positions, token_start, token_end
        ),
        is_prefilling=_slice_optional_tensor(
            common_attn_metadata.is_prefilling, req_start, req_end
        ),
        seq_lens_cpu_upper_bound=req_seq_lens_cpu,
        mm_req_doc_ranges=_slice_mm_req_doc_ranges(
            common_attn_metadata.mm_req_doc_ranges, req_start, req_end
        ),
        _seq_lens_cpu=req_seq_lens_cpu,
        _num_computed_tokens_cpu=num_computed_tokens_cpu,
        _num_computed_tokens_cache=None,
    )


@functools.lru_cache
def create_ref_sliding_window_attention_backend(
    underlying_attn_backend: type[AttentionBackend],
) -> type[AttentionBackend]:
    underlying_builder = underlying_attn_backend.get_builder_cls()
    underlying_impl = underlying_attn_backend.get_impl_cls()

    class RefSlidingWindowAttentionBuilder(underlying_builder):  # type: ignore
        @classmethod
        def get_cudagraph_support(
            cls: type["AttentionMetadataBuilder"],
            vllm_config: VllmConfig,
            kv_cache_spec,
        ) -> AttentionCGSupport:
            return AttentionCGSupport.NEVER

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._init_reorder_batch_threshold(1)
            if hasattr(self, "aot_schedule"):
                self.aot_schedule = False

        def build(
            self,
            common_prefix_len: int,
            common_attn_metadata: CommonAttentionMetadata,
            fast_build: bool = False,
        ) -> AttentionMetadata:
            metadata = super().build(
                common_prefix_len, common_attn_metadata, fast_build
            )
            decode_threshold = self.reorder_batch_threshold or 1
            num_decode_reqs, num_decode_tokens = _get_decode_prefix(
                common_attn_metadata, decode_threshold
            )
            num_reqs = common_attn_metadata.num_reqs

            if num_decode_reqs == 0 or num_decode_reqs == num_reqs:
                setattr(metadata, _SPLIT_METADATA, None)
                setattr(
                    metadata,
                    _USE_DECODE_SLIDING_WINDOW,
                    num_decode_reqs == num_reqs,
                )
                return metadata

            seq_lens_cpu = _get_seq_lens_cpu(common_attn_metadata)
            decode_common_metadata = _slice_common_attn_metadata(
                common_attn_metadata,
                req_start=0,
                req_end=num_decode_reqs,
                token_start=0,
                token_end=num_decode_tokens,
                seq_lens_cpu=seq_lens_cpu,
            )
            prefill_common_metadata = _slice_common_attn_metadata(
                common_attn_metadata,
                req_start=num_decode_reqs,
                req_end=num_reqs,
                token_start=num_decode_tokens,
                token_end=common_attn_metadata.num_actual_tokens,
                seq_lens_cpu=seq_lens_cpu,
            )

            decode_metadata = super().build(0, decode_common_metadata, fast_build)
            prefill_metadata = super().build(0, prefill_common_metadata, fast_build)
            setattr(decode_metadata, _USE_DECODE_SLIDING_WINDOW, True)
            setattr(prefill_metadata, _USE_DECODE_SLIDING_WINDOW, False)
            setattr(
                metadata,
                _SPLIT_METADATA,
                (
                    decode_metadata,
                    prefill_metadata,
                    num_decode_reqs,
                    num_decode_tokens,
                ),
            )
            setattr(metadata, _USE_DECODE_SLIDING_WINDOW, False)
            return metadata

        def update_block_table(
            self,
            metadata,
            blk_table: torch.Tensor,
            slot_mapping: torch.Tensor,
        ):
            metadata = super().update_block_table(metadata, blk_table, slot_mapping)
            split_metadata = getattr(metadata, _SPLIT_METADATA, None)
            if split_metadata is None:
                return metadata

            decode_metadata, prefill_metadata, num_decode_reqs, num_decode_tokens = (
                split_metadata
            )
            decode_metadata = super().update_block_table(
                decode_metadata,
                blk_table[:num_decode_reqs],
                slot_mapping[:num_decode_tokens],
            )
            prefill_metadata = super().update_block_table(
                prefill_metadata,
                blk_table[num_decode_reqs:],
                slot_mapping[num_decode_tokens:],
            )
            setattr(
                metadata,
                _SPLIT_METADATA,
                (
                    decode_metadata,
                    prefill_metadata,
                    num_decode_reqs,
                    num_decode_tokens,
                ),
            )
            return metadata

    class RefSlidingWindowAttentionImpl(underlying_impl):  # type: ignore
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.decode_sliding_window = self.sliding_window
            self.prefill_sliding_window = _full_attention_window(self.sliding_window)

        def _forward_with_window(
            self,
            use_sliding_window: bool,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            original_sliding_window = self.sliding_window
            self.sliding_window = (
                self.decode_sliding_window
                if use_sliding_window
                else self.prefill_sliding_window
            )
            try:
                return super().forward(
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
            finally:
                self.sliding_window = original_sliding_window

        def forward(
            self,
            layer: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if attn_metadata is None:
                return super().forward(
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

            split_metadata = getattr(attn_metadata, _SPLIT_METADATA, None)
            if split_metadata is not None:
                decode_metadata, prefill_metadata, _, num_decode_tokens = split_metadata
                self._forward_with_window(
                    True,
                    layer,
                    query[:num_decode_tokens],
                    _slice_optional_tensor(key, 0, num_decode_tokens),
                    _slice_optional_tensor(value, 0, num_decode_tokens),
                    kv_cache,
                    decode_metadata,
                    output[:num_decode_tokens],
                    _slice_output_scale(output_scale, 0, num_decode_tokens),
                    _slice_output_scale(output_block_scale, 0, num_decode_tokens),
                )
                self._forward_with_window(
                    False,
                    layer,
                    query[num_decode_tokens:],
                    _slice_optional_tensor(key, num_decode_tokens, query.shape[0]),
                    _slice_optional_tensor(value, num_decode_tokens, query.shape[0]),
                    kv_cache,
                    prefill_metadata,
                    output[num_decode_tokens:],
                    _slice_output_scale(
                        output_scale, num_decode_tokens, output.shape[0]
                    ),
                    _slice_output_scale(
                        output_block_scale, num_decode_tokens, output.shape[0]
                    ),
                )
                return output

            use_sliding_window = getattr(
                attn_metadata, _USE_DECODE_SLIDING_WINDOW, False
            )
            return self._forward_with_window(
                use_sliding_window,
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

    return subclass_attention_backend_with_overrides(
        name_prefix="RefSlidingWindowAttention_",
        attention_backend_cls=underlying_attn_backend,
        overrides={
            "get_builder_cls": lambda: RefSlidingWindowAttentionBuilder,
            "get_impl_cls": lambda: RefSlidingWindowAttentionImpl,
        },
    )


class RefSlidingWindowAttention(Attention):
    """Decoder attention with causal prefill and sliding-window decode."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        sliding_window: int,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        kv_sharing_target_layer_name: str | None = None,
        prefix: str = "",
        attn_type: str | None = None,
        **kwargs,
    ):
        if sliding_window <= 0:
            raise ValueError("sliding_window must be a positive integer.")
        if attn_type is not None:
            assert attn_type == AttentionType.DECODER, (
                "RefSlidingWindowAttention only supports AttentionType.DECODER"
            )

        self.decode_sliding_window = sliding_window
        dtype = torch.get_default_dtype()
        kv_cache_dtype = (
            cache_config.cache_dtype if cache_config is not None else "auto"
        )
        underlying_attn_backend = get_attn_backend(
            head_size,
            dtype,
            kv_cache_dtype,
            attn_type=AttentionType.DECODER,
        )
        attn_backend = create_ref_sliding_window_attention_backend(
            underlying_attn_backend
        )

        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=prefix,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            attn_backend=attn_backend,
            **kwargs,
        )

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        assert self.attn_type == AttentionType.DECODER
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            sliding_window=self.decode_sliding_window,
        )
