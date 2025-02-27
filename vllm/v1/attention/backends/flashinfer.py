# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    # Avoid turning these types into variables during type checking
    if not TYPE_CHECKING:
        BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

import numpy as np
import torch

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionState, AttentionType)
from vllm.attention.layer import Attention
from vllm.attention.ops.triton_merge_attn_states import merge_attn_states
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv

if current_platform.is_cuda():
    from vllm.vllm_flash_attn import flash_attn_varlen_func

logger = init_logger(__name__)

FLASHINFER_WORKSPACE_BUFFER = None
FLASHINFER_WRAPPER = None


class FlashInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMetadata

    @staticmethod
    def get_state_cls() -> Type["FlashInferState"]:
        return FlashInferState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False
        return use_cascade_attention(*args, **kwargs)



@dataclass
class PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters.
    """

    window_left: int
    logits_soft_cap: Optional[float]
    sm_scale: float


def get_per_layer_parameters(
        vllm_config: VllmConfig) -> Dict[str, PerLayerParameters]:
    """
    Scan all attention layers and determine some hyperparameters
    to use during `plan`.
    """

    layers = vllm_config.compilation_config.static_forward_context
    per_layer_params: Dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
        assert isinstance(layer, Attention)

        impl = layer.impl
        assert isinstance(impl, FlashInferImpl)

        # Infer hyperparameters from the attention layer
        window_size = impl.sliding_window
        window_left = window_size[0] if window_size is not None else -1
        logits_soft_cap = impl.logits_soft_cap
        sm_scale = impl.scale

        per_layer_params[key] = PerLayerParameters(window_left,
                                                   logits_soft_cap, sm_scale)

    return per_layer_params


def infer_global_hyperparameters(
        per_layer_params: Dict[str, PerLayerParameters]) -> PerLayerParameters:
    """
    Currently, FlashInfer backend only support models in which all layers share
    the same values for the following hyperparameters:
    - `window_left`
    - `logits_soft_cap`
    - `sm_scale`

    So this function asserts that all layers share the same values for these
    hyperparameters and returns the global values.
    """

    assert len(per_layer_params) > 0, "No attention layers found in the model."

    param_sets = list(per_layer_params.values())
    global_params = param_sets[0]
    for params in param_sets:
        assert params == global_params, (
            "FlashInfer backend currently only supports models in which all "
            "layers share the same values for the following hyperparameters: "
            "`window_left`, `logits_soft_cap`, `sm_scale`.")

    return global_params


class FlashInferState(AttentionState):

    def __init__(self, runner):
        self.runner = runner
        self._workspace_buffer = None
        self._wrapper = None

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = get_current_vllm_config()

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def _get_wrapper(self):
        if self._wrapper is None:
            self._wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), "NHD")
        return self._wrapper

    def begin_forward(self, attn_metadata: "FlashInferMetadata"):
        attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
        attn_metadata.decode_wrapper = self._get_decode_wrapper()
        attn_metadata.begin_forward()


@dataclass
class FlashInferMetadata:

    num_actual_tokens: int  # Number of tokens excluding padding.

    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    qo_indptr: torch.Tensor
    # The number of query/output heads
    num_qo_heads: int
    # The number of key/value heads
    num_kv_heads: int
    # The dimension of the attention heads
    head_dim: int
    # Block size of vllm
    page_size: int

    wrapper: BatchPrefillWithPagedKVCacheWrapper = None

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.
    
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    # FlashInfer 0.2 encourages passing host tensors
    device: torch.device = torch.device("cpu")

    # The FlashInfer backend currently supports only models in which all layers
    # share the same following hyperparameters:

    # The left (inclusive) window size for the attention window, when
    # set to `-1`, the window size will be set to the full length of
    # the sequence. Defaults to `-1`.
    window_left: int = -1
    # The attention logits soft capping value (used in Gemini, Grok and
    # Gemma-2, etc.), if not provided, will be set to `0`. If greater
    # than 0, the logits will be capped according to formula:
    # $$\texttt{logits\_soft\_cap} \times
    # \mathrm{tanh}(x / \texttt{logits\_soft\_cap})$$,
    # where $x$ is the input logits.
    logits_soft_cap: Optional[float] = None
    # The scale used in softmax, if not provided, will be set to
    # `1.0 / sqrt(head_dim)`.
    sm_scale: Optional[float] = None

    def __post_init__(self):
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        supported_head_sizes = FlashInferBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f" received {self.head_dim}.")

    def plan(self):
        global FLASHINFER_WORKSPACE_BUFFER, FLASHINFER_WRAPPER
        block_table_bounds = (
            (self.seq_lens + self.page_size - 1) // self.page_size)

        # An example for paged_kv_indices, paged_kv_indptr:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]

        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        mask = (torch.arange(self.block_table.size(1),
                             dtype=self.block_table.dtype,
                             device=self.block_table.device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = self.block_table[mask]

        # paged_kv_indptr is used to index into paged_kv_indices: [0, 3, 6, 8]
        # Shape: [batch_size + 1]
        paged_kv_indptr = torch.cat([
            torch.zeros(1, dtype=block_table_bounds.dtype,
                        device=block_table_bounds.device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)])

        # The number of entries in the last page of each request in
        # the paged kv cache, shape: [batch_size]
        paged_kv_last_page_len = self.seq_lens % self.page_size
        paged_kv_last_page_len = torch.where(
            paged_kv_last_page_len == 0, self.page_size, paged_kv_last_page_len)

        if FLASHINFER_WORKSPACE_BUFFER is None:
            FLASHINFER_WORKSPACE_BUFFER = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device="cuda")
            FLASHINFER_WRAPPER = BatchPrefillWithPagedKVCacheWrapper(
                FLASHINFER_WORKSPACE_BUFFER, "NHD")

        FLASHINFER_WRAPPER.plan(
            self.qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            causal=True,
            #sm_scale=self.sm_scale,
            #window_left=self.window_left,
            #logits_soft_cap=self.logits_soft_cap,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type)


class FlashInferImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlashInferBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashInfer. "
                f"Supported head sizes are: {support_head_sizes}.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens] and
        # value[:num_actual_tokens] because the reshape_and_cache_flash op uses
        # the slot_mapping's shape to determine the number of actual tokens.
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[:, 0],
            kv_cache[:, 1],
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        output = FLASHINFER_WRAPPER.run(
            query,
            kv_cache,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            out=output,
        )

        return output


def use_cascade_attention(
    common_prefix_len: int,
    query_lens: np.ndarray,
    num_query_heads: int,
    num_kv_heads: int,
    use_alibi: bool,
    use_sliding_window: bool,
    num_sms: int,
) -> bool:
    """Decide whether to use cascade attention.

    This function 1) checks whether cascade attention is supported with the
    given configuration, and 2) heuristically decides whether using cascade
    attention can improve performance.
    """
    # Too short common prefix. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 256 tokens. TODO: Tune this threshold.
    # NOTE(woosuk): This is the common case. We should return False as soon as
    # possible to avoid any unnecessary computation.
    if common_prefix_len < 256:
        return False
    # Cascade attention is currently not supported with these variants.
    if use_alibi or use_sliding_window:
        return False
    # Too few queries. Probably not worth using cascade attention.
    # We use an arbitrary threshold of 8 queries. TODO: Tune this threshold.
    num_reqs = len(query_lens)
    if num_reqs < 8:
        return False

    # Heuristics to decide whether using cascade attention is beneficial.
    # 1. When FlashDecoding is not used for normal attention, cascade attention
    #    is likely to be faster since it saves memory bandwidth.
    num_queries_per_kv = num_query_heads // num_kv_heads
    # The criteria for using FlashDecoding can be found in the following link:
    # https://github.com/vllm-project/flash-attention/blob/96266b1111111f3d11aabefaf3bacbab6a89d03c/csrc/flash_attn/flash_api.cpp#L535
    use_flash_decoding = (num_queries_per_kv > 1 and not use_sliding_window
                          and not use_alibi and np.all(query_lens == 1))
    if not use_flash_decoding:
        # Use cascade attention.
        return True

    # 2. When FlashDecoding is used for normal attention, it is not clear
    #    whether cascade attention is beneficial, because FlashDecoding can
    #    launch more CTAs than cascade attention.
    #    We use a simple performance model to compare the two methods.
    #    NOTE(woosuk): The performance model is very rough and may not be
    #    accurate.
    num_tokens = num_reqs
    # NOTE(woosuk): These are default tile sizes. flash-attn might use
    # different tile sizes (e.g., 64 or 256) depending on the configuration.
    q_tile_size = 128
    kv_tile_size = 128
    num_prefix_tiles = cdiv(common_prefix_len, kv_tile_size)

    cascade_ctas = num_query_heads * cdiv(num_tokens, q_tile_size)
    cascade_waves = cdiv(cascade_ctas, num_sms)
    cascade_time = cascade_waves * num_prefix_tiles

    flash_decoding_ctas = (num_reqs * num_kv_heads *
                           cdiv(num_queries_per_kv, q_tile_size))
    flash_decoding_ctas *= num_prefix_tiles
    flash_decoding_time = cdiv(flash_decoding_ctas, num_sms)

    # Use cascade attention if it is faster than FlashDecoding.
    return cascade_time < flash_decoding_time


def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: Tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
) -> torch.Tensor:
    assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window.")

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0

    # Process shared prefix.
    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        seqused_k=prefix_kv_lens,
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # Process suffix per query.
    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        seqused_k=suffix_kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        fa_version=fa_version,
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)