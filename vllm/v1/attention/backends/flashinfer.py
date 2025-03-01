# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
from flashinfer import (BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import use_cascade_attention

if TYPE_CHECKING:
    from vllm.v1.core.scheduler_output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

logger = init_logger(__name__)


class FlashInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> Type[FlashInferImpl]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type[FlashInferMetadata]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> Type[FlashInferMetadataBuilder]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)


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


@dataclass
class FlashInferMetadata:

    num_actual_tokens: int  # Number of tokens excluding padding.

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    qo_indptr: torch.Tensor
    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor
    # The number of query/output heads
    num_qo_heads: int
    # The number of key/value heads
    num_kv_heads: int
    # The dimension of the attention heads
    head_dim: int
    # Block size of vllm
    page_size: int
    # The data type of the paged kv cache
    data_type: torch.dtype
    # The data type of the query
    q_data_type: torch.dtype

    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    shared_qo_indptr: Optional[torch.Tensor] = None
    shared_kv_page_indptr: Optional[torch.Tensor] = None
    shared_kv_page_indices: Optional[torch.Tensor] = None
    shared_kv_last_page_len: Optional[torch.Tensor] = None

    prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper = None
    cascade_wrapper: MultiLevelCascadeAttentionWrapper = None

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    @property
    def query_start_loc(self):
        # The GPUModelRunner expects to be able to access this property.
        return self.qo_indptr

    def __post_init__(self):
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        supported_head_sizes = FlashInferBackend.get_supported_head_sizes()
        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f" received {self.head_dim}.")


class FlashInferMetadataBuilder:

    def __init__(self, runner: GPUModelRunner):
        self.runner = runner
        self._workspace_buffer = None
        self._prefill_wrapper = None  # Wrapper for prefill/append
        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = get_current_vllm_config()

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput):
        pass

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), "NHD")
        return self._prefill_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), "NHD")
        return self._cascade_wrapper

    def _plan(self, attn_metadata: FlashInferMetadata):
        if self.global_hyperparameters is None:
            self.global_hyperparameters = infer_global_hyperparameters(
                get_per_layer_parameters(self.vllm_config))
        if attn_metadata.use_cascade:
            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                [attn_metadata.shared_qo_indptr, attn_metadata.qo_indptr],
                [
                    attn_metadata.shared_kv_page_indptr,
                    attn_metadata.paged_kv_indptr
                ],
                [
                    attn_metadata.shared_kv_page_indices,
                    attn_metadata.paged_kv_indices
                ],
                [
                    attn_metadata.shared_kv_last_page_len,
                    attn_metadata.paged_kv_last_page_len
                ],
                attn_metadata.num_qo_heads,
                attn_metadata.num_kv_heads,
                attn_metadata.head_dim,
                attn_metadata.page_size,
                causal=True,
                sm_scale=self.global_hyperparameters.sm_scale,
                window_left=self.global_hyperparameters.window_left,
                logits_soft_cap=self.global_hyperparameters.logits_soft_cap,
                q_data_type=attn_metadata.q_data_type,
            )
        else:
            attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
            attn_metadata.prefill_wrapper.plan(
                attn_metadata.qo_indptr,
                attn_metadata.paged_kv_indptr,
                attn_metadata.paged_kv_indices,
                attn_metadata.paged_kv_last_page_len,
                attn_metadata.num_qo_heads,
                attn_metadata.num_kv_heads,
                attn_metadata.head_dim,
                attn_metadata.page_size,
                causal=True,
                sm_scale=self.global_hyperparameters.sm_scale,
                window_left=self.global_hyperparameters.window_left,
                logits_soft_cap=self.global_hyperparameters.logits_soft_cap,
                q_data_type=attn_metadata.q_data_type,
                kv_data_type=attn_metadata.data_type,
            )

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        page_size = self.runner.block_size
        device = self.runner.device
        qo_indptr = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            self.runner.device, non_blocking=True)
        seq_lens = self.runner.seq_lens_cpu[:num_reqs].to(self.runner.device,
                                                          non_blocking=True)
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            self.runner.device, non_blocking=True).long()

        block_table_bounds = (seq_lens + page_size - 1) // page_size

        use_cascade = common_prefix_len > 0
        if use_cascade:
            # Grab the blocks of the shared prefix from the first request.
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            shared_qo_indptr = torch.tensor([0, num_actual_tokens],
                                            dtype=torch.int32,
                                            device=device)
            shared_kv_page_indptr = torch.tensor([0, num_common_kv_blocks],
                                                 dtype=torch.int32,
                                                 device=device)
            shared_kv_page_indices = block_table[0, :num_common_kv_blocks]
            shared_kv_last_page_len = torch.tensor([page_size],
                                                   dtype=torch.int32,
                                                   device=device)
            # Remove the blocks of the shared prefix from all requests.
            block_table = block_table[:, num_common_kv_blocks:]
            block_table_bounds -= num_common_kv_blocks
        else:
            shared_qo_indptr = None
            shared_kv_page_indptr = None
            shared_kv_page_indices = None
            shared_kv_last_page_len = None

        mask = (torch.arange(block_table.size(1),
                             dtype=block_table.dtype,
                             device=block_table.device).unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table[mask]

        paged_kv_indptr = torch.cat([
            torch.zeros(1,
                        dtype=block_table_bounds.dtype,
                        device=block_table_bounds.device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32)
        ])

        paged_kv_last_page_len = seq_lens % page_size
        paged_kv_last_page_len = torch.where(paged_kv_last_page_len == 0,
                                             page_size, paged_kv_last_page_len)

        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.runner.num_query_heads,
            num_kv_heads=self.runner.num_kv_heads,
            head_dim=self.runner.head_size,
            page_size=page_size,
            data_type=self.runner.kv_cache_dtype,
            q_data_type=self.runner.dtype,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            shared_qo_indptr=shared_qo_indptr,
            shared_kv_page_indptr=shared_kv_page_indptr,
            shared_kv_page_indices=shared_kv_page_indices,
            shared_kv_last_page_len=shared_kv_last_page_len,
        )

        self._plan(attn_metadata)

        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.runner.kv_cache_dtype != self.runner.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        return use_cascade_attention(*args, **kwargs)


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
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")

    def forward(
        self,
        layer: torch.nn.Module,
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

        window_left = (self.sliding_window[0]
                       if self.sliding_window is not None else -1)

        if not attn_metadata.use_cascade:
            # Regular attention (common case).
            assert attn_metadata.prefill_wrapper is not None
            assert attn_metadata.prefill_wrapper._causal
            assert attn_metadata.prefill_wrapper._window_left == window_left
            assert attn_metadata.prefill_wrapper._logits_soft_cap == (
                self.logits_soft_cap or 0.0)
            assert attn_metadata.prefill_wrapper._sm_scale == self.scale
            output = attn_metadata.prefill_wrapper.run(
                query,
                kv_cache,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output,
            )
            return output

        # Cascade attention (rare case).
        assert attn_metadata.cascade_wrapper is not None
        output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
        return output
