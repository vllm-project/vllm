# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_current_vllm_config,
                         get_layers_from_vllm_config)
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import use_cascade_attention

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

logger = init_logger(__name__)


class FlashInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type[FlashInferImpl]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> type[FlashInferMetadata]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> type[FlashInferMetadataBuilder]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
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
        vllm_config: VllmConfig) -> dict[str, PerLayerParameters]:
    """
    Scan all attention layers and determine some hyperparameters
    to use during `plan`.
    """

    layers = get_layers_from_vllm_config(vllm_config, Attention)
    per_layer_params: dict[str, PerLayerParameters] = {}

    for key, layer in layers.items():
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
        per_layer_params: dict[str, PerLayerParameters]) -> PerLayerParameters:
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

    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # For cascade attention.
    use_cascade: bool
    shared_qo_indptr: Optional[torch.Tensor] = None
    shared_kv_page_indptr: Optional[torch.Tensor] = None
    shared_kv_page_indices: Optional[torch.Tensor] = None
    shared_kv_last_page_len: Optional[torch.Tensor] = None

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
    cascade_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None

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
        self._decode_wrapper = None  # Wrapper for decode
        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = get_current_vllm_config()

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

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

    def _get_decode_wrapper(self):
        if self._decode_wrapper is None:
            num_qo_heads = (self.runner.model_config.get_num_attention_heads(
                self.runner.parallel_config))
            num_kv_heads = self.runner.model_config.get_num_kv_heads(
                self.runner.parallel_config)
            use_tensor_cores = envs.VLLM_FLASHINFER_FORCE_TENSOR_CORES or (
                num_qo_heads // num_kv_heads > 4)
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                "NHD",
                use_tensor_cores=use_tensor_cores)
        return self._decode_wrapper

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
            # Regular attention (common case).
            # Decodes are at the front and prefills are at the back,
            # according to reorder_batch()
            if self._num_prefills > 0:
                # Decodes are first so prefills start after the last decode
                prefill_start = self._num_decodes
                attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
                assert attn_metadata.qo_indptr[prefill_start:].shape[
                    0] == self._num_prefills + 1
                assert attn_metadata.paged_kv_indptr[prefill_start:].shape[
                    0] == self._num_prefills + 1
                assert attn_metadata.paged_kv_last_page_len[
                    prefill_start:].shape[0] == self._num_prefills
                # Since prefill_wrapper.run() will be called with
                # query[num_decode_tokens:] we need to adjust the qo_indptr
                # to be relative to the start of the prefill queries.
                qo_indptr = attn_metadata.qo_indptr[
                    prefill_start:] - attn_metadata.qo_indptr[prefill_start]
                attn_metadata.prefill_wrapper.plan(
                    qo_indptr,
                    attn_metadata.paged_kv_indptr[prefill_start:],
                    attn_metadata.paged_kv_indices,
                    attn_metadata.paged_kv_last_page_len[prefill_start:],
                    attn_metadata.num_qo_heads,
                    attn_metadata.num_kv_heads,
                    attn_metadata.head_dim,
                    attn_metadata.page_size,
                    causal=True,
                    sm_scale=self.global_hyperparameters.sm_scale,
                    window_left=self.global_hyperparameters.window_left,
                    logits_soft_cap=self.global_hyperparameters.
                    logits_soft_cap,
                    q_data_type=attn_metadata.q_data_type,
                    kv_data_type=attn_metadata.data_type,
                )

            if self._num_decodes > 0:
                attn_metadata.decode_wrapper = self._get_decode_wrapper()
                attn_metadata.decode_wrapper.plan(
                    attn_metadata.paged_kv_indptr[:self._num_decodes + 1],
                    attn_metadata.paged_kv_indices,
                    attn_metadata.paged_kv_last_page_len[:self._num_decodes],
                    attn_metadata.num_qo_heads,
                    attn_metadata.num_kv_heads,
                    attn_metadata.head_dim,
                    attn_metadata.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.global_hyperparameters.sm_scale,
                    window_left=self.global_hyperparameters.window_left,
                    logits_soft_cap=self.global_hyperparameters.
                    logits_soft_cap,
                    q_data_type=attn_metadata.q_data_type,
                    kv_data_type=attn_metadata.data_type,
                )

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int):
        assert self._num_decodes + self._num_prefills == num_reqs
        assert (self._num_decode_tokens +
                self._num_prefill_tokens == num_actual_tokens)
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
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            num_prefill_tokens=self._num_prefill_tokens,
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
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
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
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
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

        num_actual_tokens = attn_metadata.num_actual_tokens
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

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back,
        # according to reorder_batch()
        if prefill_wrapper := attn_metadata.prefill_wrapper:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens
            assert prefill_wrapper is not None
            assert prefill_wrapper._causal
            assert prefill_wrapper._window_left == window_left
            assert prefill_wrapper._logits_soft_cap == (self.logits_soft_cap
                                                        or 0.0)
            assert prefill_wrapper._sm_scale == self.scale
            prefill_wrapper.run(
                prefill_query,
                kv_cache,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output[num_decode_tokens:],
            )

        if decode_wrapper := attn_metadata.decode_wrapper:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens
            assert decode_wrapper is not None
            assert decode_wrapper._window_left == window_left
            assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap
                                                       or 0.0)
            assert decode_wrapper._sm_scale == self.scale
            decode_wrapper.run(
                decode_query,
                kv_cache,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output[:num_decode_tokens],
            )

        return output_padded
