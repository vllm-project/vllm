# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

from vllm.multimodal import MultiModalPlaceholderMap

try:
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper
    from flashinfer.decode import CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper

    from vllm.vllm_flash_attn import flash_attn_varlen_func
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    # Avoid turning these types into variables during type checking
    if not TYPE_CHECKING:
        BatchDecodeWithPagedKVCacheWrapper = None
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper = None
        BatchPrefillWithPagedKVCacheWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

import torch

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.layer import Attention
from vllm.attention.ops.paged_attn import PagedAttention
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import (async_tensor_h2d, get_kv_cache_torch_dtype,
                        make_tensor_with_pad)

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

FLASHINFER_KV_CACHE_LAYOUT: str = os.getenv("FLASHINFER_KV_CACHE_LAYOUT",
                                            "NHD").upper()


class FlashInferBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

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
    def get_kv_cache_stride_order() -> Tuple[int, ...]:
        cache_layout = FLASHINFER_KV_CACHE_LAYOUT
        assert (cache_layout in ("NHD", "HND"))
        stride_order = (0, 1, 2, 3, 4) if cache_layout == "NHD" else (0, 1, 3,
                                                                      2, 4)
        return stride_order

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 128, 256]

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


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
        self._is_graph_capturing = False
        self._workspace_buffer = None
        self._decode_wrapper = None
        self._prefill_wrapper = None

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = self.runner.vllm_config
        self._kv_cache_layout = None

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def get_kv_cache_layout(self):
        if self._kv_cache_layout is None:
            self._kv_cache_layout = FLASHINFER_KV_CACHE_LAYOUT
        return self._kv_cache_layout

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), self.get_kv_cache_layout())
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
                self.get_kv_cache_layout(),
                use_tensor_cores=use_tensor_cores)
        return self._decode_wrapper

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._is_graph_capturing = True
        self._graph_decode_wrapper = None
        self._graph_slot_mapping = torch.full((max_batch_size, ),
                                              PAD_SLOT_ID,
                                              dtype=torch.long,
                                              device=self.runner.device)
        self._graph_seq_lens = torch.ones(max_batch_size,
                                          dtype=torch.int32,
                                          device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(
            self.runner.graph_block_tables).to(device=self.runner.device)
        self._graph_decode_workspace_buffer = self._get_workspace_buffer()
        self._graph_indices_buffer = torch.empty(
            max_batch_size * self.runner.cache_config.num_gpu_blocks,
            dtype=torch.int32,
            device=self.runner.device)
        self._graph_indptr_buffer = torch.empty(max_batch_size + 1,
                                                dtype=torch.int32,
                                                device=self.runner.device)
        self._graph_last_page_len_buffer = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.runner.device)
        yield
        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables
        del self._graph_decode_workspace_buffer
        del self._graph_indices_buffer
        del self._graph_indptr_buffer
        del self._graph_last_page_len_buffer
        del self._graph_decode_wrapper

    def graph_clone(self, batch_size: int):
        assert self._is_graph_capturing
        state = self.__class__(self.runner)
        state._workspace_buffer = self._graph_decode_workspace_buffer
        state._decode_wrapper = self._graph_decode_wrapper
        state._prefill_wrapper = self._get_prefill_wrapper()
        return state

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing
        _indptr_buffer = self._graph_indptr_buffer[:batch_size + 1]
        _last_page_len_buffer = self._graph_last_page_len_buffer[:batch_size]

        num_qo_heads = (self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config))
        num_kv_heads = self.runner.model_config.get_num_kv_heads(
            self.runner.parallel_config)
        use_tensor_cores = envs.VLLM_FLASHINFER_FORCE_TENSOR_CORES or (
            num_qo_heads // num_kv_heads > 4)
        self._graph_decode_wrapper = \
            CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self._graph_decode_workspace_buffer, _indptr_buffer,
            self._graph_indices_buffer, _last_page_len_buffer,
            self.get_kv_cache_layout(),
            use_tensor_cores)
        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        paged_kv_indptr_tensor_host = torch.arange(0,
                                                   batch_size + 1,
                                                   dtype=torch.int32)
        paged_kv_indices_tensor_host = torch.arange(0,
                                                    batch_size,
                                                    dtype=torch.int32)
        paged_kv_last_page_len_tensor_host = torch.full((batch_size, ),
                                                        self.runner.block_size,
                                                        dtype=torch.int32)
        query_start_loc_host = torch.arange(0,
                                            batch_size + 1,
                                            dtype=torch.int32)

        global_params = infer_global_hyperparameters(
            get_per_layer_parameters(self.vllm_config))

        attn_metadata = self.runner.attn_backend.make_metadata(
            num_prefills=0,
            slot_mapping=self._graph_slot_mapping[:batch_size],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            max_prefill_seq_len=0,
            block_tables=self._graph_block_tables,
            paged_kv_indptr=paged_kv_indptr_tensor_host,
            paged_kv_indices=paged_kv_indices_tensor_host,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor_host,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=self.runner.model_config.get_head_size(),
            page_size=self.runner.block_size,
            seq_start_loc=None,
            query_start_loc=query_start_loc_host,
            device=self.runner.device,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            use_cuda_graph=True,
            decode_wrapper=self._graph_decode_wrapper,
            prefill_wrapper=None,
            **dataclasses.asdict(global_params),
        )
        attn_metadata.begin_forward()
        return attn_metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        return {
            "slot_mapping": attn_metadata.slot_mapping,
        }

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        return

    def begin_forward(self, model_input):
        assert not self._is_graph_capturing
        state = self
        use_cuda_graph = model_input.attn_metadata.use_cuda_graph
        is_decode = model_input.attn_metadata.num_prefills == 0
        # In case of multistep chunked-prefill, there might be prefill requests
        # scheduled while CUDA graph mode is enabled. We don't run graph in that
        # case.
        if use_cuda_graph and is_decode:
            batch_size = model_input.input_tokens.shape[0]
            state = (self.runner.graph_runners[model_input.virtual_engine]
                     [batch_size].attn_state)
        model_input.attn_metadata.prefill_wrapper = state._get_prefill_wrapper(
        )
        model_input.attn_metadata.decode_wrapper = state._get_decode_wrapper()
        model_input.attn_metadata.begin_forward()


@dataclass
class FlashInferMetadata(AttentionMetadata):
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Number of query tokens for each request in the batch.
    # Currently, we require that all requests have the same number of query
    # tokens during the decoding phase. When speculavie decoding is enabled,
    # decode_query_len might be greater than 1. In all other cases, it is 1.
    decode_query_len: Optional[int] = 1

    use_cuda_graph: bool = True

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None

    # Metadata for the prefill stage
    seq_start_loc: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # used for GPU in-place advance_step
    seq_lens_tensor: Optional[torch.Tensor] = None
    block_table_bound: Optional[torch.Tensor] = None

    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    # The number of query/output heads
    num_qo_heads: Optional[int] = None
    # The number of key/value heads
    num_kv_heads: Optional[int] = None
    # The dimension of the attention heads
    head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    # FlashInfer 0.2 encourages passing host tensors
    device: torch.device = torch.device("cpu")
    is_profile_run: bool = False

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

    def begin_forward(self):
        if self.num_prefill_tokens > 0:
            if self.paged_kv_indices is None:
                return

            assert self.prefill_wrapper is not None
            assert self.query_start_loc is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            assert self.block_table_bound is not None
            assert self.seq_lens_tensor is not None
            self.query_start_loc = self.query_start_loc[:self.num_prefills + 1]
            batch_size = self.query_start_loc.shape[0] - 1
            assert batch_size >= 0
            # We will use flash attention for profiling to
            # determine the number of blocks. Therefore,
            # we don't need to prepare the input for flashinfer for profile run.
            if not self.is_profile_run:
                self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
                self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                    self.device)
                self.block_table_bound = self.block_table_bound.to(self.device)
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)
                self.paged_kv_indices = self.paged_kv_indices.to(self.device)
                self.prefill_wrapper.plan(
                    self.query_start_loc,
                    self.paged_kv_indptr[:self.num_prefills + 1],
                    self.paged_kv_indices,
                    self.paged_kv_last_page_len[:self.num_prefills],
                    self.num_qo_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    self.page_size,
                    causal=True,
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.data_type)
        if self.num_decode_tokens > 0:
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.paged_kv_last_page_len is not None
            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
            self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            # handle model warmup path
            if self.block_table_bound is not None:
                self.block_table_bound = self.block_table_bound.to(self.device)
            if self.seq_lens_tensor is not None:
                self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)

            assert self.decode_wrapper is not None
            self.decode_wrapper.plan(
                self.paged_kv_indptr[self.num_prefills:],
                self.paged_kv_indices,
                self.paged_kv_last_page_len[self.num_prefills:],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                # Disable flashinfer's pos encoding and use vllm's rope.
                pos_encoding_mode="NONE",
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                sm_scale=self.sm_scale,
                # kv-cache data type.
                kv_data_type=self.data_type,
                # query data type.
                q_data_type=self.q_data_type)

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('prefill_wrapper')
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)

    @property
    def prefill_metadata(self) -> Optional["FlashInferMetadata"]:
        if self.num_prefills == 0:
            return None
        return self

    @property
    def decode_metadata(self) -> Optional["FlashInferMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        return self

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """

        if turn_prefills_into_decodes:
            # When Multi-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            # Flashinfer doesn't support speculative decoding + chunked-prefill
            # + multi-step scheduling yet.
            assert self.decode_query_len == 1
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens_tensor is not None

        assert num_seqs > 0
        assert num_queries > 0
        assert model_input.attn_metadata is not None
        assert sampled_token_ids is not None

        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        model_input.input_tokens[:num_queries] = sampled_token_ids.flatten()

        # Update GPU tensors
        ops.advance_step_flashinfer(
            num_seqs=num_seqs,
            num_queries=num_queries,
            block_size=block_size,
            input_tokens=model_input.input_tokens,
            sampled_token_ids=model_input.input_tokens,
            input_positions=model_input.input_positions,
            seq_lens=self.seq_lens_tensor,
            slot_mapping=self.slot_mapping,
            block_tables=self.block_tables,
            paged_kv_indices=self.paged_kv_indices,
            paged_kv_indptr=self.paged_kv_indptr,
            paged_kv_last_page_len=self.paged_kv_last_page_len,
            block_table_bound=self.block_table_bound)


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = self.runner.vllm_config

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        self.paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        self.paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        self.paged_kv_last_page_len: List[int] = []
        self.total_blocks = 0
        self.is_profile_run: bool = False

        if self.global_hyperparameters is None:
            # Infer global hyperparameters, since currently we only support
            # models in which all layers share the same values for the
            # following hyperparameters:
            # - `window_left`
            # - `logits_soft_cap`
            # - `sm_scale`
            inferred_params = infer_global_hyperparameters(
                get_per_layer_parameters(self.vllm_config))
            self.global_hyperparameters = inferred_params
            self.window_left = inferred_params.window_left
            self.logits_soft_cap = inferred_params.logits_soft_cap
            self.sm_scale = inferred_params.sm_scale

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            is_profile_run = is_block_tables_empty(block_tables)

            # Compute slot mapping.
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            if is_profile_run:
                self.is_profile_run = is_profile_run
                return

            block_table = block_tables[seq_id]
            self._update_paged_kv_tensors(block_table, seq_len)

    def _update_paged_kv_tensors(self, block_table: List[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = seq_len // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_len.append(last_page_len)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        decode_query_len = max(query_lens[self.num_prefills:], default=1)

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            max_blocks = input_block_tables.shape[1]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    num_blocks = len(block_table)
                    if num_blocks <= max_blocks:
                        input_block_tables[i, :num_blocks] = block_table
                    else:
                        # It may be possible to have more blocks allocated due
                        # to lookahead slots of multi-step, however, they are
                        # not used anyway, so can be safely ignored.
                        input_block_tables[
                            i, :max_blocks] = block_table[:max_blocks]

            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)

            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )

        assert device is not None
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            self.paged_kv_indices.extend(
                [0] * (self.total_blocks - len(self.paged_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)
            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device="cpu",
                                                   dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None
            block_table_bound_tensor = None

        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        return FlashInferMetadata(
            decode_query_len=decode_query_len,
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=False,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            max_prefill_seq_len=max_prefill_seq_len,
            block_tables=block_tables,
            paged_kv_indptr=paged_kv_indptr_tensor,
            paged_kv_indices=paged_kv_indices_tensor,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            block_table_bound=block_table_bound_tensor,
            seq_lens_tensor=seq_lens_tensor,
            num_qo_heads=self.runner.model_config.get_num_attention_heads(
                self.runner.parallel_config),
            num_kv_heads=self.runner.model_config.get_num_kv_heads(
                self.runner.parallel_config),
            head_dim=self.runner.model_config.get_head_size(),
            page_size=self.block_size,
            seq_start_loc=seq_start_loc,
            query_start_loc=query_start_loc,
            device=device,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            use_cuda_graph=use_captured_graph,
            is_profile_run=self.is_profile_run,
            window_left=self.window_left,
            logits_soft_cap=self.logits_soft_cap,
            sm_scale=self.sm_scale,
        )


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
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
    ) -> None:
        if use_irope:
            logger.warning_once(
                "Using irope in FlashInfer is not supported yet, it will fall"
                " back to global attention for long context.")
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
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # TODO: directly write to output tensor
        num_heads: int = self.num_heads
        head_size: int = self.head_size
        num_kv_heads: int = self.num_kv_heads
        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes = self.alibi_slopes
        logits_soft_cap = self.logits_soft_cap

        num_tokens, hidden_size = query.shape
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        if kv_cache.numel() > 0:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
            # to process the cache when the kv_cache_dtype is fp8
            if kv_cache_dtype.startswith("fp8"):
                torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                    kv_cache_dtype)
                kv_cache = kv_cache.view(torch_dtype)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"key : {key.shape} : #prefill tokens {num_prefill_tokens} : #decode tokens {num_decode_tokens}" # noqa
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"value : {value.shape} : #prefill toks {num_prefill_tokens} : #decode toks {num_decode_tokens}" # noqa
        query = query.contiguous(
        )  # Flashinfer requires query to be contiguous
        # Query for decode. KV is not needed because it is already cached.
        # QKV for prefill.
        decode_query = query[num_prefill_tokens:]
        query = query[:num_prefill_tokens]

        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        window_left = window_size[0] if window_size is not None else -1

        prefill_output: Optional[torch.Tensor] = None
        decode_output: Optional[torch.Tensor] = None
        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        if prefill_meta := attn_metadata.prefill_metadata:
            # We will use flash attention for prefill
            # when kv_cache is not provided.
            # This happens when vllm runs the profiling to
            # determine the number of blocks.
            if kv_cache.numel() == 0:
                prefill_output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                )
            else:
                assert prefill_meta is not None
                assert prefill_meta.prefill_wrapper is not None

                assert prefill_meta.prefill_wrapper._causal
                assert prefill_meta.prefill_wrapper._window_left == window_left
                assert prefill_meta.prefill_wrapper._logits_soft_cap == (
                    logits_soft_cap or 0.0)
                assert prefill_meta.prefill_wrapper._sm_scale == softmax_scale

                prefill_output = prefill_meta.prefill_wrapper.run(
                    query,
                    kv_cache.permute(*stride_order),
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                )
        if decode_meta := attn_metadata.decode_metadata:
            assert decode_meta is not None
            assert decode_meta.decode_wrapper is not None

            assert decode_meta.decode_wrapper._window_left == window_left
            assert decode_meta.decode_wrapper._logits_soft_cap == (
                logits_soft_cap or 0.0)
            assert decode_meta.decode_wrapper._sm_scale == softmax_scale

            decode_output = decode_meta.decode_wrapper.run(
                decode_query,
                kv_cache.permute(*stride_order),
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
            )

        if prefill_output is None and decode_output is not None:
            # Decode only batch.
            output, num_tokens = decode_output, num_decode_tokens
        elif decode_output is None and prefill_output is not None:
            # Prefill only batch.
            output, num_tokens = prefill_output, num_prefill_tokens
        else:
            # Chunked prefill batch does not work with speculative decoding in
            # FlashInfer backend, so the query length for decode should be 1.
            assert prefill_output is not None
            assert decode_output is not None
            assert decode_meta is not None
            assert decode_meta.decode_query_len == 1
            decode_output = decode_output.squeeze(1)
            output = torch.cat([prefill_output, decode_output], dim=0)
        return output.view(num_tokens, hidden_size)
