# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Type

import torch
from packaging.version import Version

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata,
                                                MLACommonMetadataBuilder,
                                                MLACommonState)
from vllm.attention.backends.utils import (PerLayerParameters,
                                           get_fp8_dtype_for_flashinfer,
                                           get_mla_dims,
                                           infer_global_hyperparameters,
                                           is_block_tables_empty)
from vllm.config import get_current_vllm_config
from vllm.utils import get_kv_cache_torch_dtype

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)
try:
    import flashinfer
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 512 * 1024 * 1024
except ImportError:
    # Avoid turning these types into variables during type checking
    if not TYPE_CHECKING:
        BatchMLAPagedAttentionWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0


class FlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["FlashInferMLAMetadata"]:
        return FlashInferMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashInferMLAState"]:
        return FlashInferMLAState


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata):
    decode_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None

    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr_host: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices_host: Optional[torch.Tensor] = None

    query_start_loc_host: Optional[torch.Tensor] = None
    seq_lens_tensor_host: Optional[torch.Tensor] = None

    block_table_bound: Optional[torch.Tensor] = None

    sm_scale: float = 1.0
    logits_soft_cap: Optional[float] = None
    window_left: int = -1

    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    # FlashInfer 0.2 encourages passing host tensors
    device: torch.device = torch.device("cpu")

    _cached_prefill_metadata: Optional["FlashInferMLAMetadata"] = None
    _cached_decode_metadata: Optional["FlashInferMLAMetadata"] = None

    @property
    def decode_metadata(self):
        decode_metadata = super().decode_metadata
        # TODO: cache assignment?
        if decode_metadata is not None:
            decode_metadata.decode_wrapper = self.decode_wrapper
        return decode_metadata


class FlashInferMLAState(MLACommonState[FlashInferMLAMetadata]):

    def __init__(self, runner):
        super().__init__(runner)

        self._is_graph_capturing = False
        self._workspace_buffer = None
        self._decode_wrapper = None

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = get_current_vllm_config()

        self.kv_lora_rank, self.qk_rope_head_dim = get_mla_dims(
            self.runner.model_config)
        self.num_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        self.page_size = self.runner.block_size

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def _get_decode_wrapper(self):
        if self._decode_wrapper is None:
            self._decode_wrapper = BatchMLAPagedAttentionWrapper(
                self._get_workspace_buffer(),
                backend="fa2",
            )
        return self._decode_wrapper

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._graph_decode_wrapper = None
        self._graph_seq_lens = torch.ones(max_batch_size,
                                          dtype=torch.int32,
                                          device=self.runner.device)
        self._graph_decode_workspace_buffer = self._get_workspace_buffer()
        self._graph_indices_buffer = torch.empty(
            max_batch_size * self.runner.cache_config.num_gpu_blocks,
            dtype=torch.int32,
            device=self.runner.device)
        self._graph_query_start_loc = torch.arange(0,
                                                   max_batch_size + 1,
                                                   dtype=torch.int32,
                                                   device=self.runner.device)
        self._graph_indptr_buffer = torch.empty(max_batch_size + 1,
                                                dtype=torch.int32,
                                                device=self.runner.device)
        with super().graph_capture(max_batch_size):
            yield

        del self._graph_decode_workspace_buffer
        del self._graph_indices_buffer
        del self._graph_indptr_buffer
        del self._graph_decode_wrapper

    def graph_clone(self, batch_size: int):
        assert self._is_graph_capturing
        state = self.__class__(self.runner)
        state._workspace_buffer = self._graph_decode_workspace_buffer
        state._decode_wrapper = self._graph_decode_wrapper
        return state

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing

        self._graph_decode_wrapper = BatchMLAPagedAttentionWrapper(
            self._get_workspace_buffer(),
            use_cuda_graph=True,
            qo_indptr=self._graph_query_start_loc[:batch_size + 1],
            kv_indptr=self._graph_indptr_buffer[:batch_size + 1],
            kv_indices=self._graph_indices_buffer,
            kv_len_arr=self._graph_seq_lens[:batch_size],
            backend="fa2",
        )
        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        paged_kv_indptr_tensor_host = torch.zeros((batch_size + 1, ),
                                                  dtype=torch.int32)
        paged_kv_indices_tensor_host = torch.arange(0,
                                                    batch_size,
                                                    dtype=torch.int32)
        query_start_loc_host = torch.arange(0,
                                            batch_size + 1,
                                            dtype=torch.int32)
        seq_lens_tensor_host = torch.zeros((batch_size, ), dtype=torch.int32)
        global_params = infer_global_hyperparameters(self.vllm_config,
                                                     FlashInferMLAImpl)
        metadata = super().graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)

        metadata.num_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        metadata.paged_kv_indices_host = paged_kv_indices_tensor_host
        metadata.paged_kv_indptr_host = paged_kv_indptr_tensor_host
        metadata.query_start_loc_host = query_start_loc_host
        metadata.seq_lens_tensor_host = seq_lens_tensor_host
        metadata.data_type = kv_cache_dtype
        metadata.q_data_type = self.runner.model_config.dtype
        metadata.sm_scale = global_params.sm_scale
        metadata.device = self.runner.device
        metadata.decode_wrapper = self._graph_decode_wrapper
        self._decode_wrapper_plan(metadata)
        return metadata

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)

    def _decode_wrapper_plan(self, metadata: FlashInferMLAMetadata):
        if metadata.num_decode_tokens > 0:
            assert metadata.paged_kv_indices_host is not None
            assert metadata.paged_kv_indptr_host is not None

            assert metadata.decode_wrapper is not None

            metadata.decode_wrapper.plan(
                qo_indptr=metadata.query_start_loc_host[metadata.
                                                        num_prefills:],
                kv_indptr=metadata.paged_kv_indptr_host[metadata.
                                                        num_prefills:],
                kv_indices=metadata.paged_kv_indices_host,
                kv_len_arr=metadata.seq_lens_tensor_host[metadata.
                                                         num_prefills:],
                num_heads=self.num_heads,
                head_dim_ckv=self.kv_lora_rank,
                head_dim_kpe=self.qk_rope_head_dim,
                page_size=self.page_size,
                causal=True,
                sm_scale=metadata.sm_scale,
                q_data_type=metadata.q_data_type,
                kv_data_type=metadata.data_type,
            )

    def begin_forward(self, model_input):
        assert not self._is_graph_capturing
        super().begin_forward(model_input)

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
        model_input.attn_metadata.decode_wrapper = state._get_decode_wrapper()
        self._decode_wrapper_plan(model_input.attn_metadata)


class FlashInferMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashInferMLAMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        super().__init__(input_builder)

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None
        self.vllm_config = get_current_vllm_config()

    def prepare(self):
        super().prepare()

        # Please follow
        #   https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
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
                self.vllm_config, FlashInferMLAImpl)
            self.global_hyperparameters = inferred_params
            self.window_left = inferred_params.window_left
            self.logits_soft_cap = inferred_params.logits_soft_cap
            self.sm_scale = inferred_params.sm_scale

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        The only FlashInfer specific thing we need to do is 
         `_update_paged_kv_tensors`
        """
        block_tables = inter_data.block_tables
        is_profile_run = is_block_tables_empty(inter_data.block_tables)

        super()._add_seq_group(inter_data, chunked_prefill_enabled,
                               prefix_cache_hit)

        # It is not necessary to add paged_kv_indices, paged_kv_indptr,
        # and paged_kv_last_page_len for profile run because we will
        # create dummy inputs.
        if is_profile_run:
            self.is_profile_run = is_profile_run
            return

        for (seq_id, seq_len) in zip(inter_data.seq_ids,
                                     inter_data.orig_seq_lens):
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
        metadata = \
            super().build(seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        query_start_loc_host = metadata.query_start_loc.to("cpu")
        seq_lens_tensor_host = metadata.seq_lens_tensor.to("cpu")

        if cuda_graph_pad_size > 0:
            self.paged_kv_indptr.extend([self.paged_kv_indptr[-1]] *
                                        cuda_graph_pad_size)

            query_start_loc_host = torch.cat(
                (query_start_loc_host,
                 torch.full((cuda_graph_pad_size, ),
                            fill_value=query_start_loc_host[-1].item(),
                            dtype=torch.int32,
                            device="cpu")))

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
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None

        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        metadata.paged_kv_indices_host = paged_kv_indices_tensor
        metadata.paged_kv_indptr_host = paged_kv_indptr_tensor
        metadata.query_start_loc_host = query_start_loc_host
        metadata.seq_lens_tensor_host = seq_lens_tensor_host
        metadata.data_type = kv_cache_dtype
        metadata.q_data_type = self.runner.model_config.dtype
        metadata.sm_scale = self.global_hyperparameters.sm_scale
        metadata.device = self.runner.device
        return metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        raise NotImplementedError


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

        if Version(flashinfer.__version__) < Version("0.2.3"):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support FlashInfer version < 0.2.3"
            )

        unsupported_features = [
            alibi_slopes,
            blocksparse_params,
        ]

        self.sliding_window = ((sliding_window - 1, 0) \
            if sliding_window is not None else (-1, -1))
        self.logits_soft_cap = logits_soft_cap

        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 FlashInfer MLA not yet supported")

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank:]

        o = decode_meta.decode_wrapper.run(
            q_nope,
            q_pe,
            kv_c_cache,
            k_pe_cache,
            return_lse=False,
        )
        return self._v_up_proj_and_o_proj(o)
