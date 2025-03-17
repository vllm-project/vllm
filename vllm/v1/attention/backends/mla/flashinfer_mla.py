# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from packaging.version import Version

from vllm.attention.backends.abstract import AttentionType
from vllm.attention.backends.utils import (PerLayerParameters, get_mla_dims,
                                           infer_global_hyperparameters)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)

try:
    import flashinfer
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 512 * 1024 * 1024
except ImportError:
    # Avoid turning these types into variables during type checking
    if not TYPE_CHECKING:
        BatchMLAPagedAttentionWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

logger = init_logger(__name__)


class FlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["FlashInferMLAMetadata"]:
        return FlashInferMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    decode_wrapper: BatchMLAPagedAttentionWrapper


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashInferMLAMetadata]):

    def __init__(self, runner):
        super().__init__(runner)

        # TODO: Tune this parameter
        self.max_decode_q_len = 16

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters: Optional[PerLayerParameters] = None

        self.vllm_config = get_current_vllm_config()

        self.kv_lora_rank, self.qk_rope_head_dim = get_mla_dims(
            self.runner.model_config)
        self.num_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        self.page_size = self.runner.block_size
        self.kv_cache_dtype = self.runner.kv_cache_dtype
        self.q_dtype = self.runner.model_config.dtype

        self._workspace_buffer = torch.empty(FLASHINFER_WORKSPACE_BUFFER_SIZE,
                                             dtype=torch.uint8,
                                             device=self.runner.device)

        self._decode_wrapper = BatchMLAPagedAttentionWrapper(
            self._workspace_buffer,
            backend="fa2",
        )

        self._paged_kv_indptr_host = None

    def _build_decode(self, num_decodes: int, num_decode_tokens: int,
                      input_positions: torch.Tensor, block_table: torch.Tensor,
                      seq_lens: torch.Tensor) -> FlashInferMLADecodeMetadata:

        # Infer on first build so it happens after the dummy run
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

        # An example for paged_kv_indices, paged_kv_indptr:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        #
        # For efficiency we flatten the block_table into `paged_kv_indices`,
        # this means every requests page indices starts at a multiple of
        # `max_num_blocks_per_req`, so we construct `paged_kv_indptr` to be:
        # [0, 1, 2, 3 ..., max_num_reqs - 1] * max_num_blocks_per_req

        input_batch = self.runner.input_batch
        if self._paged_kv_indptr_host is None:
            self._paged_kv_indptr_host = torch.arange(
                input_batch.max_num_reqs, dtype=torch.int32,
                device="cpu") * input_batch.max_num_blocks_per_req

        query_start_loc_host = self.runner.query_start_loc_cpu[:num_decodes +
                                                               1]
        paged_kv_indptr_host = self._paged_kv_indptr_host[:num_decodes]
        paged_kv_indices_host = input_batch.block_table.\
            block_table[:num_decodes, ...].flatten()

        seq_lens_host = self.runner.seq_lens_cpu[:num_decodes]

        decode_wrapper = self._decode_wrapper
        decode_wrapper.plan(
            qo_indptr=query_start_loc_host,
            kv_indptr=paged_kv_indptr_host,
            kv_indices=paged_kv_indices_host,
            kv_len_arr=seq_lens_host,
            num_heads=self.num_heads,
            head_dim_ckv=self.kv_lora_rank,
            head_dim_kpe=self.qk_rope_head_dim,
            page_size=self.page_size,
            causal=True,
            sm_scale=self.sm_scale,
            q_data_type=self.q_dtype,
            kv_data_type=self.kv_cache_dtype,
        )

        return FlashInferMLADecodeMetadata(
            decode_wrapper=decode_wrapper,
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
        )


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
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

        decode_meta = attn_metadata.decode
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
