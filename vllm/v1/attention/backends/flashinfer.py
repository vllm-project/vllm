# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

import numpy as np
import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)
from flashinfer.decode import _get_range_buf, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey, kFp8StaticTensorSym, kNvfp4Quant)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import cdiv, is_pin_memory_available
from vllm.utils.flashinfer import (flashinfer_disable_q_quantization,
                                   supports_trtllm_attention,
                                   use_trtllm_attention)
from vllm.v1.attention.backends.flash_attn import use_cascade_attention
# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.attention.backends.utils import (AttentionCGSupport,
                                              AttentionMetadataBuilder,
                                              CommonAttentionMetadata,
                                              get_kv_cache_layout,
                                              get_per_layer_parameters,
                                              infer_global_hyperparameters,
                                              split_decodes_and_prefills)
# yapf: enable
from vllm.v1.kv_cache_interface import AttentionSpec

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(block_tables_prefill_ptr +
                            batch_idx * block_table_stride +
                            mock_block_table_idx).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    # Dequantize K
    k_scale_val = tl.load(k_scale_ptr)
    offset = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * k_scale_val
    mock_cache_offset = (batch_idx * block_table_stride + mock_block_table_idx
                         + 1) * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)

    # Dequantize V
    v_scale_val = tl.load(v_scale_ptr)
    offset = (orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE +
              tl.arange(0, K_CACHE_STRIDE))
    fp8_vals = tl.load(kv_cache_ptr + offset)
    dequantized_vals = fp8_vals.to(tl.float32) * v_scale_val
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) *
        KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE))
    dequantized_vals = dequantized_vals.to(dequant_dtype)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_vals)


def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)
    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s,
                                dtype=dequant_dtype,
                                device=kv_cache.device)
    # we simply sequentially index the pages needed by this prefill
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device=block_tables_prefill.device,
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table


class FlashInferBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def get_supported_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256]

    @classmethod
    def validate_head_size(cls, head_size: int) -> None:
        supported_head_sizes = cls.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            attn_type = cls.__name__.removesuffix("Backend")
            raise ValueError(
                f"Head size {head_size} is not supported by {attn_type}. "
                f"Supported head sizes are: {supported_head_sizes}. "
                "Set VLLM_ATTENTION_BACKEND=FLEX_ATTENTION to use "
                "FlexAttention backend which supports all head sizes.")

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

    @staticmethod
    def get_kv_cache_stride_order() -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND":
            stride_order = (0, 1, 3, 2, 4)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


@dataclass
class FlashInferMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.

    # The data type of the query
    q_data_type: torch.dtype

    slot_mapping: torch.Tensor

    # For flashinfer trtllm batch decode
    max_q_len: int
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table_tensor: torch.Tensor
    prefill_use_trtllm: bool
    decode_use_trtllm: bool

    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # For cascade attention (CPU for planning).
    use_cascade: bool

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
    cascade_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None

    qo_indptr_gpu: Optional[torch.Tensor] = None
    paged_kv_indptr_gpu: Optional[torch.Tensor] = None


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    reorder_batch_threshold: ClassVar[int] = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self._workspace_buffer = None
        self._prefill_wrapper = None  # Wrapper for prefill/append
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(self.model_config.max_model_len,
                                     self.kv_cache_spec.block_size)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        self.enable_cuda_graph = (self.compilation_config.cudagraph_mode.\
            decode_mode() == CUDAGraphMode.FULL)
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper] = {}
            self._decode_cudagraph_max_bs = min(
                max_num_reqs, self.compilation_config.max_capture_size)

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config)
        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        FlashInferBackend.validate_head_size(self.head_dim)
        self.page_size = self.kv_cache_spec.block_size

        self.cache_dtype = self.cache_config.cache_dtype
        if self.cache_dtype.startswith("fp8"):
            self.kv_cache_dtype = (
                FlashInferBackend.get_fp8_dtype_for_flashinfer(
                    self.cache_dtype))
        else:
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # Use model dtype as q dtype when TRTLLM attn is not supported, or
        # VLLM_FLASHINFER_DISABLE_Q_QUANTIZATION is set to 1. Otherwise, try to
        # use fp8 q if kv cache is fp8, and will fall back to model dtype
        # if TRTLLM attention kernel is not used when building attn metadata
        if supports_trtllm_attention() and \
            not flashinfer_disable_q_quantization():
            self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl))
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks and not supports_trtllm_attention():
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs.")
        # Preparing persistent buffers (device-side)
        self.paged_kv_indptr = torch.zeros(max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.paged_kv_indices = torch.zeros(
            max_num_pages,  # max num pages possible
            dtype=torch.int32,
            device=self.device)
        self.paged_kv_last_page_len = torch.zeros(max_num_reqs,
                                                  dtype=torch.int32,
                                                  device=self.device)
        # host-side buffer
        pin_memory = is_pin_memory_available()
        self.paged_kv_indptr_cpu = torch.zeros(max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=pin_memory)
        self.paged_kv_indptr_np = self.paged_kv_indptr_cpu.numpy()
        self.paged_kv_indptr_buffer = torch.zeros_like(
            self.paged_kv_indptr_cpu, pin_memory=pin_memory)
        self.paged_kv_indices_cpu = torch.zeros(max_num_pages,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pin_memory=pin_memory)
        self.paged_kv_last_page_len_cpu = torch.zeros(max_num_reqs,
                                                      dtype=torch.int32,
                                                      device="cpu",
                                                      pin_memory=pin_memory)
        self.paged_kv_last_page_len_np = (
            self.paged_kv_last_page_len_cpu.numpy())

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.zeros(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.device)
        return self._workspace_buffer

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), get_kv_cache_layout())
        return self._prefill_wrapper

    def _get_decode_wrapper(self,
                            batch_size: int,
                            use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(
                batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr[:batch_size + 1]
                paged_kv_indices = self.paged_kv_indices
                paged_kv_last_page_len = self.paged_kv_last_page_len[:
                                                                     batch_size]
            else:
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_last_page_len = None
            decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_cuda_graph=use_cudagraph,
                paged_kv_indptr_buffer=paged_kv_indptr,
                paged_kv_indices_buffer=paged_kv_indices,
                paged_kv_last_page_len_buffer=paged_kv_last_page_len,
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
                use_tensor_cores=True,
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout())
        return self._cascade_wrapper

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens =\
            split_decodes_and_prefills(common_attn_metadata,
                                       decode_threshold=self.reorder_batch_threshold)

        page_size = self.page_size
        max_q_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()
        block_table_tensor = common_attn_metadata.block_table_tensor

        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size

        use_cascade = common_prefix_len > 0
        if use_cascade:
            # Grab the blocks of the shared prefix from the first request.
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device='cpu')
            shared_kv_page_indptr_cpu = torch.tensor([0, num_common_kv_blocks],
                                                     dtype=torch.int32,
                                                     device='cpu')
            shared_kv_page_indices_cpu = block_table_tensor[
                0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor([page_size],
                                                       dtype=torch.int32,
                                                       device='cpu')

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks
        else:
            shared_qo_indptr_cpu = None
            shared_kv_page_indptr_cpu = None
            shared_kv_page_indices_cpu = None
            shared_kv_last_page_len_cpu = None

        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr_np[1:num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_buffer[:num_reqs +
                                    1] = (self.paged_kv_indptr_cpu[:num_reqs +
                                                                   1])
        paged_kv_indptr = self.paged_kv_indptr[:num_reqs + 1]
        paged_kv_indptr.copy_(self.paged_kv_indptr_buffer[:num_reqs + 1],
                              non_blocking=True)

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr_np[num_reqs]
        paged_kv_indices = self.paged_kv_indices[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs, )](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len_np[:num_reqs] = np.where(
            paged_kv_last_page_len_np == 0,
            page_size,
            paged_kv_last_page_len_np,
        )

        prefill_use_trtllm = use_trtllm_attention(self.num_qo_heads,
                                                  self.num_kv_heads,
                                                  num_prefill_tokens,
                                                  max_seq_len,
                                                  self.cache_dtype,
                                                  self.q_data_type,
                                                  has_sinks=self.has_sinks)
        decode_use_trtllm = use_trtllm_attention(self.num_qo_heads,
                                                 self.num_kv_heads,
                                                 num_decode_tokens,
                                                 max_seq_len,
                                                 self.cache_dtype,
                                                 self.q_data_type,
                                                 has_sinks=self.has_sinks)
        if self.has_sinks and not (prefill_use_trtllm and decode_use_trtllm):
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs.")

        # If TRTLLM attention is not used, the q quantization is not supported.
        # Fall back to use model dtype.
        if not (prefill_use_trtllm and decode_use_trtllm):
            self.q_data_type = self.model_config.dtype

        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            q_data_type=self.q_data_type,
            slot_mapping=common_attn_metadata.slot_mapping,
            max_q_len=max_q_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table_tensor=block_table_tensor,
            prefill_use_trtllm=prefill_use_trtllm,
            decode_use_trtllm=decode_use_trtllm,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
        )

        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
        paged_kv_indptr_cpu = self.paged_kv_indptr_cpu[:1 + num_reqs]
        paged_kv_last_page_len_cpu = self.paged_kv_last_page_len_cpu[:num_reqs]

        if attn_metadata.use_cascade:
            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                [shared_qo_indptr_cpu, qo_indptr_cpu],
                [shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                [shared_kv_page_indices_cpu, paged_kv_indices],
                [shared_kv_last_page_len_cpu, paged_kv_last_page_len_cpu],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                window_left=self.window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
            )
        else:
            # Regular attention (common case).
            # Decodes are at the front and prefills are at the back,
            # according to reorder_batch()
            num_prefills = attn_metadata.num_prefills
            num_decodes = attn_metadata.num_decodes
            if num_prefills > 0:
                # Decodes are first so prefills start after the last decode
                prefill_start = num_decodes
                attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
                assert qo_indptr_cpu[prefill_start:].shape[
                    0] == num_prefills + 1
                assert paged_kv_indptr_cpu[prefill_start:].shape[
                    0] == num_prefills + 1
                assert paged_kv_last_page_len_cpu[prefill_start:].shape[
                    0] == num_prefills
                # Since prefill_wrapper.run() will be called with
                # query[num_decode_tokens:] we need to adjust the qo_indptr
                # to be relative to the start of the prefill queries.
                qo_indptr_cpu = qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[
                    prefill_start]
                paged_kv_indptr_cpu = paged_kv_indptr_cpu[prefill_start:]
                if not attn_metadata.prefill_use_trtllm:
                    attn_metadata.prefill_wrapper.plan(
                        qo_indptr_cpu,
                        paged_kv_indptr_cpu,
                        paged_kv_indices,
                        paged_kv_last_page_len_cpu[prefill_start:],
                        self.num_qo_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.page_size,
                        causal=True,
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                    )
                else:
                    attn_metadata.qo_indptr_gpu = qo_indptr_cpu.to(
                        self.device, non_blocking=True)
                    attn_metadata.paged_kv_indptr_gpu = paged_kv_indptr_cpu.to(
                        self.device, non_blocking=True)

            if num_decodes > 0:
                pure_decode = num_prefills == 0
                # possible required padding for cudagraph replay
                use_cudagraph = (self.enable_cuda_graph and pure_decode and
                                 num_decodes <= self._decode_cudagraph_max_bs)
                if use_cudagraph:
                    num_input_tokens = (
                        self.vllm_config.pad_for_cudagraph(num_decodes))
                    # Carefully fulfill the padding region with reasonable value
                    # on cpu.
                    # Make sure paged_kv_indptr_cpu is not decreasing
                    self.paged_kv_indptr_cpu[1 + num_decodes:1 +
                                             num_input_tokens].fill_(
                                                 paged_kv_indptr_cpu[-1])
                    # Fill the remaining paged_kv_last_page_len_cpu with 1.
                    # This is because flashinfer treats 0 as a full page
                    # instead of empty.
                    self.paged_kv_last_page_len_cpu[
                        num_decodes:num_input_tokens].fill_(1)

                else:
                    num_input_tokens = num_decodes

                attn_metadata.decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph)
                if not attn_metadata.decode_use_trtllm:
                    # Use the persistent buffer with padding length,
                    # instead of the same address but chunked version
                    # in atten_metadata when using cudagraph.
                    fast_plan_decode(
                        attn_metadata.decode_wrapper,
                        self.paged_kv_indptr_cpu[:num_input_tokens + 1],
                        paged_kv_indices,
                        self.paged_kv_last_page_len_cpu[:num_input_tokens],
                        seq_lens_cpu[:num_input_tokens],
                        self.num_qo_heads,
                        self.num_kv_heads,
                        self.head_dim,
                        self.page_size,
                        # Disable flashinfer's pos encoding and use vllm's rope.
                        pos_encoding_mode="NONE",
                        sm_scale=self.sm_scale,
                        window_left=self.window_left,
                        logits_soft_cap=self.logits_soft_cap,
                        q_data_type=self.q_data_type,
                        kv_data_type=self.kv_cache_dtype,
                    )
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
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
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[int] = None,
        sinks: Optional[torch.Tensor] = None,
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
        self.window_left = (self.sliding_window[0]
                            if self.sliding_window is not None else -1)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferImpl")

        self.sinks: Optional[torch.Tensor] = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}.")
            self.sinks = sinks

        self.support_trtllm_attn = (supports_trtllm_attention()
                                    and num_heads % num_kv_heads == 0)
        self.bmm1_scale: Optional[float] = None
        self.bmm2_scale: Optional[float] = None
        self.o_sf_scale: Optional[float] = None

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return (self.support_trtllm_attn
                and self.kv_cache_dtype.startswith("fp8")
                and quant_key in (kFp8StaticTensorSym, kNvfp4Quant))

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: Optional[torch.Tensor] = None,
        output_scale: Optional[torch.Tensor] = None,
        output_block_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: KV cache tensor with different possible shapes:
                - NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
                - HND: [num_blocks, 2, num_kv_heads, block_size, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        if self.bmm1_scale is None:
            self.bmm1_scale = (layer._q_scale_float * layer._k_scale_float *
                               self.scale)

        if self.bmm2_scale is None:
            self.bmm2_scale = layer._v_scale_float

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, "output_block_scale "\
                "is not supported when fusion has not happened"
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE, \
                "Query must be FP8 when attn+quant fusion happened."
            assert (attn_metadata.prefill_use_trtllm and
                    attn_metadata.decode_use_trtllm), "Must use TRT-LLM attn"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, \
                    "output_block_scale should not be provided for fp8 output"
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, \
                    "output_block_scale is required for nvfp4 output"
            else:
                raise ValueError(f"Unsupported output dtype: {output.dtype}")

            # TRTLLM attn kernel requires to scale to pass as a host scalar,
            # store the o scale as a host scalar in warmup run with cuda graph
            # not enabled
            if layer._o_scale_float is None:
                layer._o_scale_float = output_scale.cpu().item()
                if output.dtype == FP8_DTYPE:
                    self.bmm2_scale = self.bmm2_scale / layer._o_scale_float
                elif output.dtype == FP4_DTYPE:
                    self.o_sf_scale = layer._o_scale_float

        # Insert FP8 quant for query
        if attn_metadata.q_data_type == FP8_DTYPE:
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
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

            # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
            # to process the cache when the kv_cache_dtype is fp8
            if self.kv_cache_dtype.startswith("fp8"):
                torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                    self.kv_cache_dtype)
                kv_cache = kv_cache.view(torch_dtype)

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

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)
        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back,
        # according to reorder_batch()
        if num_prefill_tokens > 0:
            prefill_wrapper = attn_metadata.prefill_wrapper
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens
            assert prefill_wrapper is not None

            if not attn_metadata.prefill_use_trtllm:
                assert prefill_wrapper._causal
                assert prefill_wrapper._window_left == self.window_left
                assert prefill_wrapper._logits_soft_cap == (
                    self.logits_soft_cap or 0.0)
                assert prefill_wrapper._sm_scale == self.scale
                prefill_wrapper.run(
                    prefill_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[num_decode_tokens:],
                )
            else:
                # prefill_query may be non-contiguous
                prefill_query = prefill_query.contiguous()
                workspace_buffer = prefill_wrapper._float_workspace_buffer
                block_tables_prefill = attn_metadata.block_table_tensor[
                    num_decode_tokens:]
                seq_lens_prefill = attn_metadata.seq_lens[num_decode_tokens:]

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert prefill_query.is_contiguous()
                assert kv_cache_permute.is_contiguous()
                assert workspace_buffer.is_contiguous()
                assert block_tables_prefill.is_contiguous()
                assert seq_lens_prefill.is_contiguous()

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(data=output[num_decode_tokens:],
                                    scale=output_block_scale,
                                    scale_start_index=num_decode_tokens,
                                    original_shape=prefill_query.shape)
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                if attn_metadata.q_data_type != FP8_DTYPE \
                    and self.kv_cache_dtype.startswith("fp8"):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill
                    mock_kv_cache, mock_block_table = (
                        trtllm_prefill_attn_kvfp8_dequant(
                            kv_cache_permute,
                            block_tables_prefill,
                            layer._k_scale,
                            layer._v_scale,
                            attn_metadata.q_data_type,
                        ))
                else:
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.max_q_len,
                    max_kv_len=attn_metadata.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.qo_indptr_gpu,
                    cum_seq_lens_kv=attn_metadata.paged_kv_indptr_gpu,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                )

        if num_decode_tokens > 0:
            decode_wrapper = attn_metadata.decode_wrapper
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens
            assert decode_wrapper is not None

            if not attn_metadata.decode_use_trtllm:
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap
                                                           or 0.0)
                assert decode_wrapper._sm_scale == self.scale
                decode_wrapper.run(
                    decode_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[:num_decode_tokens],
                )
            else:
                # decode_query may be non-contiguous
                decode_query = decode_query.contiguous()
                workspace_buffer = decode_wrapper._float_workspace_buffer
                block_tables_decode = attn_metadata.\
                    block_table_tensor[:num_decode_tokens]
                seq_lens_decode = attn_metadata.seq_lens[:num_decode_tokens]

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert decode_query.is_contiguous()
                assert kv_cache_permute.is_contiguous()
                assert workspace_buffer.is_contiguous()
                assert block_tables_decode.is_contiguous()
                assert seq_lens_decode.is_contiguous()

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(data=output[:num_decode_tokens],
                                    scale=output_block_scale,
                                    scale_start_index=0,
                                    original_shape=decode_query.shape)
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=kv_cache_permute,
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                )
        return output_padded


def fast_plan_decode(
    self,  # decode wrapper
    indptr_cpu: torch.Tensor,
    indices: torch.Tensor,
    last_page_len_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str = "NONE",
    window_left: int = -1,
    logits_soft_cap: Optional[float] = None,
    q_data_type: Optional[Union[str, torch.dtype]] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    data_type: Optional[Union[str, torch.dtype]] = None,
    sm_scale: Optional[float] = None,
    rope_scale: Optional[float] = None,
    rope_theta: Optional[float] = None,
    non_blocking: bool = True,
) -> None:
    """
    A faster version of BatchDecodeWithPagedKVCacheWrapper::plan used for
    cudagraph capture/replay, while the no cudagraph version turns back
    to the original plan.
    using original plan after passing host-side buffers:
    - only host-to-device copy of indptr and last_page_len buffers
    Modifications for cudagraph:
    - only host-to-device copy of indptr and last_page_len buffers.
    - avoid device-to-device copy of indices buffer.

    Part of the code get inspiration from the original plan from FlashInfer repo
    and the implementation of fast_decode_plan for FlashInfer in SGlang repo.
    """
    # Warm up with the original plan if it is first call, and always run the
    # original plan if we run for dynamic shape. For fixed shape (cudagraph),
    # this warm up is to generate the _cached_module for the decode wrapper.
    if not self.is_cuda_graph_enabled or \
        getattr(self, "vllm_first_call", True):
        self.plan(
            indptr_cpu,
            indices,
            last_page_len_cpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode,
            window_left,
            logits_soft_cap,
            q_data_type,
            kv_data_type,
            data_type,
            sm_scale,
            rope_scale,
            rope_theta,
            non_blocking,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    batch_size = len(last_page_len_cpu)
    if logits_soft_cap is None:
        logits_soft_cap = 0.0

    # Handle data types consistently
    if data_type is not None:
        if q_data_type is None:
            q_data_type = data_type
        if kv_data_type is None:
            kv_data_type = data_type
    elif q_data_type is None:
        q_data_type = "float16"

    if kv_data_type is None:
        kv_data_type = q_data_type
    q_data_type = getattr(torch, q_data_type) if isinstance(
        q_data_type, str) else q_data_type
    kv_data_type = getattr(torch, kv_data_type) if isinstance(
        kv_data_type, str) else kv_data_type

    if batch_size != self._fixed_batch_size:
        raise ValueError(
            "The batch size should be fixed in cudagraph mode, the runtime "
            "batch size {} mismatches the batch size set during "
            "initialization {}".format(batch_size, self._fixed_batch_size))
    if len(indices) > len(self._paged_kv_indices_buf):
        raise ValueError(
            "The size of indices should be less than or equal to the "
            "allocated buffer")

    # host-to-device copy for the indptr buffer
    self._paged_kv_indptr_buf.copy_(indptr_cpu, non_blocking=True)
    # host-to-device copy for the last_page_len buffer
    self._paged_kv_last_page_len_buf.copy_(last_page_len_cpu,
                                           non_blocking=True)

    qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")

    try:
        # Make sure we pass exactly 15 arguments for tensor core version
        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            indptr_cpu,
            seq_lens_cpu,
            batch_size,  # total_num_rows
            batch_size,
            num_qo_heads,
            num_kv_heads,
            page_size,
            self.is_cuda_graph_enabled,
            head_dim,
            head_dim,
            False,  # causal
        )
    except Exception as e:
        raise RuntimeError(f"Error in tensor core plan: {e}") from e

    self._pos_encoding_mode = pos_encoding_mode
    self._window_left = window_left
    self._logits_soft_cap = logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta


@triton.jit
def _copy_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_blocks + req_idx)
    end_idx = tl.load(cu_num_blocks + req_idx + 1)
    num_blocks = end_idx - start_idx

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        block_ids = tl.load(row_ptr + i + offset, mask=i + offset < num_blocks)
        tl.store(page_indices + start_idx + i + offset,
                 block_ids,
                 mask=i + offset < num_blocks)
