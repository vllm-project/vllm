# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils import cdiv
from vllm.v1.attention.backends.flash_attn import use_cascade_attention
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder, CommonAttentionMetadata, get_kv_cache_layout,
    get_per_layer_parameters, infer_global_hyperparameters,
    reorder_batch_to_split_decodes_and_prefills, split_decodes_and_prefills)
from vllm.v1.kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

logger = init_logger(__name__)


class FlashInferBackend(AttentionBackend):

    accept_output_buffer: bool = True
    cached_sm100a_supported: Optional[bool] = None

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
    def use_trtllm_decode_attention(
        batch_size: int,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_qo_heads: int,
        num_kv_heads: int,
        attn_head_size: int,
    ) -> bool:
        if FlashInferBackend.cached_sm100a_supported is None:
            FlashInferBackend.cached_sm100a_supported = (
                current_platform.has_device_capability(100))
        if not FlashInferBackend.cached_sm100a_supported:
            return False
        if (num_qo_heads // num_kv_heads > 8
                or num_qo_heads % num_kv_heads != 0 or attn_head_size != 128):
            return False
        env_value = envs.VLLM_USE_TRTLLM_DECODE_ATTENTION
        if env_value is not None:
            logger.info_once("VLLM_USE_TRTLLM_DECODE_ATTENTION is set to %s",
                             env_value)
            # Environment variable is set - respect it
            # Making the conditional check for zero because
            # the path is automatically enabled if the batch size condition
            # is satisfied.
            no_use_trtllm = env_value == "0"
            if not no_use_trtllm:
                logger.info_once(
                    "VLLM_USE_TRTLLM_DECODE_ATTENTION is set to 1, "
                    "using TRTLLM decode attention.")
            return not no_use_trtllm
        else:
            # Environment variable not set - use auto-detection
            # Only supports attention head size of 128
            use_trtllm = (FlashInferBackend.cached_sm100a_supported
                          and batch_size <= 256 and max_seq_len < 131072
                          and kv_cache_dtype == "auto")
            if use_trtllm:
                logger.warning_once(
                    "Using TRTLLM decode attention (auto-detected).")
        return use_trtllm

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

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    qo_indptr_cpu: torch.Tensor
    # An example for paged_kv_indices, paged_kv_indptr:
    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1] (CPU for plan)
    paged_kv_indptr_cpu: torch.Tensor
    # The page indices of the paged kv cache (on device for plan)
    paged_kv_indices: torch.Tensor
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size] (CPU for plan)
    paged_kv_last_page_len_cpu: torch.Tensor
    # The number of query/output heads
    num_qo_heads: int
    # The number of key/value heads
    num_kv_heads: int
    # The dimension of the attention heads
    head_dim: int
    # Block size of vllm
    page_size: int
    # The data type of the paged kv cache
    kv_data_type: torch.dtype
    # The data type of the query
    q_data_type: torch.dtype

    slot_mapping: torch.Tensor

    # For flashinfer trtllm batch decode
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table_tensor: torch.Tensor

    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    # For cascade attention (CPU for planning).
    use_cascade: bool
    shared_qo_indptr_cpu: Optional[torch.Tensor] = None
    shared_kv_page_indptr_cpu: Optional[torch.Tensor] = None
    shared_kv_page_indices_cpu: Optional[torch.Tensor] = None
    shared_kv_last_page_len_cpu: Optional[torch.Tensor] = None

    prefill_wrapper: Optional[BatchPrefillWithPagedKVCacheWrapper] = None
    decode_wrapper: Optional[BatchDecodeWithPagedKVCacheWrapper] = None
    cascade_wrapper: Optional[MultiLevelCascadeAttentionWrapper] = None

    def __post_init__(self):
        if self.head_dim is not None:
            FlashInferBackend.validate_head_size(self.head_dim)


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        self.device = device
        self._workspace_buffer = None
        self._prefill_wrapper = None  # Wrapper for prefill/append
        self._decode_wrapper = None  # Wrapper for decode
        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl))

        self.vllm_config = vllm_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_spec = kv_cache_spec
        max_num_blocks_per_request = cdiv(
            vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size)
        self.block_table_arange = torch.arange(max_num_blocks_per_request,
                                               dtype=torch.int32,
                                               device=self.device)

    def reorder_batch(self, input_batch: InputBatch,
                      scheduler_output: SchedulerOutput) -> bool:
        return reorder_batch_to_split_decodes_and_prefills(input_batch,
                                                           scheduler_output,
                                                           decode_threshold=1)

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.device)
        return self._workspace_buffer

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(), get_kv_cache_layout())
        return self._prefill_wrapper

    def _get_decode_wrapper(self):
        if self._decode_wrapper is None:
            num_qo_heads = (
                self.vllm_config.model_config.get_num_attention_heads(
                    self.vllm_config.parallel_config))
            num_kv_heads = self.vllm_config.model_config.get_num_kv_heads(
                self.vllm_config.parallel_config)
            use_tensor_cores = envs.VLLM_FLASHINFER_FORCE_TENSOR_CORES or (
                num_qo_heads // num_kv_heads > 4)
            self._decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                use_tensor_cores=use_tensor_cores)
        return self._decode_wrapper

    def _get_cascade_wrapper(self):
        if self._cascade_wrapper is None:
            self._cascade_wrapper = MultiLevelCascadeAttentionWrapper(
                2, self._get_workspace_buffer(), get_kv_cache_layout())
        return self._cascade_wrapper

    def _plan(self, num_prefills: int, num_decodes: int,
              attn_metadata: FlashInferMetadata):
        if attn_metadata.use_cascade:
            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                [
                    attn_metadata.shared_qo_indptr_cpu,
                    attn_metadata.qo_indptr_cpu
                ],
                [
                    attn_metadata.shared_kv_page_indptr_cpu,
                    attn_metadata.paged_kv_indptr_cpu
                ],
                [
                    attn_metadata.shared_kv_page_indices_cpu,
                    attn_metadata.paged_kv_indices
                ],
                [
                    attn_metadata.shared_kv_last_page_len_cpu,
                    attn_metadata.paged_kv_last_page_len_cpu
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
                kv_data_type=attn_metadata.kv_data_type,
            )
        else:
            # Regular attention (common case).
            # Decodes are at the front and prefills are at the back,
            # according to reorder_batch()
            if num_prefills > 0:
                # Decodes are first so prefills start after the last decode
                prefill_start = num_decodes
                attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
                assert attn_metadata.qo_indptr_cpu[prefill_start:].shape[
                    0] == num_prefills + 1
                assert attn_metadata.paged_kv_indptr_cpu[prefill_start:].shape[
                    0] == num_prefills + 1
                assert attn_metadata.paged_kv_last_page_len_cpu[
                    prefill_start:].shape[0] == num_prefills
                # Since prefill_wrapper.run() will be called with
                # query[num_decode_tokens:] we need to adjust the qo_indptr
                # to be relative to the start of the prefill queries.
                qo_indptr_cpu = attn_metadata.qo_indptr_cpu[
                    prefill_start:] - attn_metadata.qo_indptr_cpu[prefill_start]
                attn_metadata.prefill_wrapper.plan(
                    qo_indptr_cpu,
                    attn_metadata.paged_kv_indptr_cpu[prefill_start:],
                    attn_metadata.paged_kv_indices,
                    attn_metadata.paged_kv_last_page_len_cpu[prefill_start:],
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
                    kv_data_type=attn_metadata.kv_data_type,
                )

            if num_decodes > 0:
                attn_metadata.decode_wrapper = self._get_decode_wrapper()
                if not FlashInferBackend.use_trtllm_decode_attention(
                        num_decodes, attn_metadata.max_seq_len,
                        self.cache_config.cache_dtype,
                        attn_metadata.num_qo_heads, attn_metadata.num_kv_heads,
                        attn_metadata.head_dim):
                    attn_metadata.decode_wrapper.plan(
                        attn_metadata.paged_kv_indptr_cpu[:num_decodes + 1],
                        attn_metadata.paged_kv_indices,
                        attn_metadata.paged_kv_last_page_len_cpu[:num_decodes],
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
                        kv_data_type=attn_metadata.kv_data_type,
                    )

    def build(self,
              common_prefix_len: int,
              common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> FlashInferMetadata:
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens =\
            split_decodes_and_prefills(common_attn_metadata)

        page_size = self.kv_cache_spec.block_size
        max_seq_len = common_attn_metadata.seq_lens_cpu.max()
        seq_lens = common_attn_metadata.seq_lens
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        block_table_tensor = common_attn_metadata.block_table_tensor

        block_table_bounds_cpu = (seq_lens_cpu + page_size - 1) // page_size

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
            block_table_bounds_cpu -= num_common_kv_blocks
        else:
            shared_qo_indptr_cpu = None
            shared_kv_page_indptr_cpu = None
            shared_kv_page_indices_cpu = None
            shared_kv_last_page_len_cpu = None

        max_num_blocks = block_table_bounds_cpu.max()
        block_table_bounds = block_table_bounds_cpu.to(self.device,
                                                       non_blocking=True)
        mask = (self.block_table_arange[:max_num_blocks].unsqueeze(0)
                < block_table_bounds.unsqueeze(1))
        paged_kv_indices = block_table_tensor[:, :max_num_blocks][mask]

        paged_kv_indptr_cpu = torch.zeros(len(block_table_bounds_cpu) + 1,
                                          dtype=torch.int32,
                                          device='cpu')
        paged_kv_indptr_cpu[1:] = block_table_bounds_cpu.cumsum(
            dim=0, dtype=torch.int32)

        paged_kv_last_page_len_cpu = seq_lens_cpu % page_size
        paged_kv_last_page_len_cpu = torch.where(
            paged_kv_last_page_len_cpu == 0, page_size,
            paged_kv_last_page_len_cpu)
        cache_dtype = self.cache_config.cache_dtype
        if cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                cache_dtype)
        else:
            kv_cache_dtype = self.kv_cache_spec.dtype
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            qo_indptr_cpu=common_attn_metadata.query_start_loc_cpu,
            paged_kv_indptr_cpu=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len_cpu=paged_kv_last_page_len_cpu,
            num_qo_heads=self.vllm_config.model_config.get_num_attention_heads(
                self.vllm_config.parallel_config),
            num_kv_heads=self.kv_cache_spec.num_kv_heads,
            head_dim=self.kv_cache_spec.head_size,
            page_size=page_size,
            kv_data_type=kv_cache_dtype,
            q_data_type=self.vllm_config.model_config.dtype,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            shared_qo_indptr_cpu=shared_qo_indptr_cpu,
            shared_kv_page_indptr_cpu=shared_kv_page_indptr_cpu,
            shared_kv_page_indices_cpu=shared_kv_page_indices_cpu,
            shared_kv_last_page_len_cpu=shared_kv_last_page_len_cpu,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table_tensor=block_table_tensor,
        )

        self._plan(num_prefills, num_decodes, attn_metadata)

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
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

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
        output_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape -
            # NHD: [num_blocks, 2, block_size, num_kv_heads, head_size]
            # HND: [num_blocks, 2,  num_kv_heads, block_size, head_size]


            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported"
                " for FlashInferImpl")

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

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)
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
                kv_cache_permute,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output[num_decode_tokens:],
            )
        if decode_wrapper := attn_metadata.decode_wrapper:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens
            assert decode_wrapper is not None
            if not FlashInferBackend.use_trtllm_decode_attention(
                    attn_metadata.num_decodes, attn_metadata.max_seq_len,
                    self.kv_cache_dtype, attn_metadata.num_qo_heads,
                    attn_metadata.num_kv_heads, attn_metadata.head_dim):
                assert decode_wrapper._window_left == window_left
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
                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                if num_decode_tokens > 0:
                    # decode_query may be non-contiguous
                    decode_query = decode_query.contiguous()
                    block_tables_decode = attn_metadata.block_table_tensor[:
                                                                           num_decode_tokens]
                    seq_lens_decode = attn_metadata.seq_lens[:
                                                             num_decode_tokens]
                    workspace_buffer = decode_wrapper._float_workspace_buffer

                    assert get_kv_cache_layout() == "HND"
                    assert decode_query.is_contiguous()
                    assert kv_cache_permute.is_contiguous()
                    assert block_tables_decode.is_contiguous()
                    assert seq_lens_decode.is_contiguous()
                    assert workspace_buffer.is_contiguous()

                    trtllm_batch_decode_with_kv_cache(
                        query=decode_query,
                        kv_cache=kv_cache_permute,
                        workspace_buffer=workspace_buffer,
                        block_tables=block_tables_decode,
                        seq_lens=seq_lens_decode,
                        max_seq_len=attn_metadata.max_seq_len,
                        bmm1_scale=layer._k_scale_float * self.scale,
                        bmm2_scale=layer._v_scale_float,
                        out=output[:num_decode_tokens],
                    )
        return output_padded
