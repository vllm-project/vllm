# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import _get_range_buf
from typing_extensions import override

from vllm import envs
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    KVCacheLayoutType,
    get_dcp_local_seq_lens,
    get_kv_cache_layout,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

FP8_DTYPE = current_platform.fp8_dtype()

logger = init_logger(__name__)


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
    ):
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )

    def plan(
        self,
        qo_indptr_cpu: torch.Tensor,
        paged_kv_indptr_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len_cpu: torch.Tensor,
        page_size: int,
        num_qo_heads: int,
        dcp_world_size: int,
        num_kv_heads: int,
        head_dim: int,
        sm_scale: float,
        window_left: int,
        logits_soft_cap: float | None,
        q_data_type: torch.dtype,
        kv_cache_dtype: torch.dtype,
        prefill_fixed_split_size: int,
        disable_split_kv: bool,
    ):
        """Plan the prefill operation with given parameters."""
        self._context.plan(
            qo_indptr_cpu,
            paged_kv_indptr_cpu,
            paged_kv_indices,
            paged_kv_last_page_len_cpu,
            num_qo_heads * dcp_world_size,
            num_kv_heads,
            head_dim,
            page_size,
            causal=False,  # This is context run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_cache_dtype,
            fixed_split_size=prefill_fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self._new_tokens.plan(
            qo_indptr=qo_indptr_cpu,
            kv_indptr=qo_indptr_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            head_dim_vo=head_dim,
            causal=True,  # This is newtokens run
            sm_scale=sm_scale,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
        )

    def run(
        self,
        layer: torch.nn.Module,
        prefill_query: torch.Tensor,
        kv_cache_permute: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        out: torch.Tensor,
    ):
        prefill_query_across_dcp = get_dcp_group().all_gather(
            prefill_query.contiguous(), dim=1
        )
        output_context_tmp, lse_context_tmp = self._context.run(
            prefill_query_across_dcp,
            kv_cache_permute,
            k_scale=layer._k_scale_float,
            v_scale=layer._v_scale_float,
            return_lse=True,
        )
        output_context, lse_context = cp_lse_ag_out_rs(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
            is_lse_base_on_e=False,
        )
        lse_context = lse_context.transpose(0, 1).contiguous()

        output_query, lse_query = self._new_tokens.run(
            prefill_query,
            key,
            value,
            return_lse=True,
        )
        lse_query = lse_query.transpose(0, 1).contiguous()

        merge_attn_states(
            out,
            output_context,
            lse_context,
            output_query,
            lse_query,
        )
        return out


class FlashInferBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Note: Not sure for all platforms, but on Blackwell,
        # only support a page size of 16, 32, 64.
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @staticmethod
    def get_impl_cls() -> type["FlashInferImpl"]:
        return FlashInferImpl

    @staticmethod
    def get_builder_cls() -> type["FlashInferMetadataBuilder"]:
        return FlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, 2, block_size, num_kv_heads, head_size)
            return (1, 0, 2, 3, 4, 5)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3, 4)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, 2, num_kv_heads, num_layers, block_size, head_size)
            return (1, 2, 4, 0, 3, 5)
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

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """FlashInfer native backend does not support sinks.

        Use TRTLLM_ATTN backend on Blackwell or FLASH_ATTN on earlier GPUs.
        """
        return False

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        from vllm.platforms import current_platform

        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


@dataclass
class FlashInferMetadata:
    num_actual_tokens: int
    """Total number of tokens in the batch (excluding padding)."""

    slot_mapping: torch.Tensor
    """Tensor for writing K/V to the cache. Shape: [num_actual_tokens]"""

    q_data_type: torch.dtype

    num_decodes: int
    num_decode_tokens: int
    num_prefills: int
    num_prefill_tokens: int

    prefill: FIPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | None
    """
    Holds the metadata for the decode portion of the batch.
    Will be `None` if `num_decode_tokens == 0`.
    """

    # --- Special Case: Cascade Attention ---

    use_cascade: bool
    """
    If True, the entire batch is a cascade attention call, and the
    `prefill` and `decode` fields will both be None.
    """

    cascade_wrapper: MultiLevelCascadeAttentionWrapper | None


class FlashInferMetadataBuilder(AttentionMetadataBuilder[FlashInferMetadata]):
    reorder_batch_threshold: int = 1

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.attention_config = vllm_config.attention_config
        self._workspace_buffer = None
        self._prefill_wrapper: (
            BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper | None
        ) = None  # Wrapper for prefill/append
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        if vllm_is_batch_invariant():
            self.decode_fixed_split_size = 2048
            self.prefill_fixed_split_size = 4096
            self.disable_split_kv = True
        else:
            self.decode_fixed_split_size = -1
            self.prefill_fixed_split_size = -1
            self.disable_split_kv = False

        self.compilation_config = vllm_config.compilation_config
        max_num_pages_per_req = cdiv(
            self.model_config.max_model_len, self.kv_cache_spec.block_size
        )
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req
        speculative_config = vllm_config.speculative_config
        num_spec_tokens = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            else 0
        )
        self.enable_cuda_graph = (
            self.compilation_config.cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
        )
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchDecodeWithPagedKVCacheWrapper
            ] = {}
            self._decode_cudagraph_max_bs = (1 + num_spec_tokens) * max_num_reqs
            if self.compilation_config.max_cudagraph_capture_size is not None:
                self._decode_cudagraph_max_bs = min(
                    self._decode_cudagraph_max_bs,
                    self.compilation_config.max_cudagraph_capture_size,
                )
        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
            self.dcp_kv_cache_interleave_size = (
                vllm_config.parallel_config.dcp_kv_cache_interleave_size
            )
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
            self.dcp_kv_cache_interleave_size = 1
        self.use_dcp = self.dcp_world_size > 1

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            self.vllm_config.parallel_config
        )

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        self.page_size = self.kv_cache_spec.block_size

        self.cache_dtype = self.cache_config.cache_dtype
        if self.cache_dtype.startswith("fp8"):
            self.kv_cache_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.cache_dtype
            )
        else:
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # FlashInfer native always uses model dtype for query.
        self.q_data_type = self.model_config.dtype

        # FlashInfer native does not support speculative decoding as decode
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        self._cascade_wrapper = None  # Wrapper for cascade attention

        # Global hyperparameters shared by all attention layers
        # TODO: discard this for trtllm-gen backend
        self.global_hyperparameters = infer_global_hyperparameters(
            get_per_layer_parameters(vllm_config, layer_names, FlashInferImpl)
        )
        self.sm_scale = self.global_hyperparameters.sm_scale
        self.window_left = self.global_hyperparameters.window_left
        self.logits_soft_cap = self.global_hyperparameters.logits_soft_cap
        self.has_sinks = self.global_hyperparameters.has_sinks
        if self.has_sinks:
            raise NotImplementedError(
                "FlashInfer native backend does not support attention sinks. "
                "Please use TRTLLM_ATTN backend on Blackwell or FLASH_ATTN "
                "on earlier GPUs."
            )
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = (
            not envs.VLLM_USE_V2_MODEL_RUNNER and is_pin_memory_available()
        )
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

        if self.head_dim == 256 and current_platform.is_device_capability_family(100):
            # https://github.com/flashinfer-ai/flashinfer/issues/1993 reports that
            # head size 256 and block size 16 is not supported on blackwell.
            assert kv_cache_spec.block_size != 16, (
                "There is a bug in FlashInfer "
                "block_size 16 head size 256 support. Please avoid this combination by "
                "passing --block-size 32 or --block-size 64."
            )

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype = torch.int32
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=True,
        )

    @override  # type: ignore[misc]
    @classmethod
    def get_cudagraph_support(
        cls: type["FlashInferMetadataBuilder"],
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Get the cudagraph support level for FlashInfer attention.

        FlashInfer native only supports UNIFORM_SINGLE_TOKEN_DECODE.
        UNIFORM_BATCH requires TRTLLM attention (separate backend).
        """
        return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if vllm_is_batch_invariant():
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    def _get_prefill_wrapper(
        self,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                )
            else:
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), get_kv_cache_layout()
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_decode_wrapper(self, batch_size: int, use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_kv_indptr = self.paged_kv_indptr.gpu[: batch_size + 1]
                paged_kv_indices = self.paged_kv_indices.gpu
                paged_kv_last_page_len = self.paged_kv_last_page_len.gpu[:batch_size]
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
                2, self._get_workspace_buffer(), get_kv_cache_layout()
            )
        return self._cascade_wrapper

    def _compute_flashinfer_kv_metadata(
        self,
        num_blocks_np: np.ndarray,
        seq_lens_np: np.ndarray,
        block_table_tensor: torch.Tensor,
        num_reqs: int,
        page_size: int,
    ) -> torch.Tensor:
        """
        Compute paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len for FlashInfer
        attention.

        Results are stored in self.paged_kv_indptr,
        self.paged_kv_indices, self.paged_kv_last_page_len buffers.

        Returns paged_kv_indices, a GPU tensor with shape [num_actual_pages].
        """
        # write self.paged_kv_indptr_cpu inplace (0-index is always 0)
        np.cumsum(
            num_blocks_np,
            dtype=np.int32,
            out=self.paged_kv_indptr.np[1 : num_reqs + 1],
        )
        # NOTE(woosuk): Because self.paged_kv_indptr_cpu can be modified
        # after this line (e.g., for cuda graphs), we need to copy the data to
        # self.paged_kv_indptr_buffer to avoid race condition.
        self.paged_kv_indptr_cpu_buffer[: num_reqs + 1] = self.paged_kv_indptr.cpu[
            : num_reqs + 1
        ]
        paged_kv_indptr = self.paged_kv_indptr.gpu[: num_reqs + 1]
        paged_kv_indptr.copy_(
            self.paged_kv_indptr_cpu_buffer[: num_reqs + 1], non_blocking=True
        )

        # write self.paged_kv_indices inplace
        num_actual_pages = self.paged_kv_indptr.np[num_reqs]
        paged_kv_indices = self.paged_kv_indices.gpu[:num_actual_pages]
        _copy_page_indices_kernel[(num_reqs,)](
            paged_kv_indices,
            block_table_tensor,
            block_table_tensor.stride(0),
            paged_kv_indptr,
            BLOCK_SIZE=1024,
        )

        # write self.paged_kv_last_page_len_cpu inplace
        paged_kv_last_page_len_np = seq_lens_np % page_size
        self.paged_kv_last_page_len.np[:num_reqs] = np.where(
            (paged_kv_last_page_len_np == 0) & (seq_lens_np != 0),
            page_size,
            paged_kv_last_page_len_np,
        )
        return paged_kv_indices

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=True,
            )
        )

        page_size = self.page_size
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide dispatch mode (cascade vs regular)
        use_cascade = common_prefix_len > 0

        # Validate global hyperparameters for FlashInfer native backend
        if not self.global_hyperparameters.has_same_window_lefts:
            raise ValueError(
                "Window left is not the same for all layers. "
                "One potential fix is to set disable_sliding_window=True"
            )

        assert self.global_hyperparameters.has_same_all_params, (
            "FlashInfer backend currently only supports models in which "
            "all layers share the same values for the following "
            "hyperparameters: `window_left`, `logits_soft_cap`, "
            "`sm_scale`."
        )

        # Step 2: Initialize the output metadata
        # Leave prefill/decode/cascade_wrapper empty, to be populated
        # case by case depending on the batch contents and backend selection.
        attn_metadata = FlashInferMetadata(
            num_actual_tokens=num_actual_tokens,
            slot_mapping=common_attn_metadata.slot_mapping,
            q_data_type=self.q_data_type,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # FlashInfer native always needs seq_lens_cpu for KV metadata.
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        seq_lens_np = seq_lens_cpu.numpy()
        num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size

        # Adjust seq_lens_cpu for DCP
        if self.use_dcp:
            assert seq_lens_cpu is not None
            if num_prefills > 0:
                qo_indptr_prefill_cpu = (
                    qo_indptr_cpu[num_decodes:] - qo_indptr_cpu[num_decodes]
                )
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                seq_lens_cpu[num_decodes:] = (
                    seq_lens_cpu[num_decodes:] - query_lens_prefill_cpu
                )

            seq_lens_cpu = get_dcp_local_seq_lens(
                seq_lens_cpu,
                self.dcp_world_size,
                self.dcp_rank,
                self.dcp_kv_cache_interleave_size,
            )

        # Adjust num_block_np for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
            assert common_prefix_len % page_size == 0
            num_common_kv_blocks = common_prefix_len // page_size
            num_blocks_np -= num_common_kv_blocks

        # Compute paged_kv_indices for FlashInfer native
        paged_kv_indices = self._compute_flashinfer_kv_metadata(
            num_blocks_np,
            seq_lens_np,
            block_table_tensor,
            num_reqs,
            page_size,
        )

        # Early-out for cascade attention
        if use_cascade:
            # Grab the blocks of the shared prefix from the first request.
            num_common_kv_blocks = common_prefix_len // page_size

            # Create CPU versions directly for cascade (no GPU versions needed)
            shared_qo_indptr_cpu = torch.tensor(
                [0, num_actual_tokens], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indptr_cpu = torch.tensor(
                [0, num_common_kv_blocks], dtype=torch.int32, device="cpu"
            )
            shared_kv_page_indices_cpu = block_table_tensor[0, :num_common_kv_blocks]
            shared_kv_last_page_len_cpu = torch.tensor(
                [page_size], dtype=torch.int32, device="cpu"
            )

            # Remove the blocks of the shared prefix from all requests.
            block_table_tensor = block_table_tensor[:, num_common_kv_blocks:]
            num_blocks_np -= num_common_kv_blocks

            assert paged_kv_indices is not None
            paged_kv_indptr_cpu = self.paged_kv_indptr.cpu[: 1 + num_reqs]
            paged_kv_last_page_len_cpu = self.paged_kv_last_page_len.cpu[:num_reqs]

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
            return attn_metadata

        # Step 3: Handle prefill and decode pathways case by case
        ## PREFILL PATHWAY
        if num_prefills > 0:
            # Slices for shared prefill metadata
            prefill_start = num_decodes
            qo_indptr_prefill_cpu = (
                qo_indptr_cpu[prefill_start:] - qo_indptr_cpu[prefill_start]
            )
            assert qo_indptr_prefill_cpu.shape[0] == num_prefills + 1

            prefill_wrapper = self._get_prefill_wrapper()
            # Slicing CPU buffers that are only needed for FI native prefills
            paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                prefill_start:num_reqs
            ]
            assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
            paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                prefill_start : num_reqs + 1
            ]
            assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
            if self.use_dcp:
                assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                prefill_wrapper.plan(
                    qo_indptr_cpu=qo_indptr_prefill_cpu,
                    paged_kv_indptr_cpu=paged_kv_indptr_prefill_cpu,
                    paged_kv_indices=paged_kv_indices,
                    paged_kv_last_page_len_cpu=paged_kv_last_page_len_prefill_cpu,
                    page_size=self.page_size,
                    num_qo_heads=self.num_qo_heads,
                    dcp_world_size=self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_cache_dtype=self.kv_cache_dtype,
                    prefill_fixed_split_size=self.prefill_fixed_split_size,
                    disable_split_kv=self.disable_split_kv,
                )
            else:
                assert isinstance(
                    prefill_wrapper,
                    BatchPrefillWithPagedKVCacheWrapper,
                )
                prefill_wrapper.plan(
                    qo_indptr_prefill_cpu,
                    paged_kv_indptr_prefill_cpu,
                    paged_kv_indices,
                    paged_kv_last_page_len_prefill_cpu,
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
                    o_data_type=self.model_config.dtype,
                    fixed_split_size=self.prefill_fixed_split_size,
                    disable_split_kv=self.disable_split_kv,
                )
            attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)

        ## DECODE PATHWAY
        if num_decodes > 0:
            pure_decode = num_prefills == 0
            use_cudagraph = (
                self.enable_cuda_graph
                and pure_decode
                and num_decode_tokens <= self._decode_cudagraph_max_bs
            )
            num_input_tokens = num_decode_tokens

            decode_wrapper = self._get_decode_wrapper(num_input_tokens, use_cudagraph)
            # Use the persistent buffer with padding length,
            # instead of the same address but chunked version
            # in atten_metadata when using cudagraph.
            fast_plan_decode(
                decode_wrapper,
                self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                paged_kv_indices,
                self.paged_kv_last_page_len.cpu[:num_input_tokens],
                seq_lens_cpu[:num_input_tokens],
                self.num_qo_heads * self.dcp_world_size,
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
                o_data_type=self.model_config.dtype,
                fixed_split_size=self.decode_fixed_split_size,
                disable_split_kv=self.disable_split_kv,
            )
            attn_metadata.decode = FIDecode(wrapper=decode_wrapper)
        return attn_metadata

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        if self.kv_cache_spec.dtype != self.vllm_config.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        # TODO: Cascade attention doesn't work, disable it for now
        # return use_cascade_attention(*args, **kwargs)
        return False


class FlashInferImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
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
        self.window_left = (
            self.sliding_window[0] if self.sliding_window is not None else -1
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashInferImpl"
            )

        # FlashInfer native does not support sinks
        if sinks is not None:
            raise ValueError(
                "FlashInfer native backend does not support attention sinks. "
                "Use TRTLLM_ATTN backend on Blackwell or FLASH_ATTN on earlier GPUs."
            )

        # FlashInfer native does not support quantized query input
        self.supports_quant_query_input = False

    def fused_output_quant_supported(self, quant_key: QuantKey):
        # FlashInfer native does not support fused output quantization
        return False

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
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
            return output.fill_(0)

        # Ensure query dtype matches the expected dtype from attention metadata
        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        # FlashInfer native does not support fused output quantization
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "FlashInfer native backend does not support fused output "
                "quantization. Use TRTLLM_ATTN backend for this feature."
            )

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
        # to process the cache when the kv_cache_dtype is fp8
        if self.kv_sharing_target_layer_name is None and self.kv_cache_dtype.startswith(
            "fp8"
        ):
            torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                self.kv_cache_dtype
            )
            kv_cache = kv_cache.view(torch_dtype)

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]
        key = key[:num_actual_tokens]
        value = value[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        if attn_metadata.use_cascade:
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        # When using spec decoding, num_decodes can be < num_decode_tokens
        # because some decode requests may have more than one query token.
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        kv_cache_permute = kv_cache.permute(*stride_order)

        use_dcp = self.dcp_world_size > 1

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            assert isinstance(attn_metadata.prefill, FIPrefill)
            prefill_wrapper = attn_metadata.prefill.wrapper
            assert prefill_wrapper is not None
            if use_dcp:
                assert isinstance(prefill_wrapper, BatchDCPPrefillWrapper)
                assert prefill_wrapper._context._window_left == self.window_left
                assert prefill_wrapper._context._logits_soft_cap == (
                    self.logits_soft_cap or 0.0
                )
                assert prefill_wrapper._context._sm_scale == self.scale
                assert not prefill_wrapper._context._causal
                assert prefill_wrapper._new_tokens._window_left == self.window_left
                assert prefill_wrapper._new_tokens._logits_soft_cap == (
                    self.logits_soft_cap or 0.0
                )
                assert prefill_wrapper._new_tokens._sm_scale == self.scale
                assert prefill_wrapper._new_tokens._causal

                prefill_wrapper.run(
                    layer,
                    prefill_query,
                    kv_cache_permute,
                    key[num_decode_tokens:],
                    value[num_decode_tokens:],
                    out=output[num_decode_tokens:],
                )
            else:
                assert isinstance(prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper)
                assert prefill_wrapper._window_left == self.window_left
                assert prefill_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert prefill_wrapper._sm_scale == self.scale
                assert prefill_wrapper._causal
                prefill_wrapper.run(
                    prefill_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[num_decode_tokens:],
                )

        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            assert isinstance(attn_metadata.decode, FIDecode)
            decode_wrapper = attn_metadata.decode.wrapper
            assert decode_wrapper is not None
            assert decode_wrapper._window_left == self.window_left
            assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
            assert decode_wrapper._sm_scale == self.scale

            if use_dcp:
                decode_query = get_dcp_group().all_gather(
                    decode_query.contiguous(), dim=-2
                )
                output_tmp = torch.empty_like(decode_query)
                lse = torch.empty(
                    (decode_query.size(0), decode_query.size(1)),
                    dtype=torch.float32,
                    device=decode_query.device,
                )
                decode_wrapper.run(
                    decode_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output_tmp,
                    lse=lse,
                    return_lse=True,
                )
                output[:num_decode_tokens] = cp_lse_ag_out_rs(
                    output_tmp,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=False,
                )
            else:
                decode_wrapper.run(
                    decode_query,
                    kv_cache_permute,
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                    out=output[:num_decode_tokens],
                )
        return output_padded

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
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
                slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )


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
    logits_soft_cap: float | None = None,
    q_data_type: str | torch.dtype | None = "float16",
    kv_data_type: str | torch.dtype | None = None,
    o_data_type: str | torch.dtype | None = None,
    data_type: str | torch.dtype | None = None,
    sm_scale: float | None = None,
    rope_scale: float | None = None,
    rope_theta: float | None = None,
    non_blocking: bool = True,
    fixed_split_size: int = -1,
    disable_split_kv: bool = False,
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
    if not self.is_cuda_graph_enabled or getattr(self, "vllm_first_call", True):
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
            o_data_type,
            data_type,
            sm_scale,
            rope_scale,
            rope_theta,
            non_blocking,
            None,  # block_tables
            None,  # seq_lens
            fixed_split_size,
            disable_split_kv,
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
    q_data_type = (
        getattr(torch, q_data_type) if isinstance(q_data_type, str) else q_data_type
    )
    kv_data_type = (
        getattr(torch, kv_data_type) if isinstance(kv_data_type, str) else kv_data_type
    )

    if batch_size != self._fixed_batch_size:
        raise ValueError(
            "The batch size should be fixed in cudagraph mode, the runtime "
            "batch size {} mismatches the batch size set during "
            "initialization {}".format(batch_size, self._fixed_batch_size)
        )
    if len(indices) > len(self._paged_kv_indices_buf):
        raise ValueError(
            "The size of indices should be less than or equal to the allocated buffer"
        )

    # host-to-device copy for the indptr buffer
    self._paged_kv_indptr_buf.copy_(indptr_cpu, non_blocking=True)
    # host-to-device copy for the last_page_len buffer
    self._paged_kv_last_page_len_buf.copy_(last_page_len_cpu, non_blocking=True)

    qo_indptr_host = _get_range_buf(batch_size + 1, "cpu")

    try:
        # Make sure we pass exactly 19 arguments for tensor core version
        args = [
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
            window_left,
        ]
        if self._backend == "fa2":
            args.append(fixed_split_size)
            args.append(disable_split_kv)
            args.append(0)  # num_colocated_ctas
        self._plan_info = self._cached_module.plan(
            *args,
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
        tl.store(
            page_indices + start_idx + i + offset,
            block_ids,
            mask=i + offset < num_blocks,
        )
