# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashInfer."""

from dataclasses import dataclass
from functools import partial
from typing import ClassVar

import numpy as np
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    MultiLevelCascadeAttentionWrapper,
)
from flashinfer.decode import fast_decode_plan, trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache
from flashinfer.utils import FP4Tensor
from typing_extensions import override

from vllm import envs
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_current_vllm_config_or_none,
)
from vllm.config.cache import CacheDType
from vllm.distributed.parallel_state import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    can_use_trtllm_attention,
    use_trtllm_attention,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import (
    PIN_MEMORY,
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
    is_strictly_contiguous,
    nvfp4_kv_cache_full_dim,
    nvfp4_kv_cache_split_views,
)
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
    get_num_attention_heads_from_layers,
    get_per_layer_parameters,
    infer_global_hyperparameters,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.utils import CpuGpuBuffer

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8

logger = init_logger(__name__)


def _vllm_nvfp4_kv_vosplit_requested() -> bool:
    """VLLM_NVFP4_KV_VOSPLIT opts head_size > 256 NVFP4 layers into the FA2
    two-pass VO split (Gemma 4 global D=512 full-attention layers).

    Default-on (see vllm/envs.py); only an explicit "0" disables it. The
    model-config routing (vllm/model_executor/models/config.py) gates the
    Gemma 4 -> FLASHINFER decision on the same flag, so the backend must
    honor it here too."""
    return bool(envs.VLLM_NVFP4_KV_VOSPLIT)


def _vo_split_factor(head_size: int, is_fa2_nvfp4: bool) -> int:
    """Number of VO passes for the FlashInfer FA2 path.

    The FA2 nvfp4 kernel trait guard rejects HEAD_DIM_VO > 256 (the
    per-thread output-accumulator fragments do not fit the register
    budget), but HEAD_DIM_QK=512 is fine, and attention decomposes
    EXACTLY along the VO dimension: S = Q @ K^T and the softmax are
    identical per pass, and O = [P @ V_left | P @ V_right] concatenates
    with no LSE merge. So a Gemma 4 full-attention head (Q=K=V=512 wide;
    the cache stores V at 512) runs as ``ceil(head_size/256)`` passes of
    ``(head_dim_qk=512, head_dim_vo=256)``, each over a head-dim slice of
    the 512-wide V cache (and, for NVFP4, of its per-16-element scale
    factors).

    The split is dtype-independent (the guard counts only accumulator
    fragments). For NVFP4 it additionally requires the contiguous
    ``[all-data | all-SF]`` cache layout (``contiguous_sf_layout=True`` in
    ``nvfp4_kv_cache_split_views``) so each chunk's data and scale factors
    are contiguous and slice cleanly along the head dim; the trtllm-gen
    per-page swizzle does not commute with head-dim slicing.
    """
    if head_size <= 256:
        return 1
    if is_fa2_nvfp4 and not _vllm_nvfp4_kv_vosplit_requested():
        raise ValueError(
            f"NVFP4 KV with head_size={head_size} on the SM12x FA2 path "
            "needs the two-pass VO split (the FA2 kernel caps HEAD_DIM_VO "
            "at 256). Set VLLM_NVFP4_KV_VOSPLIT=1 to enable it, or keep "
            "these layers on a different KV dtype."
        )
    split = -(-head_size // 256)  # ceil(head_size / 256)
    if head_size % split != 0 or (is_fa2_nvfp4 and (head_size // split) % 16 != 0):
        raise ValueError(
            "The VO split needs head_size divisible into <=256-wide chunks"
            f"{' of whole 16-element scale blocks' if is_fa2_nvfp4 else ''}; "
            f"got head_size={head_size}."
        )
    return split


trtllm_gen_workspace_buffer = None


def _get_trtllm_gen_workspace_buffer():
    global trtllm_gen_workspace_buffer
    if trtllm_gen_workspace_buffer is None:
        trtllm_gen_workspace_buffer = torch.zeros(
            envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda"
        )
    return trtllm_gen_workspace_buffer


@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    src_stride_page,
    src_stride_kv,
    src_stride_head,
    DST_K_CACHE_STRIDE: tl.constexpr,
    DST_KV_CACHE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty

    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    mock_page_idx = batch_idx * block_table_stride + mock_block_table_idx + 1
    head_offsets = tl.arange(0, HEAD_STRIDE)

    for h in range(NUM_KV_HEADS):
        h_off = tl.cast(h, tl.int64)

        # Read K from source (supports non-contiguous page/kv/head strides)
        src_k = orig_page_num * src_stride_page + h_off * src_stride_head + head_offsets
        fp8_k = tl.load(kv_cache_ptr + src_k)
        dequant_k = (fp8_k.to(tl.float32) * k_scale_val).to(dequant_dtype)

        # Write K to contiguous mock cache
        dst_k = mock_page_idx * DST_KV_CACHE_STRIDE + h * HEAD_STRIDE + head_offsets
        tl.store(mock_kv_cache_ptr + dst_k, dequant_k)

        # Read V from source (offset by src_stride_kv for the V half)
        src_v = (
            orig_page_num * src_stride_page
            + src_stride_kv
            + h_off * src_stride_head
            + head_offsets
        )
        fp8_v = tl.load(kv_cache_ptr + src_v)
        dequant_v = (fp8_v.to(tl.float32) * v_scale_val).to(dequant_dtype)

        # Write V to contiguous mock cache
        dst_v = (
            mock_page_idx * DST_KV_CACHE_STRIDE
            + DST_K_CACHE_STRIDE
            + h * HEAD_STRIDE
            + head_offsets
        )
        tl.store(mock_kv_cache_ptr + dst_v, dequant_v)


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

    num_kv_heads, block_size, head_size = s[2], s[3], s[4]
    head_stride = block_size * head_size
    k_cache_stride = num_kv_heads * head_stride
    kv_cache_stride = k_cache_stride * s[1]

    strides = kv_cache.stride()
    assert strides[3] == head_size and strides[4] == 1, (
        "For kv cache layouts, (block_size, head_size) "
        f"dimensions must be contiguous, got strides {strides}"
    )

    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    # mock kv cache contains just the pages needed by this prefill
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device=kv_cache.device)
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
        strides[0],
        strides[1],
        strides[2],
        k_cache_stride,
        kv_cache_stride,
        head_stride,
        num_kv_heads,
    )
    return mock_kv_cache, mock_block_table


class BatchDCPPrefillWrapper:
    def __init__(
        self,
        workspace_buffer: torch.Tensor | None = None,
        dcp_a2a: bool = False,
    ):
        if dcp_a2a:
            self._dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self._dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)
        self._context = BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, get_kv_cache_layout()
        )
        self._new_tokens = BatchPrefillWithRaggedKVCacheWrapper(workspace_buffer)

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
            qo_indptr=qo_indptr_cpu,
            paged_kv_indptr=paged_kv_indptr_cpu,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len_cpu,
            num_qo_heads=num_qo_heads * dcp_world_size,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
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
        output_context, lse_context = self._dcp_combine(
            output_context_tmp,
            lse_context_tmp,
            get_dcp_group(),
            return_lse=True,
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
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "nvfp4",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Page sizes >= 128 only run on the trtllm-gen dynamic kernel (GQA/MQA
        # on Blackwell); advertise them only when usable so selection never
        # picks a large kernel block we cannot serve.
        use_large_pages = False
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and vllm_config.model_config is not None:
            pc = vllm_config.parallel_config
            mc = vllm_config.model_config
            num_qo_heads = mc.get_num_attention_heads(pc)
            num_kv_heads = mc.get_num_kv_heads(pc)
            use_large_pages = (
                num_kv_heads > 0
                and num_qo_heads // num_kv_heads > 1
                and can_use_trtllm_attention(num_qo_heads, num_kv_heads)
            )
        if not use_large_pages:
            return [16, 32, 64]
        return [16, 32, 64, 128, 256, 512, 1024]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER"

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

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
        if cache_dtype_str == "nvfp4":
            # Packed layout: fp4 data + fp8 block scales in last dim
            last_dim = nvfp4_kv_cache_full_dim(head_size)
            return (num_blocks, 2, block_size, num_kv_heads, last_dim)
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
    def get_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        elif kv_cache_dtype == "nvfp4":
            return torch.uint8
        else:
            raise ValueError(f"Unrecognized dtype: {kv_cache_dtype}")

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        return [64, 128, 256, 512]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # FlashInfer supports SM75+, but is currently broken on SM75 (Turing):
        # https://github.com/flashinfer-ai/flashinfer/issues/3620 (fix:
        # https://github.com/flashinfer-ai/flashinfer/pull/3621). Temporarily
        # raise the floor to SM80 so it is not auto-selected on SM75 until
        # that fix lands; revert to DeviceCapability(7, 5) once it does.
        return capability >= DeviceCapability(8, 0) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    def supports_sink(cls) -> bool:
        """FlashInfer supports sinks when TRTLLM attention is available (SM100)."""
        from vllm.utils.flashinfer import (
            force_use_trtllm_attention,
            supports_trtllm_attention,
        )

        # Respect explicit disable flag (e.g.,
        # --attention-config.use_trtllm_attention=0)
        if force_use_trtllm_attention() is False:
            return False

        # Check if TRTLLM is supported on this platform
        return supports_trtllm_attention()

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """mm-prefix LMs (Gemma 3 / Gemma 4 multimodal: image-token spans
        attend bidirectionally) are served via FA2 packed custom masks on
        the FI-native prefill path. Decode is untouched: spans live in the
        prompt, so decode queries are strictly causal. Knob-gated and
        default-on; set VLLM_FLASHINFER_MM_PREFIX=0 to route mm-prefix
        models away from FlashInfer instead (e.g. for debugging)."""
        return envs.VLLM_FLASHINFER_MM_PREFIX

    @classmethod
    def get_required_kv_cache_layout(cls) -> KVCacheLayoutType | None:
        capability = current_platform.get_device_capability()
        if capability is not None and capability.major == 10:
            return "HND"
        return None

    forward_includes_kv_cache_update: bool = False


@dataclass
class FIPrefillGroup:
    """One partition of a prefill batch whose requests do not all share the
    same attention semantics. The group key is the per-request tuple
    ``(is_mm, causal)``, subsuming two structurally parallel mechanisms:

    * mm-prefix (Gemma 3 / Gemma 4 multimodal): requests whose image-token
      span intersects the query window need span-level bidirectional
      masking; plain requests stay causal.
    * per-request causal (DiffusionGemma): encoder/commit requests are
      causal, denoise requests are bidirectional, mixed within a batch.
    * composed mm x causal (multimodal DiffusionGemma): requests group by
      BOTH keys; each group's wrapper carries the right causal flag AND
      the right packed mask.

    The attention semantics are baked into ``wrapper`` at plan() time, so
    the dataclass only needs the wrapper plus this group's gather index."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper
    """Planned for exactly this group's requests. Plain-causal groups plan
    causal=True with the layer-group window; non-causal groups plan
    causal=False; mm groups plan a packed custom mask ((base AND
    sliding-window) OR span-bidirectional) with the chosen causal base
    folded into the mask, window_left=-1."""

    token_indices: torch.Tensor
    """Rows of the prefill query/output owned by this group: GPU int64,
    the concatenation of per-request token ranges from query_start_loc.
    Needed because the groups may interleave within the batch."""

    num_tokens: int


@dataclass
class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper

    prefill_groups: list[FIPrefillGroup] | None = None
    """Per-semantics wrapper grouping when the prefill batch mixes
    attention semantics -- image-token spans intersect the query window of
    at least one request (mm-prefix) and/or ``CommonAttentionMetadata.causal``
    is a per-request tensor (DiffusionGemma). None on the scalar-causal
    no-mm fast path (=> byte-identical legacy single-wrapper path)."""


@dataclass
class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper


@dataclass
class TRTLLMPrefill:
    """Metadata for the TRTLLM prefill pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to prefill requests.
    Shape: [num_prefills, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to prefill requests.
    Shape: [num_prefills]
    """

    cum_seq_lens_q: torch.Tensor
    cum_seq_lens_kv: torch.Tensor

    max_q_len: int
    """
    The maximum query length *among prefill requests*.
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


@dataclass
class TRTLLMDecode:
    """Metadata for the TRTLLM decode pathway."""

    block_tables: torch.Tensor
    """
    The slice of the block table tensor corresponding *only* to decode requests.
    Shape: [num_decodes, max_num_blocks_per_seq]
    """

    seq_lens: torch.Tensor
    """
    The slice of the sequence lengths tensor corresponding *only* to decode requests.
    Shape: [num_decodes]
    """

    max_seq_len: int
    """The maximum sequence length for KV Cache."""


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
    causal: bool

    prefill: FIPrefill | TRTLLMPrefill | None
    """
    Holds the metadata for the prefill portion of the batch.
    Will be `None` if `num_prefill_tokens == 0`.
    """

    decode: FIDecode | TRTLLMDecode | None
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
        self._noncausal_prefill_wrapper: BatchPrefillWithPagedKVCacheWrapper | None = (
            None  # Wrapper for non-causal prefill (DFlash)
        )
        # Secondary persistent prefill wrappers for the grouped-prefill path
        # (FIPrefillGroup). Each group key that is NOT plain-causal (which
        # reuses self._prefill_wrapper) gets its own wrapper, since every
        # group plans once and all run in the same step. Keyed by
        # (is_mm, causal); every wrapper is built nvfp4-aware (so the FA2/
        # trtllm-gen backend + VO split apply identically), unlike the
        # backend="auto" DFlash _noncausal_prefill_wrapper which rejects nvfp4.
        #   (True,  True ) mm spans, causal base   -> packed custom mask
        #   (True,  False) mm spans, non-causal    -> packed custom mask
        #   (False, False) plain non-causal (DiffusionGemma denoise)
        # The (False, True) plain-causal key uses self._prefill_wrapper.
        self._grouped_prefill_wrappers: dict[
            tuple[bool, bool], BatchPrefillWithPagedKVCacheWrapper
        ] = {}
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        if envs.VLLM_BATCH_INVARIANT:
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
        self.dcp_a2a = (
            self.use_dcp and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Compatible with models with non-uniform per-layer head counts.
        self.num_qo_heads = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or self.model_config.get_num_attention_heads(self.vllm_config.parallel_config)

        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        # Gemma 4 full-attention is SYMMETRIC: Q=K=V=512-wide per head (the
        # KV cache stores V at 512). There is no native 256-wide V. The FA2
        # nvfp4 kernel caps HEAD_DIM_VO at 256, so a 512-wide V/O is run as
        # ceil(head_size/256) two-pass VO-split chunks: each pass uses
        # head_dim_qk=full head_size, head_dim_vo=head_size//vo_split, over a
        # head-dim slice of the 512-wide V cache; the per-pass outputs
        # concatenate exactly (identical S/softmax, no LSE merge).
        # vo_split == 1 for head_size <= 256 (ordinary single-pass).
        self.page_size = self.kv_cache_spec.block_size

        if self.kv_cache_spec.kv_quant_mode != KVQuantMode.NONE:
            self.cache_dtype = self.cache_config.cache_dtype
            # Cannot use self.kv_cache_spec.dtype here because kv_cache_spec
            # storage dtype may not be the same as the op dtype (uint8 vs fp8_e4m3)
            self.is_kvcache_nvfp4 = self.cache_dtype == "nvfp4"
            self.use_fa2_nvfp4_kv = False
            if self.is_kvcache_nvfp4:
                if current_platform.is_device_capability_family(120):
                    # Consumer Blackwell (sm120/sm121): no trtllm-gen FP4 FMHA,
                    # so route NVFP4 KV through FlashInfer's FA2 paged reader.
                    # The cache stores packed uint8 fp4 data; scale factors are a
                    # separate contiguous tensor (see contiguous_sf_layout).
                    self.use_fa2_nvfp4_kv = True
                    self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                        "nvfp4"
                    )
                elif current_platform.is_device_capability_family(100):
                    # sm100f trtllm-gen: kv_cache_dtype stays the string "nvfp4"
                    # which is passed to FlashInferImpl.
                    self.kv_cache_dtype = self.cache_dtype
                else:
                    raise ValueError(
                        "--kv-cache-dtype nvfp4 requires sm100f (datacenter "
                        "Blackwell) or sm120/sm121 (consumer Blackwell)"
                    )
            else:
                self.kv_cache_dtype = FlashInferBackend.get_dtype_for_flashinfer(
                    self.cache_dtype
                )
        else:
            self.cache_dtype = "auto"
            self.is_kvcache_nvfp4 = False
            self.use_fa2_nvfp4_kv = False
            assert self.kv_cache_spec.dtype == self.model_config.dtype
            self.kv_cache_dtype = self.kv_cache_spec.dtype

        # Use model dtype as q dtype when TRTLLM attn is not supported, or
        # --attention-config.disable_flashinfer_q_quantization is set to 1. Otherwise,
        # try to use fp8 q if kv cache is fp8, and will fall back to model dtype
        # if TRTLLM attention kernel is not used when building attn metadata
        can_use_trtllm = can_use_trtllm_attention(self.num_qo_heads, self.num_kv_heads)

        # Page sizes >= 128 require the trtllm-gen GQA/MQA path (guaranteed by
        # get_supported_kernel_block_sizes).
        assert self.page_size <= 64 or (
            can_use_trtllm and self.num_qo_heads // self.num_kv_heads > 1
        ), f"Unexpected FlashInfer page size {self.page_size} without trtllm-gen GQA"

        if (
            can_use_trtllm
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        ):
            if self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv:
                # trtllm-gen NVFP4 KV uses FP8 queries; the FA2 paged
                # path (sm12x) uses model-dtype queries.
                self.q_data_type = FlashInferBackend.get_dtype_for_flashinfer(
                    "fp8_e4m3"
                )
            elif self.is_kvcache_nvfp4:
                self.q_data_type = self.model_config.dtype
            else:
                self.q_data_type = self.kv_cache_dtype
        else:
            self.q_data_type = self.model_config.dtype

        # Prefer TRTLLM attention for decoding in all cases.
        # This allows us to use AttentionCGSupport.UNIFORM_BATCH mode.
        self.use_trtllm_decode_attention = can_use_trtllm
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=can_use_trtllm)

        # Two-pass VO split for head_size > 256 (Gemma 4 full-attention,
        # head_dim_qk=512). vo_split == 1 leaves all existing paths
        # untouched.
        self.vo_split = _vo_split_factor(self.head_dim, self.use_fa2_nvfp4_kv)
        if self.vo_split > 1:
            # BatchDecodeWithPagedKVCacheWrapper.plan() has no head_dim_vo,
            # so route every request through the VO-split-planned prefill
            # wrapper: threshold 0 classifies nothing as decode, and a
            # causal qo_len==1 prefill computes exactly what decode would.
            self.reorder_batch_threshold = 0
            logger.info_once(
                "FA2 VO split (%s KV): head_size %d runs as %d passes of "
                "head_dim_vo=%d; decode requests use the prefill wrapper.",
                self.cache_dtype,
                self.head_dim,
                self.vo_split,
                self.head_dim // self.vo_split,
            )

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
        if self.has_sinks and not can_use_trtllm:
            raise NotImplementedError(
                "FlashInfer backend currently does not support attention "
                "sinks, please use trtllm on blackwell or flash attention on "
                "earlier GPUs."
            )

        # mm-prefix (PrefixLM) span-level bidirectional masking for this
        # layer group. Gemma4 with use_bidirectional_attention='vision'
        # applies bidirectional image spans ONLY to sliding_attention
        # layers (full-attention layers stay plain causal: the static
        # equivalent of gemma4_mm._clear_mm_prefix_for_full_attn_layers,
        # decided here at build time because FlashInfer bakes masks into
        # wrapper plans). Gemma3 applies the spans to all layers.
        self.mm_prefix_enabled = (
            envs.VLLM_FLASHINFER_MM_PREFIX
            and self.model_config is not None
            and self.model_config.is_mm_prefix_lm
        )
        if self.mm_prefix_enabled:
            bidi_mode = getattr(
                self.model_config.hf_text_config,
                "use_bidirectional_attention",
                None,
            )
            if bidi_mode == "vision" and self.window_left < 0:
                self.mm_prefix_enabled = False
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer mm-prefix custom masks are not wired for "
                    "DCP; unset VLLM_FLASHINFER_MM_PREFIX or disable DCP."
                )
        if self.mm_prefix_enabled:
            logger.info_once(
                "FlashInfer mm-prefix: image-token spans of multimodal "
                "requests run with FA2 packed custom masks on a second "
                "prefill wrapper (layer group window_left=%d).",
                self.window_left,
            )
        # Preparing persistent buffers
        # Since we do not have explicit synchronization in ModelRunnerV2, we do not pin
        # reused CPU buffers to avoid a race condition between step N async copies to
        # GPU and step N+1 buffer updates.
        self.pin_memory = not vllm_config.use_v2_model_runner and PIN_MEMORY
        self.paged_kv_indptr = self._make_buffer(max_num_reqs + 1)
        self.paged_kv_indptr_cpu_buffer = torch.zeros_like(
            self.paged_kv_indptr.cpu, pin_memory=self.pin_memory
        )  # Extra buffer for mutable paged_kv_indptr.cpu in cuda graph mode
        self.paged_kv_indices = self._make_buffer(max_num_pages)
        self.paged_kv_last_page_len = self._make_buffer(max_num_reqs)

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

        This depends on whether we can use TRTLLM attention for decodes, since we can
        only do UNIFORM_SINGLE_TOKEN_DECODE if it is unavailable.
        To check this, we must call can_use_trtllm_attention with the number of KV
        heads from the kv_cache_spec. We check all available KV cache specs and
        only return UNIFORM_BATCH if all of them support TRTLLM attention.
        """
        # For UniformTypeKVCacheSpecs, check all contained specs
        kv_specs = (
            kv_cache_spec.kv_cache_specs.values()
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs)
            else [kv_cache_spec]
        )
        num_qo_heads = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        has_trtllm_support: bool = len(kv_specs) > 0
        for spec in kv_specs:
            if not isinstance(spec, AttentionSpec):
                # FlashInfer only applies to attention, so we don't consider other types
                # of KV spec (e.g. Mamba) here. This is mostly for type checking.
                continue
            if not can_use_trtllm_attention(
                num_qo_heads=num_qo_heads,
                num_kv_heads=spec.num_kv_heads,
            ):
                has_trtllm_support = False
                break

        if has_trtllm_support:
            return AttentionCGSupport.UNIFORM_BATCH
        else:
            return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            buffer_size = envs.VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE
            if envs.VLLM_BATCH_INVARIANT:
                buffer_size = FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT
            self._workspace_buffer = torch.zeros(
                buffer_size, dtype=torch.uint8, device=self.device
            )
        return self._workspace_buffer

    def set_workspace_buffer(self, workspace_buffer: torch.Tensor):
        self._workspace_buffer = workspace_buffer

    def _get_prefill_wrapper(
        self,
        causal: bool = True,
    ) -> BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper:
        if not causal:
            if self.use_dcp:
                raise NotImplementedError(
                    "FlashInfer non-causal prefill is not supported with DCP yet."
                )
            if self.is_kvcache_nvfp4:
                raise NotImplementedError(
                    "FlashInfer non-causal attention is not supported with "
                    "NVFP4 KV cache."
                )
            if self._noncausal_prefill_wrapper is None:
                self._noncausal_prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend="auto",
                )
            return self._noncausal_prefill_wrapper

        if self._prefill_wrapper is None:
            if self.use_dcp:
                self._prefill_wrapper = BatchDCPPrefillWrapper(
                    workspace_buffer=self._get_workspace_buffer(),
                    dcp_a2a=self.dcp_a2a,
                )
            else:
                # NVFP4 KV: FlashInfer FA2 paged reader on consumer Blackwell
                # (sm120/sm121); trtllm-gen on sm100f.
                if self.use_fa2_nvfp4_kv:
                    backend = "fa2"
                elif self.is_kvcache_nvfp4:
                    backend = "trtllm-gen"
                else:
                    backend = "auto"
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend=backend,
                )
        assert self._prefill_wrapper is not None
        return self._prefill_wrapper

    def _get_group_prefill_wrapper(
        self, is_mm: bool, causal: bool
    ) -> BatchPrefillWithPagedKVCacheWrapper:
        # Persistent prefill wrapper for one (is_mm, causal) group key of the
        # grouped prefill path (FIPrefillGroup). The plain-causal key reuses
        # the primary self._prefill_wrapper (so the legacy single-wrapper plan
        # is untouched when only one group exists); every other key -- including
        # plain non-causal -- gets its own lazily-built wrapper from the SAME
        # nvfp4-aware construction (the VO split + NVFP4 jit module are applied
        # at plan()/run() time). Only reachable on the grouped path (no DCP).
        if not is_mm and causal:
            wrapper = self._get_prefill_wrapper()
            assert isinstance(wrapper, BatchPrefillWithPagedKVCacheWrapper)
            return wrapper
        key = (is_mm, causal)
        wrapper = self._grouped_prefill_wrappers.get(key)
        if wrapper is None:
            if self.use_fa2_nvfp4_kv:
                backend = "fa2"
            elif self.is_kvcache_nvfp4:
                backend = "trtllm-gen"
            else:
                backend = "auto"
            wrapper = BatchPrefillWithPagedKVCacheWrapper(
                self._get_workspace_buffer(),
                get_kv_cache_layout(),
                backend=backend,
            )
            self._grouped_prefill_wrappers[key] = wrapper
        return wrapper

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
            # NVFP4 KV: FlashInfer FA2 paged reader on consumer Blackwell
            # (sm120/sm121); trtllm-gen on sm100f.
            if self.use_fa2_nvfp4_kv:
                backend = "fa2"
            elif self.is_kvcache_nvfp4:
                backend = "trtllm-gen"
            else:
                backend = "auto"
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
                backend=backend,
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
        self.paged_kv_last_page_len.gpu[:num_reqs].copy_(
            self.paged_kv_last_page_len.cpu[:num_reqs], non_blocking=True
        )
        return paged_kv_indices

    def _mm_prefix_prefill_spans(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        num_decodes: int,
        num_prefills: int,
    ) -> list[list[tuple[int, int]]] | None:
        """Per-prefill-request image spans that intersect the query window.

        Spans are document positions, inclusive [start, end] (the
        mm_req_doc_ranges convention shared with the Triton/FlexAttention
        mm-prefix paths; valid iff start < end). A span fully inside the
        computed context needs nothing: K/V projections are mask-independent
        and the in-window queries are text, i.e. causal. vLLM forces
        --disable-chunked-mm-input for mm-prefix models, so spans do not
        straddle the window boundary in practice; partial overlaps are
        still handled correctly by the absolute-position mask.

        Returns None when no prefill request has an intersecting span
        (the scalar-causal fast path stays byte-identical).
        """
        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if not mm_ranges:
            return None
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu
        span_lists: list[list[tuple[int, int]]] = []
        any_spans = False
        for j in range(num_prefills):
            req_idx = num_decodes + j
            kv_len = int(seq_lens_cpu[req_idx])
            qo_len = int(qo_indptr_cpu[req_idx + 1] - qo_indptr_cpu[req_idx])
            context_len = kv_len - qo_len
            spans = [
                (int(s), int(e))
                for s, e in mm_ranges.get(req_idx, [])
                if s < e and e >= context_len and s < kv_len
            ]
            any_spans = any_spans or bool(spans)
            span_lists.append(spans)
        return span_lists if any_spans else None

    def _build_mm_prefix_custom_mask(
        self,
        qo_lens: list[int],
        kv_lens: list[int],
        span_lists: list[list[tuple[int, int]]],
        causal_base: bool = True,
    ) -> torch.Tensor:
        """Boolean (qo_len x kv_len)-per-request mask, flattened row-major
        and concatenated in group order; FlashInfer's plan() bit-packs it
        per request (segment_packbits). Composition matches the Triton /
        FlexAttention mm-prefix contract:
        (base AND sliding-window) OR (q in span AND kv in span),
        with query rows end-aligned to the KV sequence. The mm wrapper is
        planned with window_left=-1 because the mask already carries the
        sliding window; spans must OVERRIDE it (FlashInfer's in-kernel SW
        would AND it instead).

        ``base`` is the causal lower-triangle when ``causal_base`` is True
        (the autoregressive mm ladder) and the all-attend matrix when False
        (composed mm x non-causal: a DiffusionGemma denoise request that
        also carries image spans -- text region bidirectional, spans stay
        bidirectional). The sliding-window AND still applies to the base in
        both cases; spans OR over it unchanged."""
        masks = []
        for qo_len, kv_len, spans in zip(qo_lens, kv_lens, span_lists):
            q_abs = torch.arange(
                kv_len - qo_len, kv_len, device=self.device, dtype=torch.int32
            )
            k_abs = torch.arange(kv_len, device=self.device, dtype=torch.int32)
            if causal_base:
                mask = k_abs[None, :] <= q_abs[:, None]
                if self.window_left >= 0:
                    mask &= (q_abs[:, None] - k_abs[None, :]) <= self.window_left
            else:
                mask = torch.ones(
                    (qo_len, kv_len), device=self.device, dtype=torch.bool
                )
                if self.window_left >= 0:
                    # Non-causal base: symmetric window around the query.
                    mask &= (q_abs[:, None] - k_abs[None, :]).abs() <= self.window_left
            for start, end in spans:
                q_in = (q_abs >= start) & (q_abs <= end)
                k_in = (k_abs >= start) & (k_abs <= end)
                mask |= q_in[:, None] & k_in[None, :]
            masks.append(mask.reshape(-1))
        return torch.cat(masks)

    def _plan_prefill_groups(
        self,
        prefill_mm_spans: list[list[tuple[int, int]]] | None,
        causal_prefill_cpu: torch.Tensor | None,
        qo_indptr_prefill_cpu: torch.Tensor,
        paged_kv_indptr_prefill_cpu: torch.Tensor,
        paged_kv_last_page_len_prefill_cpu: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        seq_lens_prefill_cpu: torch.Tensor,
        o_dtype: torch.dtype,
    ) -> list[FIPrefillGroup]:
        """Plan one prefill wrapper per distinct attention-semantics
        partition of the batch, keyed by the per-request tuple
        ``(is_mm, causal)``:

        * ``is_mm`` is True when the request has an image-token span that
          intersects its query window (mm-prefix); the mm group plans a
          packed custom mask. Source: ``prefill_mm_spans`` (None => all
          is_mm=False).
        * ``causal`` is the per-request causal flag (DiffusionGemma:
          encoder/commit causal, denoise bidirectional). Source:
          ``causal_prefill_cpu`` (None => scalar-causal batch, all causal).

        Subsumes mm-only (is_mm varies), causal-only (causal varies), and
        composed mm x causal. Each request's tokens and KV pages are
        contiguous ranges of the batch-level arrays, so a group is fully
        described by per-request indptr deltas plus gathered index subranges.
        plan(custom_mask=...) requires the FlashInfer-side mask_indptr device
        fix because the mask lives on GPU while the indptr arrays stay on CPU.
        """
        num_prefills = int(qo_indptr_prefill_cpu.numel()) - 1
        is_mm = [
            bool(prefill_mm_spans[i]) if prefill_mm_spans is not None else False
            for i in range(num_prefills)
        ]
        if causal_prefill_cpu is not None:
            is_causal = [bool(causal_prefill_cpu[i]) for i in range(num_prefills)]
        else:
            is_causal = [True] * num_prefills
        keys = [(is_mm[i], is_causal[i]) for i in range(num_prefills)]

        qo_lens_cpu = qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
        # paged_kv_indptr is NOT rebased to 0 (its offsets index the full
        # paged_kv_indices array), so deltas are taken before regrouping.
        kv_page_counts_cpu = (
            paged_kv_indptr_prefill_cpu[1:] - paged_kv_indptr_prefill_cpu[:-1]
        )
        groups: list[FIPrefillGroup] = []
        # Deterministic key order: plain-causal first (so a degenerate
        # single-group batch reuses the primary wrapper exactly as the
        # mm-only path did), then the rest.
        for group_mm, group_causal in (
            (False, True),
            (False, False),
            (True, True),
            (True, False),
        ):
            req_indices = torch.tensor(
                [
                    i
                    for i in range(num_prefills)
                    if keys[i] == (group_mm, group_causal)
                ],
                dtype=torch.int64,
            )
            if req_indices.numel() == 0:
                continue
            group_qo_indptr = torch.zeros(req_indices.numel() + 1, dtype=torch.int32)
            torch.cumsum(qo_lens_cpu[req_indices], dim=0, out=group_qo_indptr[1:])
            group_kv_indptr = torch.zeros(req_indices.numel() + 1, dtype=torch.int32)
            torch.cumsum(
                kv_page_counts_cpu[req_indices], dim=0, out=group_kv_indptr[1:]
            )
            token_indices_cpu = torch.cat(
                [
                    torch.arange(
                        int(qo_indptr_prefill_cpu[i]),
                        int(qo_indptr_prefill_cpu[i + 1]),
                        dtype=torch.int64,
                    )
                    for i in req_indices.tolist()
                ]
            )
            page_gather_cpu = torch.cat(
                [
                    torch.arange(
                        int(paged_kv_indptr_prefill_cpu[i]),
                        int(paged_kv_indptr_prefill_cpu[i + 1]),
                        dtype=torch.int64,
                    )
                    for i in req_indices.tolist()
                ]
            )
            group_kv_indices = torch.index_select(
                paged_kv_indices, 0, page_gather_cpu.to(self.device)
            )
            wrapper = self._get_group_prefill_wrapper(group_mm, group_causal)
            if group_mm:
                # The packed mask carries the group's causal base AND the
                # sliding window AND the span bidirectionality wholesale, so
                # the wrapper plans causal=False, window_left=-1 regardless.
                assert prefill_mm_spans is not None
                custom_mask = self._build_mm_prefix_custom_mask(
                    [int(qo_lens_cpu[i]) for i in req_indices.tolist()],
                    [int(seq_lens_prefill_cpu[i]) for i in req_indices.tolist()],
                    [prefill_mm_spans[i] for i in req_indices.tolist()],
                    causal_base=group_causal,
                )
                plan_causal = False
                plan_window_left = -1
            else:
                custom_mask = None
                plan_causal = group_causal
                plan_window_left = self.window_left
            wrapper.plan(
                qo_indptr=group_qo_indptr,
                paged_kv_indptr=group_kv_indptr,
                paged_kv_indices=group_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu[
                    req_indices
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim_qk=self.head_dim,
                # == head_dim_qk unless the VO split is active; then the
                # impl runs each group's wrapper once per V slice.
                head_dim_vo=self.head_dim // self.vo_split,
                page_size=self.page_size,
                causal=plan_causal,
                custom_mask=custom_mask,
                sm_scale=self.sm_scale,
                window_left=plan_window_left,
                logits_soft_cap=self.logits_soft_cap,
                q_data_type=self.q_data_type,
                kv_data_type=self.kv_cache_dtype,
                o_data_type=o_dtype,
                fixed_split_size=self.prefill_fixed_split_size,
                disable_split_kv=self.disable_split_kv,
            )
            wrapper.vllm_prefill_fixed_split_size = self.prefill_fixed_split_size
            wrapper.vllm_disable_split_kv = self.disable_split_kv
            groups.append(
                FIPrefillGroup(
                    wrapper=wrapper,
                    token_indices=token_indices_cpu.to(self.device),
                    num_tokens=int(token_indices_cpu.numel()),
                )
            )
        return groups

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashInferMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        causal = common_attn_metadata.causal
        # DiffusionGemma passes a per-request causal tensor (encoder/commit
        # causal, denoise bidirectional, mixed within a batch). For dispatch it
        # behaves like the non-causal whole-batch path (all FI-native prefill,
        # no TRTLLM/decode); the per-request flags are consumed by the grouped
        # planner. Collapse to a scalar False for the decisions below -- the
        # legacy scalar-bool ``causal`` path is unchanged.
        per_request_causal = isinstance(causal, torch.Tensor) and num_reqs > 0
        causal_dispatch = False if per_request_causal else causal
        if causal_dispatch:
            num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
                split_decodes_and_prefills(
                    common_attn_metadata,
                    decode_threshold=self.reorder_batch_threshold,
                    require_uniform=True,
                )
            )
        else:
            # FlashInfer decode/TRTLLM paths cannot express non-causal
            # query-query attention, so DFlash / DiffusionGemma run as native
            # prefill.
            num_decodes = 0
            num_prefills = num_reqs
            num_decode_tokens = 0
            num_prefill_tokens = num_actual_tokens

        page_size = self.page_size
        max_seq_len = common_attn_metadata.max_seq_len
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        qo_indptr = common_attn_metadata.query_start_loc
        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        # Step 1: Decide which dispatch modes to use:
        # - Cascade attention (distinct mode)
        # - Prefill (FI native or TRTLLM)
        # - Decode (FI native or TRTLLM)
        use_cascade = common_prefix_len > 0
        uses_spec_reorder = self.reorder_batch_threshold > 1
        # Page sizes >= 128 must use trtllm-gen; force it for prefill too.
        prefill_force_trtllm = (
            True if page_size >= 128 else self.attention_config.use_trtllm_attention
        )
        prefill_use_trtllm = causal_dispatch and use_trtllm_attention(
            self.num_qo_heads,
            self.num_kv_heads,
            num_prefill_tokens,
            max_seq_len,
            self.dcp_world_size,
            self.cache_dtype,
            self.q_data_type,
            is_prefill=True,
            force_use_trtllm=prefill_force_trtllm,
            has_sinks=self.has_sinks,
            has_spec=uses_spec_reorder,
        )
        decode_use_trtllm = (
            causal_dispatch
            and self.use_trtllm_decode_attention
            and self.dcp_world_size <= 1
        )

        if not causal_dispatch and self.use_dcp:
            raise NotImplementedError(
                "FlashInfer non-causal prefill is not supported with DCP yet."
            )
        if not causal_dispatch and self.use_trtllm_decode_attention:
            logger.warning_once(
                "Using FlashInfer for draft model non-causal attention; TRTLLM "
                "can still be used for target model causal attention."
            )
        # mm-prefix: prefill requests whose image span intersects the
        # query window need the FI-native custom-mask path (TRTLLM has no
        # custom masks). Decode stays causal: spans live in the prompt.
        prefill_mm_spans: list[list[tuple[int, int]]] | None = None
        if self.mm_prefix_enabled and num_prefills > 0:
            prefill_mm_spans = self._mm_prefix_prefill_spans(
                common_attn_metadata, num_decodes, num_prefills
            )
            if prefill_mm_spans is not None:
                prefill_use_trtllm = False

        if per_request_causal and (
            use_cascade or self.use_dcp or self.reorder_batch_threshold > 1
        ):
            raise NotImplementedError(
                "Per-request causal flags (DiffusionGemma) require the "
                "FlashInfer-native prefill pathway (no cascade, DCP, or "
                "spec-decode batch reordering)."
            )

        all_uses_trtllm = causal_dispatch and (
            (num_prefills == 0 or prefill_use_trtllm)
            and (num_decodes == 0 or decode_use_trtllm)
        )

        if not all_uses_trtllm:
            if self.has_sinks:
                raise NotImplementedError(
                    "FlashInfer backend currently does not support attention "
                    "sinks, please use trtllm on blackwell or flash attention "
                    "on earlier GPUs."
                )

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

            # The q quantization is not supported for non-trtllm attention,
            # fall back to model dtype.
            self.q_data_type = self.model_config.dtype

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
            causal=causal,
            use_cascade=use_cascade,
            prefill=None,
            decode=None,
            cascade_wrapper=None,
        )

        # Guard access to seq_lens_cpu, which may not always be needed
        # and can be expensive to retrieve in async mode.
        # When all attention (both prefill and decode) uses TRTLLM,
        # seq_lens_cpu is not needed since TRTLLM paths use GPU tensors
        # (block_tables, seq_lens) directly.
        needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm
        seq_lens_cpu = common_attn_metadata.seq_lens_cpu if needs_seq_lens_cpu else None
        seq_lens_np = seq_lens_cpu.numpy() if seq_lens_cpu is not None else None
        num_blocks_np = (
            (seq_lens_np + (page_size - 1)) // page_size
            if seq_lens_np is not None
            else None
        )

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

        # Compute paged_kv_indices if necessary
        # paged_kv_indices is only needed for FlashInfer native paths;
        # TRTLLM paths use block_tables directly on GPU.
        needs_paged_kv_indices = use_cascade or not all_uses_trtllm
        if needs_paged_kv_indices:
            assert num_blocks_np is not None
            assert seq_lens_np is not None
            paged_kv_indices = self._compute_flashinfer_kv_metadata(
                num_blocks_np,
                seq_lens_np,
                block_table_tensor,
                num_reqs,
                page_size,
            )
        else:
            paged_kv_indices = None

        # Early-out for cascade attention
        if use_cascade:
            assert num_blocks_np is not None
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
                qo_indptr_arr=[shared_qo_indptr_cpu, qo_indptr_cpu],
                paged_kv_indptr_arr=[shared_kv_page_indptr_cpu, paged_kv_indptr_cpu],
                paged_kv_indices_arr=[shared_kv_page_indices_cpu, paged_kv_indices],
                paged_kv_last_page_len=[
                    shared_kv_last_page_len_cpu,
                    paged_kv_last_page_len_cpu,
                ],
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
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

            if prefill_use_trtllm:
                # Create GPU versions
                qo_indptr_prefill_gpu = (
                    qo_indptr[prefill_start:] - qo_indptr[prefill_start]
                )
                # Compute cum_seq_lens_kv on GPU to avoid CPU sync.
                # This is the cumulative sum of the number of KV cache
                # blocks per prefill request.
                prefill_seq_lens = seq_lens[prefill_start:]
                num_blocks_per_req = (prefill_seq_lens + page_size - 1) // page_size
                paged_kv_indptr_prefill_gpu = self.paged_kv_indptr.gpu[
                    prefill_start : num_reqs + 1
                ]
                # Assign to slice to avoid cpu sync.
                paged_kv_indptr_prefill_gpu[:1] = 0
                torch.cumsum(
                    num_blocks_per_req,
                    dim=0,
                    out=paged_kv_indptr_prefill_gpu[1:],
                )
                # Compute max_q_len for prefill requests
                query_lens_prefill_cpu = (
                    qo_indptr_prefill_cpu[1:] - qo_indptr_prefill_cpu[:-1]
                )
                max_q_len_prefill = int(query_lens_prefill_cpu.max().item())
                attn_metadata.prefill = TRTLLMPrefill(
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    cum_seq_lens_q=qo_indptr_prefill_gpu,
                    cum_seq_lens_kv=paged_kv_indptr_prefill_gpu,
                    max_q_len=max_q_len_prefill,
                    max_seq_len=max_seq_len,
                )
            else:
                # Per-request causal (DiffusionGemma) makes attn_metadata.causal
                # a tensor; the grouped planner overrides this wrapper anyway, so
                # fetch the primary causal wrapper as a placeholder (avoids the
                # tensor truth-check and the nvfp4 non-causal raise).
                prefill_wrapper = self._get_prefill_wrapper(
                    causal=True if per_request_causal else attn_metadata.causal
                )
                # Slicing CPU buffers that are only needed for FI native prefills
                paged_kv_last_page_len_prefill_cpu = self.paged_kv_last_page_len.cpu[
                    prefill_start:num_reqs
                ]
                assert paged_kv_last_page_len_prefill_cpu.shape[0] == num_prefills
                paged_kv_indptr_prefill_cpu = self.paged_kv_indptr.cpu[
                    prefill_start : num_reqs + 1
                ]
                assert paged_kv_indptr_prefill_cpu.shape[0] == num_prefills + 1
                prefill_groups: list[FIPrefillGroup] | None = None
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
                    # NVFP4 trtllm kernel only supports FP8 output;
                    # use FP8 o_data_type so the wrapper matches the
                    # FP8 output buffer allocated in forward().
                    o_dtype = (
                        FP8_DTYPE if (self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv) else self.model_config.dtype
                    )
                    if prefill_mm_spans is not None or per_request_causal:
                        assert seq_lens_cpu is not None
                        # The .cpu() sync on the causal tensor is acceptable
                        # here -- the scalar path's plan() consumes CPU arrays
                        # in this same spot anyway.
                        causal_prefill_cpu = (
                            common_attn_metadata.causal[prefill_start:num_reqs]
                            .cpu()
                            .bool()
                            if per_request_causal
                            else None
                        )
                        prefill_groups = self._plan_prefill_groups(
                            prefill_mm_spans,
                            causal_prefill_cpu,
                            qo_indptr_prefill_cpu,
                            paged_kv_indptr_prefill_cpu,
                            paged_kv_last_page_len_prefill_cpu,
                            paged_kv_indices,
                            seq_lens_cpu[prefill_start:num_reqs],
                            o_dtype,
                        )
                        # The impl's forward dispatches on prefill_groups; this
                        # field only feeds its isinstance/identity asserts.
                        prefill_wrapper = prefill_groups[0].wrapper
                    else:
                        prefill_wrapper.plan(
                            qo_indptr=qo_indptr_prefill_cpu,
                            paged_kv_indptr=paged_kv_indptr_prefill_cpu,
                            paged_kv_indices=paged_kv_indices,
                            paged_kv_last_page_len=paged_kv_last_page_len_prefill_cpu,
                            num_qo_heads=self.num_qo_heads,
                            num_kv_heads=self.num_kv_heads,
                            head_dim_qk=self.head_dim,
                            # == head_dim_qk unless the VO split is active;
                            # then each pass plans head_dim_vo=head_size//
                            # vo_split (256 for Gemma 4 512-wide heads) and
                            # the impl runs the wrapper once per V slice
                            # (_run_vo_split_prefill).
                            head_dim_vo=self.head_dim // self.vo_split,
                            page_size=self.page_size,
                            causal=attn_metadata.causal,
                            sm_scale=self.sm_scale,
                            window_left=self.window_left,
                            logits_soft_cap=self.logits_soft_cap,
                            q_data_type=self.q_data_type,
                            kv_data_type=self.kv_cache_dtype,
                            o_data_type=o_dtype,
                            fixed_split_size=self.prefill_fixed_split_size,
                            disable_split_kv=self.disable_split_kv,
                        )
                attn_metadata.prefill = FIPrefill(
                    wrapper=prefill_wrapper, prefill_groups=prefill_groups
                )

        ## DECODE PATHWAY
        if num_decodes > 0:
            if decode_use_trtllm:
                assert num_decode_tokens % num_decodes == 0, (
                    "TRTLLM decode requires uniform query lengths per request. "
                    f"Got {num_decode_tokens=} and {num_decodes=}."
                )
                attn_metadata.decode = TRTLLMDecode(
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )
            else:
                assert seq_lens_cpu is not None
                pure_decode = num_prefills == 0
                use_cudagraph = (
                    self.enable_cuda_graph
                    and pure_decode
                    and num_decode_tokens <= self._decode_cudagraph_max_bs
                )
                num_input_tokens = num_decode_tokens

                decode_wrapper = self._get_decode_wrapper(
                    num_input_tokens, use_cudagraph
                )
                # Use the persistent buffer with padding length,
                # instead of the same address but chunked version
                # in atten_metadata when using cudagraph.
                # NVFP4 trtllm kernel only supports FP8 output;
                # use FP8 o_data_type so the wrapper matches the
                # FP8 output buffer allocated in forward().
                o_dtype = (
                    FP8_DTYPE if (self.is_kvcache_nvfp4 and not self.use_fa2_nvfp4_kv) else self.model_config.dtype
                )
                fast_plan_decode(
                    decode_wrapper,
                    indptr_cpu=self.paged_kv_indptr.cpu[: num_input_tokens + 1],
                    indices=paged_kv_indices,
                    last_page_len_cpu=self.paged_kv_last_page_len.cpu[
                        :num_input_tokens
                    ],
                    num_qo_heads=self.num_qo_heads * self.dcp_world_size,
                    num_kv_heads=self.num_kv_heads,
                    head_dim=self.head_dim,
                    page_size=self.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.sm_scale,
                    window_left=self.window_left,
                    logits_soft_cap=self.logits_soft_cap,
                    q_data_type=self.q_data_type,
                    kv_data_type=self.kv_cache_dtype,
                    o_data_type=o_dtype,
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
        self.is_kvcache_nvfp4 = kv_cache_dtype == "nvfp4"
        self.use_fa2_nvfp4_kv = (
            self.is_kvcache_nvfp4
            and current_platform.is_device_capability_family(120)
        )
        self.fp4_data_dim = head_size // 2 if self.is_kvcache_nvfp4 else 0
        # Two-pass VO split factor for head_size > 256 (Gemma 4 D=512); 1
        # otherwise. Must match the builder's vo_split: the wrapper is
        # planned with head_dim_vo = head_size // vo_split, so the impl must
        # run it once per V slice when vo_split > 1.
        self.vo_split = _vo_split_factor(head_size, self.use_fa2_nvfp4_kv)
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

        self.sinks: torch.Tensor | None = None
        if sinks is not None:
            if sinks.shape[0] != num_heads:
                raise ValueError(
                    "Sinks must have the same number of heads as the number of "
                    f"heads in the layer. Expected {num_heads}, but got "
                    f"{sinks.shape[0]}."
                )
            self.sinks = sinks

        self.support_trtllm_attn = can_use_trtllm_attention(num_heads, num_kv_heads)
        vllm_config = get_current_vllm_config_or_none()
        self.supports_quant_query_input = (
            self.support_trtllm_attn
            and vllm_config is not None
            and not vllm_config.attention_config.disable_flashinfer_q_quantization
        )
        self.bmm1_scale: float | None = None
        self.bmm2_scale: float | None = None
        self.o_sf_scale: float | None = None

        # Pre-allocated FP8 output buffer for NVFP4 without fused output quant.
        if self.is_kvcache_nvfp4 and vllm_config is not None:
            max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            self._nvfp4_fp8_out = torch.empty(
                (max_num_tokens, num_heads, head_size),
                dtype=FP8_DTYPE,
                device="cuda",
            )
        else:
            self._nvfp4_fp8_out = None

        dcp_a2a = (
            vllm_config is not None
            and vllm_config.parallel_config.decode_context_parallel_size > 1
            and vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )
        if dcp_a2a:
            self.dcp_combine = partial(dcp_a2a_lse_reduce, is_lse_base_on_e=False)
        else:
            self.dcp_combine = partial(cp_lse_ag_out_rs, is_lse_base_on_e=False)

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return (
            self.support_trtllm_attn
            and is_quantized_kv_cache(self.kv_cache_dtype)
            and quant_key in (kFp8StaticTensorSym, kNvfp4Dynamic)
        )

    # FlashInfer requires attention sinks to be float32
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if self.sinks is not None and self.sinks.dtype != torch.float32:
            self.sinks = self.sinks.to(torch.float32)

    def _run_vo_split_prefill(
        self,
        wrapper: BatchPrefillWithPagedKVCacheWrapper,
        query: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        kv_sf: tuple[torch.Tensor, torch.Tensor] | None,
        out: torch.Tensor,
        *,
        k_scale: float,
        v_scale: float,
    ) -> None:
        """Multi-pass FA2 prefill run for head_size > 256 (Gemma 4 D=512).

        The wrapper is planned with head_dim_vo = head_size // vo_split, and
        each pass consumes a head-dim slice of V (and, for NVFP4, of the V
        scale factors): S = Q @ K^T and the softmax are recomputed
        identically per pass, so the per-pass outputs concatenate exactly
        along the head dim with no LSE merge. narrow() keeps the full
        tensor's strides, which the FA2 path requires.

        NVFP4 V slicing relies on the contiguous ``[all-data | all-SF]``
        cache layout (``contiguous_sf_layout=True``): the V data view's last
        dim is ``head_size // 2`` packed e2m1 bytes and the V scale view's
        last dim is ``head_size // 16`` fp8 scales, both contiguous along the
        head dim, so chunk ``i`` is data[i*chunk//2 : ...] and
        scale[i*chunk//16 : ...].
        """
        split = self.vo_split
        head_chunk = self.head_size // split
        k_cache, v_cache = kv_cache
        if self.is_kvcache_nvfp4:
            assert kv_sf is not None
            k_sf, v_sf = kv_sf
            data_step = head_chunk // 2  # packed e2m1, 2 elements per byte
            sf_step = head_chunk // 16  # one fp8 scale per 16 elements
        else:
            k_sf = v_sf = None
            data_step = head_chunk
            sf_step = 0
        for i in range(split):
            v_cache_i = v_cache.narrow(-1, i * data_step, data_step)
            kv_sf_i = (
                (k_sf, v_sf.narrow(-1, i * sf_step, sf_step))
                if v_sf is not None
                else None
            )
            # The kernel needs a contiguous output; write into a chunk
            # buffer and copy into the head-dim slice of the full output.
            out_i = torch.empty(
                (*out.shape[:-1], head_chunk),
                dtype=out.dtype,
                device=out.device,
            )
            wrapper.run(
                query,
                (k_cache, v_cache_i),
                k_scale=k_scale,
                v_scale=v_scale,
                out=out_i,
                kv_cache_sf=kv_sf_i,
            )
            out.narrow(-1, i * head_chunk, head_chunk).copy_(out_i)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: torch.Tensor,
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
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # Ensure query dtype matches the expected dtype from attention metadata
        assert attn_metadata.q_data_type == query.dtype, (
            f"Query dtype mismatch: expected {attn_metadata.q_data_type}, "
            f"got {query.dtype}"
        )

        if self.bmm1_scale is None:
            self.bmm1_scale = self.scale
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm1_scale *= layer._q_scale_float * layer._k_scale_float

        if self.bmm2_scale is None:
            self.bmm2_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                self.bmm2_scale *= layer._v_scale_float

        prefill_use_trtllm = isinstance(attn_metadata.prefill, TRTLLMPrefill)
        decode_use_trtllm = isinstance(attn_metadata.decode, TRTLLMDecode)

        # The attn+quant fusion happens when output_scale is provided.
        if output_scale is None:
            assert output_block_scale is None, (
                "output_block_scale is not supported when fusion has not happened"
            )
        else:
            assert attn_metadata.q_data_type == FP8_DTYPE, (
                "Query must be FP8 when attn+quant fusion happened."
            )
            assert (attn_metadata.num_prefills == 0 or prefill_use_trtllm) and (
                attn_metadata.num_decodes == 0 or decode_use_trtllm
            ), "Must use TRT-LLM attn"

            if output.dtype == FP8_DTYPE:
                assert output_block_scale is None, (
                    "output_block_scale should not be provided for fp8 output"
                )
            elif output.dtype == FP4_DTYPE:
                assert output_block_scale is not None, (
                    "output_block_scale is required for nvfp4 output"
                )
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

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # FlashInfer treats uint8 KV cache as NVFP4. vLLM stores FP8 KV cache
        # as uint8 bytes, so pass FP8 caches with their logical dtype.
        if not self.is_kvcache_nvfp4 and kv_cache.dtype == torch.uint8:
            fp8_view_dtype = None
            if self.kv_cache_dtype in ("fp8", "fp8_e4m3", torch.float8_e4m3fn):
                fp8_view_dtype = torch.float8_e4m3fn
            elif self.kv_cache_dtype in ("fp8_e5m2", torch.float8_e5m2):
                fp8_view_dtype = torch.float8_e5m2
            if fp8_view_dtype is not None:
                kv_cache = kv_cache.view(fp8_view_dtype)

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
        kv_cache_permute = kv_cache.permute(*stride_order)  # HND and contiguous
        # Fix degenerate strides on any size-1 dimension (e.g. num_kv_heads=1
        # with TP=8).  PyTorch permits non-canonical strides on size-1 dims;
        # CUDA TMA requires ≥16-byte alignment on all non-outermost strides.
        # canonicalize_singleton_dim_strides patches metadata via as_strided —
        # zero-copy.  See vllm.utils.torch_utils.
        fixed = canonicalize_singleton_dim_strides(kv_cache_permute)
        if fixed is not kv_cache_permute:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashInfer): "
                "shape=%s, strides before=%s, strides after=%s",
                kv_cache_permute.shape,
                kv_cache_permute.stride(),
                fixed.stride(),
            )
        kv_cache_permute = fixed

        # For NVFP4, the kv_cache last dim is full_dim (data + scale packed).
        # Split into correctly-strided data and scale views.
        nvfp4_kv_data = None
        nvfp4_kv_block_scales = None
        if self.is_kvcache_nvfp4:
            nvfp4_kv_data, nvfp4_kv_block_scales = nvfp4_kv_cache_split_views(
                kv_cache_permute, contiguous_sf_layout=self.use_fa2_nvfp4_kv
            )

        use_dcp = self.dcp_world_size > 1

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back.
        if num_prefill_tokens > 0:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens

            if not prefill_use_trtllm:
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
                    assert isinstance(
                        prefill_wrapper, BatchPrefillWithPagedKVCacheWrapper
                    )
                    prefill_groups = attn_metadata.prefill.prefill_groups
                    assert prefill_wrapper._logits_soft_cap == (
                        self.logits_soft_cap or 0.0
                    )
                    assert prefill_wrapper._sm_scale == self.scale
                    if prefill_groups is None:
                        # Legacy fast path: single scalar-causal wrapper.
                        assert prefill_wrapper._window_left == self.window_left
                        assert prefill_wrapper._causal == attn_metadata.causal
                    else:
                        # Grouped prefill. Each group's wrapper carries its own
                        # semantics from plan() time: masked (mm) groups fold
                        # causal base + sliding window + spans into the packed
                        # mask (planned non-causal, window_left=-1); unmasked
                        # groups plan the layer-group window and their per-group
                        # causal flag (True for plain causal, False for a
                        # DiffusionGemma non-causal denoise group).
                        for group in prefill_groups:
                            if group.wrapper._custom_mask_buf is None:
                                assert (
                                    group.wrapper._window_left == self.window_left
                                )
                            else:
                                assert not group.wrapper._causal
                                assert group.wrapper._window_left == -1

                    if self.is_kvcache_nvfp4:
                        kv_cache_permute = nvfp4_kv_data
                    kv_cache_sf = (
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    )

                    # NVFP4 trtllm kernel only supports FP8 output.
                    # Use a pre-allocated FP8 buffer and dequantize
                    # afterwards.
                    needs_fp8_out_prefill = (
                        self.is_kvcache_nvfp4
                        and output.dtype != FP8_DTYPE
                        and not self.use_fa2_nvfp4_kv
                    )
                    if needs_fp8_out_prefill:
                        out_prefill = self._nvfp4_fp8_out[:num_prefill_tokens]
                    else:
                        out_prefill = output[num_decode_tokens:]

                    if prefill_groups is not None:
                        # Grouped prefill: each attention-semantics partition
                        # (plain causal, plain non-causal, or a packed
                        # custom-mask mm group) runs its own planned wrapper
                        # over a gathered copy of its query rows, then scatters
                        # back. Gather/scatter is by token index because the
                        # groups interleave within the batch. The VO-split
                        # branch is taken per group as on the single path.
                        if self.is_kvcache_nvfp4:
                            assert isinstance(kv_cache_permute, tuple)
                        for group in prefill_groups:
                            group_query = torch.index_select(
                                prefill_query, 0, group.token_indices
                            )
                            group_out = torch.empty(
                                (group.num_tokens, *out_prefill.shape[1:]),
                                dtype=out_prefill.dtype,
                                device=out_prefill.device,
                            )
                            if self.vo_split > 1:
                                self._run_vo_split_prefill(
                                    group.wrapper,
                                    group_query,
                                    kv_cache_permute,
                                    kv_cache_sf,
                                    group_out,
                                    k_scale=layer._k_scale_float,
                                    v_scale=layer._v_scale_float,
                                )
                            else:
                                group.wrapper.run(
                                    group_query,
                                    kv_cache_permute,
                                    k_scale=layer._k_scale_float,
                                    v_scale=layer._v_scale_float,
                                    out=group_out,
                                    kv_cache_sf=kv_cache_sf,
                                )
                            out_prefill.index_copy_(
                                0, group.token_indices, group_out
                            )
                    elif self.vo_split > 1:
                        # head_size > 256: run ceil(head_size/256) passes,
                        # each over a head-dim slice of the 512-wide V cache,
                        # and concatenate the per-pass outputs. The wrapper is
                        # planned with head_dim_vo = head_size // vo_split.
                        if self.is_kvcache_nvfp4:
                            assert isinstance(kv_cache_permute, tuple)
                            assert isinstance(kv_cache_sf, tuple)
                        self._run_vo_split_prefill(
                            prefill_wrapper,
                            prefill_query,
                            kv_cache_permute,
                            kv_cache_sf,
                            out_prefill,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                        )
                    else:
                        prefill_wrapper.run(
                            prefill_query,
                            kv_cache_permute,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_prefill,
                            kv_cache_sf=kv_cache_sf,
                        )

                    if needs_fp8_out_prefill:
                        output[
                            num_decode_tokens : num_decode_tokens + num_prefill_tokens
                        ].copy_(out_prefill.to(output.dtype))
            else:
                assert isinstance(attn_metadata.prefill, TRTLLMPrefill)
                # prefill_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                prefill_query = prefill_query.contiguous()
                prefill_query = canonicalize_singleton_dim_strides(prefill_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_prefill = attn_metadata.prefill.block_tables
                seq_lens_prefill = attn_metadata.prefill.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(prefill_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_prefill)
                assert is_strictly_contiguous(seq_lens_prefill)

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[num_decode_tokens:],
                        scale=output_block_scale,
                        scale_start_index=num_decode_tokens,
                        original_shape=prefill_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[num_decode_tokens:]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = (
                    self.is_kvcache_nvfp4
                    and output.dtype != FP8_DTYPE
                    and not self.use_fa2_nvfp4_kv
                )
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_prefill_tokens]

                prefill_kv_block_scales = None
                if self.is_kvcache_nvfp4:
                    # NVFP4 trtllm-gen kernel requires FP8 query.
                    assert attn_metadata.q_data_type == FP8_DTYPE, (
                        "NVFP4 KV cache requires FP8 quantized queries for "
                        "trtllm-gen prefill. Set "
                        "disable_flashinfer_q_quantization=False."
                    )
                    mock_kv_cache = nvfp4_kv_data
                    mock_block_table = block_tables_prefill
                    prefill_kv_block_scales = nvfp4_kv_block_scales
                elif (
                    attn_metadata.q_data_type != FP8_DTYPE
                    and self.kv_cache_dtype.startswith("fp8")
                ):
                    # TRTLLM prefill attention does not support BF16 Q
                    # and fp8 kv cache. So to enable prefill attention
                    # with fp8 kv cache, we can construct a mock block
                    # and mock kv cache with BF16 KV involved in the prefill
                    #
                    kv_cache_permute = canonicalize_singleton_dim_strides(
                        kv_cache_permute
                    )
                    kv_strides = kv_cache_permute.stride()
                    assert (
                        kv_strides[-1] == 1
                        and kv_strides[-2] == kv_cache_permute.shape[-1]
                    ), (
                        "KV cache inner dims (block_size, head_size) must be "
                        f"contiguous, got strides {kv_strides}"
                    )
                    mock_kv_cache, mock_block_table = trtllm_prefill_attn_kvfp8_dequant(
                        kv_cache_permute,
                        block_tables_prefill,
                        layer._k_scale,
                        layer._v_scale,
                        attn_metadata.q_data_type,
                    )
                else:
                    mock_kv_cache = kv_cache_permute
                    mock_block_table = block_tables_prefill

                trtllm_batch_context_with_kv_cache(
                    query=prefill_query,
                    kv_cache=mock_kv_cache,
                    workspace_buffer=workspace_buffer,
                    block_tables=mock_block_table,
                    seq_lens=seq_lens_prefill,
                    max_q_len=attn_metadata.prefill.max_q_len,
                    max_kv_len=attn_metadata.prefill.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    batch_size=attn_metadata.num_prefills,
                    cum_seq_lens_q=attn_metadata.prefill.cum_seq_lens_q,
                    cum_seq_lens_kv=attn_metadata.prefill.cum_seq_lens_kv,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    kv_cache_sf=prefill_kv_block_scales,
                )

                if needs_fp8_out:
                    output[
                        num_decode_tokens : num_decode_tokens + num_prefill_tokens
                    ].copy_(out[:num_prefill_tokens].to(output.dtype))

        if num_decode_tokens > 0:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens

            # The VO split routes every request (including would-be decodes)
            # through the prefill wrapper (builder sets reorder_batch_threshold
            # = 0), because BatchDecodeWithPagedKVCacheWrapper.plan() has no
            # head_dim_vo. So no decode tokens should reach this block.
            assert self.vo_split == 1, (
                "FA2 VO split routes decodes through the prefill wrapper; "
                f"unexpected {num_decode_tokens} decode tokens with "
                f"vo_split={self.vo_split}."
            )

            if not decode_use_trtllm:
                assert isinstance(attn_metadata.decode, FIDecode)
                decode_wrapper = attn_metadata.decode.wrapper
                assert decode_wrapper is not None
                assert decode_wrapper._window_left == self.window_left
                assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap or 0.0)
                assert decode_wrapper._sm_scale == self.scale

                if self.is_kvcache_nvfp4:
                    kv_cache_permute = nvfp4_kv_data
                kv_cache_sf = nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None

                # NVFP4 kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = (
                    self.is_kvcache_nvfp4
                    and output.dtype != FP8_DTYPE
                    and not self.use_fa2_nvfp4_kv
                )
                if needs_fp8_out:
                    out_decode = self._nvfp4_fp8_out[:num_decode_tokens]
                else:
                    out_decode = output[:num_decode_tokens]

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
                        kv_cache_sf=kv_cache_sf,
                    )
                    output[:num_decode_tokens] = self.dcp_combine(
                        output_tmp,
                        lse,
                        get_dcp_group(),
                    )
                else:
                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                    )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out_decode.to(output.dtype))
            else:
                assert isinstance(attn_metadata.decode, TRTLLMDecode)
                # decode_query may be non-contiguous or have degenerate strides
                # on size=1 dims. contiguous() ensures memory layout; then
                # canonicalize_singleton_dim_strides fixes any remaining
                # degenerate strides on size=1 dims for TMA alignment.
                decode_query = decode_query.contiguous()
                decode_query = canonicalize_singleton_dim_strides(decode_query)
                workspace_buffer = _get_trtllm_gen_workspace_buffer()
                block_tables_decode = attn_metadata.decode.block_tables
                seq_lens_decode = attn_metadata.decode.seq_lens

                # This path needs to be enabled with VLLM_KV_CACHE_LAYOUT = HND
                assert get_kv_cache_layout() == "HND"
                assert is_strictly_contiguous(decode_query)
                assert is_strictly_contiguous(workspace_buffer)
                assert is_strictly_contiguous(block_tables_decode)
                assert is_strictly_contiguous(seq_lens_decode)
                kv_cache_permute = canonicalize_singleton_dim_strides(kv_cache_permute)
                kv_strides = kv_cache_permute.stride()
                assert (
                    kv_strides[-1] == 1 and kv_strides[-2] == kv_cache_permute.shape[-1]
                ), (
                    "KV cache inner dims (block_size, head_size) must be "
                    f"contiguous, got strides {kv_strides}"
                )

                if output.dtype == FP4_DTYPE:
                    assert self.o_sf_scale is not None
                    out = FP4Tensor(
                        data=output[:num_decode_tokens],
                        scale=output_block_scale,
                        scale_start_index=0,
                        original_shape=decode_query.shape,
                    )
                else:
                    assert self.o_sf_scale is None
                    out = output[:num_decode_tokens]

                # NVFP4 trtllm kernel only supports FP8 output.
                # Use a pre-allocated FP8 buffer and dequantize afterwards.
                needs_fp8_out = (
                    self.is_kvcache_nvfp4
                    and output.dtype != FP8_DTYPE
                    and not self.use_fa2_nvfp4_kv
                )
                if needs_fp8_out:
                    out = self._nvfp4_fp8_out[:num_decode_tokens]

                if num_decode_tokens % attn_metadata.num_decodes != 0:
                    # This gets triggered when the dummy_run forces
                    # attention to be initialized with q_len = 0
                    q_len_per_req = 1
                else:
                    q_len_per_req = num_decode_tokens // attn_metadata.num_decodes

                trtllm_batch_decode_with_kv_cache(
                    query=decode_query,
                    kv_cache=(
                        nvfp4_kv_data if self.is_kvcache_nvfp4 else kv_cache_permute
                    ),
                    workspace_buffer=workspace_buffer,
                    block_tables=block_tables_decode,
                    seq_lens=seq_lens_decode,
                    max_seq_len=attn_metadata.decode.max_seq_len,
                    bmm1_scale=self.bmm1_scale,
                    bmm2_scale=self.bmm2_scale,
                    window_left=self.window_left,
                    sinks=self.sinks,
                    o_sf_scale=self.o_sf_scale,
                    out=out,
                    q_len_per_req=q_len_per_req,
                    kv_cache_sf=(
                        nvfp4_kv_block_scales if self.is_kvcache_nvfp4 else None
                    ),
                )

                if needs_fp8_out:
                    output[:num_decode_tokens].copy_(out.to(output.dtype))
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
            k_cache = kv_cache[:, 0]
            v_cache = kv_cache[:, 1]
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                k_cache,
                v_cache,
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
            indptr=indptr_cpu,
            indices=indices,
            last_page_len=last_page_len_cpu,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            o_data_type=o_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            block_tables=None,
            seq_lens=None,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )
        self.vllm_first_call = False
        return

    assert self.is_cuda_graph_enabled, "Should be cudagraph only here"

    fast_decode_plan(
        self,
        indptr=indptr_cpu,
        indices=indices,
        last_page_len=last_page_len_cpu,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        data_type=data_type,
        sm_scale=sm_scale,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
        non_blocking=non_blocking,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


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
