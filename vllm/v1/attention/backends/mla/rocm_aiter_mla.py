# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import functools
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Final

import regex as re
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    QueryLenSupport,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, largest_power_of_2_divisor
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    get_num_attention_heads_from_layers,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.rocm_aiter_mla_reduce import (
    reduce_mla_segment_partials,
)
from vllm.v1.kv_cache_interface import AttentionSpec, is_quantized_kv_cache

logger = init_logger(__name__)


def _segmented_mla_page_size(block_size: int) -> int:
    """Largest supported power-of-two subpage dividing a physical KV block."""
    assert block_size > 0
    return min(128, largest_power_of_2_divisor(block_size))


@functools.lru_cache(maxsize=1)
def _get_mla_gluon():
    """Load the small-head Gluon MLA entry point."""
    unified_module = "aiter.ops.triton.gluon.mla_gluon"
    try:
        from aiter.ops.triton.gluon.mla_gluon import mla_gluon

        return mla_gluon
    except ModuleNotFoundError as unified_import_error:
        if not unified_module.startswith(unified_import_error.name or ""):
            raise
        legacy_module = "aiter.ops.triton.gluon.mla_decode_gluon"
        try:
            from aiter.ops.triton.gluon.mla_decode_gluon import mla_decode_gluon

            return mla_decode_gluon
        except ModuleNotFoundError as legacy_import_error:
            if not legacy_module.startswith(legacy_import_error.name or ""):
                raise
            raise RuntimeError(
                "ROCM_AITER_MLA requires an AITER build with the small-head "
                "Gluon MLA kernel (mla_gluon or mla_decode_gluon) when decode "
                "heads are fewer than 16."
            ) from unified_import_error


@functools.lru_cache(maxsize=1)
def _fp8_mla_prefill_supported() -> bool:
    """Auto-detect FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

    Requires gfx950 plus an AITER build that exports both kernels.  When
    either is missing we silently fall back to ``flash_attn_varlen_func``.
    """
    try:
        from vllm.platforms.rocm import on_gfx950
    except Exception:  # noqa: BLE001
        return False
    if not on_gfx950():
        return False
    try:
        from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


@functools.lru_cache(maxsize=1)
def _aiter_mla_native_h24_reducer_supported() -> bool:
    """Whether AITER's JIT reducer supports the native H24/512 shape."""
    try:
        from aiter.jit.core import AITER_CSRC_DIR

        reduce_source = Path(AITER_CSRC_DIR) / "kernels" / "mla" / "reduce.cu"
        source = "".join(reduce_source.read_text(encoding="utf-8").split())
    except (ImportError, OSError):
        return False
    return "MLA_REDUCE_CASE_EF(NUM_HEAD,24,HEAD_DIM,512," in source


@functools.lru_cache(maxsize=1)
def _aiter_mla_native_h24_metadata_supported() -> bool:
    """Whether AITER's fast MLA metadata planner accepts native H24.

    The reducer and metadata planner have independent shape dispatch. Checking
    only the reducer can route H24 into a planner that rejects it before the
    attention kernel launches. Until AITER exposes a capability API, inspect
    the shipped JIT source for an explicit native-H24 planner branch.
    """
    try:
        from aiter.jit.core import AITER_CSRC_DIR

        metadata_source = (
            Path(AITER_CSRC_DIR) / "kernels" / "mla" / "metadata" / "v1_2_device.cuh"
        )
        source = "".join(metadata_source.read_text(encoding="utf-8").split())
    except (ImportError, OSError):
        return False
    return "num_heads==24" in source


def _aiter_mla_native_h24_supported() -> bool:
    """Whether the complete AITER decode path supports native H24."""
    return (
        _aiter_mla_native_h24_reducer_supported()
        and _aiter_mla_native_h24_metadata_supported()
    )


@functools.lru_cache(maxsize=1)
def _gluon_mla_decode_supported() -> bool:
    """The small-head Gluon MLA decode kernel only has a gfx950 (CDNA4) build.

    Its tiling needs ~160 KiB of LDS, which exceeds CDNA3's 64 KiB, so on
    gfx942 there is no kernel to fall through to and selecting it asserts
    (``mla_gluon requires gfx950``). Restrict Gluon decode to gfx950; other
    archs use the asm persistent decode, which ``get_mla_padded_q`` makes
    correct for any 1..15 heads.
    """
    try:
        from vllm.platforms.rocm import on_gfx950
    except Exception:  # noqa: BLE001
        return False
    return on_gfx950()


@functools.lru_cache(maxsize=1)
def _get_segmented_mla_decode():
    """Load AITER's segmented MLA decode with unreduced partial output."""
    from aiter.ops.triton.attention.mla import mla_decode_fwd

    return mla_decode_fwd


@functools.lru_cache(maxsize=1)
def _get_aiter_mla_decode():
    from aiter.mla import mla_decode_fwd

    return mla_decode_fwd


@functools.lru_cache(maxsize=1)
def _segmented_mla_decode_supported() -> bool:
    """Whether AITER exposes the segmented MLA decode used by DCP verify."""
    if not _gluon_mla_decode_supported():
        return False
    try:
        _get_segmented_mla_decode()
    except Exception:  # noqa: BLE001
        return False
    return True


@functools.lru_cache(maxsize=1)
def _gluon_mla_wrapper_source() -> str | None:
    try:
        return inspect.getsource(_get_mla_gluon())
    except Exception:  # noqa: BLE001
        return None


_GLUON_MAX_HEADS_PATTERN = re.compile(r"requires nhead <= (\d+)")


@functools.lru_cache(maxsize=1)
def _gluon_mla_max_bh16_heads() -> int:
    """Read the supported head bound from the installed AITER wrapper."""
    fallback = AiterMLAHelper._AITER_MIN_MLA_HEADS
    source = _gluon_mla_wrapper_source()
    if source is None:
        return fallback
    match = _GLUON_MAX_HEADS_PATTERN.search(source)
    if match is None:
        return fallback
    return max(fallback, int(match.group(1)))


def _aiter_mla_small_head_mode() -> str:
    """Small-head (<16) MLA decode kernel selection.

    Controlled by ``VLLM_ROCM_AITER_MLA_ASM_PADDING``:

    - ``"auto"`` (default): let the arch decide -- divisor head counts keep the
      Gluon decode where a build exists (gfx950), everything else (non-divisor
      counts and all counts on gfx942) uses the padded persistent-scheduling
      ASM decode.
    - ``"gluon"``: prefer the Gluon path wherever a build exists.
    - ``"asm"``: force the padded persistent-scheduling ASM decode.

    On gfx942 (no Gluon build) the ASM path is always used regardless of this
    setting; ``"gluon"`` there falls back to ASM with a one-time warning.
    """
    import vllm.envs as envs

    mode = (envs.VLLM_ROCM_AITER_MLA_ASM_PADDING or "auto").lower()
    if mode == "gluon" and not _gluon_mla_decode_supported():
        logger.warning_once(
            "VLLM_ROCM_AITER_MLA_ASM_PADDING=gluon requested, but this device "
            "has no Gluon MLA decode build (Gluon requires gfx950); using the "
            "padded persistent-scheduling ASM decode instead."
        )
    return mode


def _dense_causal_mla_attn(
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_window: torch.Tensor,
    scale: float,
    qlen: int,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense causal MLA over the in-hand verify window.

    ``q_*`` are ``[R*qlen, H, D]``; ``k_window`` is ``[R*qlen, kv_lora+rope]``
    in the same request/token order. Returns ``(out, lse)`` with natural-log
    LSE so it combines with Gluon ``return_lse=True``.
    """
    num_rows, num_heads, _ = q_nope.shape
    assert num_rows % qlen == 0, f"rows {num_rows} not divisible by qlen={qlen}"
    bs = num_rows // qlen
    q = torch.cat([q_nope, q_pe], dim=-1)
    qb = q.view(bs, qlen, num_heads, -1)
    kb = k_window.view(bs, qlen, -1)
    scores = torch.einsum("bihd,bjd->bhij", qb, kb).float() * scale
    causal = torch.ones(qlen, qlen, dtype=torch.bool, device=q.device).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [bs, H, qlen], natural log
    probs = torch.exp(scores - lse.unsqueeze(-1))
    vb = kb[..., :kv_lora_rank].float()
    out = torch.einsum("bhij,bjd->bihd", probs, vb)
    return (
        out.reshape(num_rows, num_heads, kv_lora_rank).to(q_nope.dtype),
        lse.permute(0, 2, 1).reshape(num_rows, num_heads),
    )


def _lse_combine_natural(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two partial attentions over disjoint keys (natural-log LSE)."""
    out = torch.empty_like(out_a, dtype=out_dtype)
    lse = torch.empty_like(lse_a.transpose(0, 1))
    merge_attn_states(
        output=out,
        output_lse=lse,
        prefix_output=out_a,
        prefix_lse=lse_a.transpose(0, 1),
        suffix_output=out_b,
        suffix_lse=lse_b.transpose(0, 1),
    )
    return out, lse.transpose(0, 1)


class AiterMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The aiter MLA decode kernel always operates with page_size=1
        # internally (the wrapper flattens kv_buffer via .view(-1, 1, 1, H)).
        # We support any kernel_block_size by expanding block-level indices
        # into per-token flat indices in the metadata builder.
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "ROCM_AITER_MLA"

    @staticmethod
    def get_impl_cls() -> type["AiterMLAImpl"]:
        return AiterMLAImpl

    @staticmethod
    def get_builder_cls() -> type["AiterMLAMetadataBuilder"]:
        return AiterMLAMetadataBuilder


@dataclass
class AiterMLADecodeMetadata(MLACommonDecodeMetadata):
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: torch.Tensor | None = None
    # The page indices of the paged kv cache
    paged_kv_indices: torch.Tensor | None = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: torch.Tensor | None = None
    # The query indptr, shape : [num_decode + 1]
    qo_indptr: torch.Tensor | None = None
    # The dtype of MLA out tensor
    attn_out_dtype: torch.dtype = torch.bfloat16
    # The max query output length: int
    max_qo_len: int | None = None
    # Kernel max_qo_len after optional ASM LSE-PS pad (None = no pad).
    kernel_max_qo_len: int | None = None
    # Minimum KV length used by Gluon to choose a safe split count.
    min_kv_seq_len: int = 1
    # Per-token causal KV views for multi-token Gluon verification.
    verify_row_indptr: torch.Tensor | None = None
    verify_row_page_table: torch.Tensor | None = None
    verify_row_lens: torch.Tensor | None = None
    verify_min_kv_seq_len: int = 1
    # Committed-shard metadata for segmented DCP verification.
    dcp_verify_block_table: torch.Tensor | None = None
    dcp_verify_qo_indptr: torch.Tensor | None = None
    dcp_verify_max_kv_seq_len: int = 1
    # Small-head decode uses Gluon (avoids padding to 16).
    use_gluon_decode: bool = False
    # Whether persistent MLA metadata was computed
    has_persistent_metadata: bool = False


@dataclass
class AiterMLAMetadata(MLACommonMetadata[AiterMLADecodeMetadata]):
    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # FP8 ASM prefill persistent-scheduling (PS) metadata.  Populated by
    # AiterMLAMetadataBuilder._build_fp8_prefill_ps_metadata when prefill
    # tokens are present and FP8 MLA prefill is supported on the device.
    # Left as None on hosts/configs that fall back to flash_attn_varlen_func.
    fp8_prefill_qo_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indptr: torch.Tensor | None = None
    fp8_prefill_kv_indices: torch.Tensor | None = None
    fp8_prefill_work_indptr: torch.Tensor | None = None
    fp8_prefill_work_info_set: torch.Tensor | None = None
    fp8_prefill_reduce_indptr: torch.Tensor | None = None
    fp8_prefill_reduce_final_map: torch.Tensor | None = None
    fp8_prefill_reduce_partial_map: torch.Tensor | None = None
    fp8_prefill_max_q_len: int | None = None
    fp8_prefill_num_partial_tiles: int | None = None


# Tile size used by the mla_prefill_ps_asm_fwd assembly kernel.
_FP8_PREFILL_TILE_Q = 256


class AiterMLAMetadataBuilder(MLACommonMetadataBuilder[AiterMLAMetadata]):
    # TODO(luka, lucas): audit this as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.UNIFORM

    @staticmethod
    def _uniform_padded_mtp_qo_len(
        qo_len: torch.Tensor,
        max_qo_len: int,
        num_decode_tokens: int,
    ) -> int:
        num_reqs = qo_len.numel()
        if num_reqs == 0 or num_decode_tokens <= 0:
            return 0

        # Full-CG pads q to a captured token count while leaving
        # query_start_loc flat for dummy requests. Only synthesize dummy rows
        # when every padded request maps to the same qlen and the q buffer has
        # exactly that many rows.
        if num_decode_tokens <= int(qo_len.sum().item()):
            return 0
        if num_decode_tokens % num_reqs != 0:
            return 0

        uniform_qo_len = num_decode_tokens // num_reqs
        if uniform_qo_len <= 1:
            return 0

        positive_qo_len = qo_len[qo_len > 0]
        if positive_qo_len.numel() == qo_len.numel():
            return 0
        if positive_qo_len.numel() > 0:
            if max_qo_len != uniform_qo_len:
                return 0
            if not torch.all(positive_qo_len == uniform_qo_len):
                return 0

        zero_positions = torch.nonzero(qo_len == 0, as_tuple=False).flatten()
        if zero_positions.numel() > 0:
            first_zero = int(zero_positions[0].item())
            if torch.any(qo_len[first_zero:] > 0):
                return 0

        return uniform_qo_len

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        parallel_config = vllm_config.parallel_config
        group_dcp_world_size = parallel_config.decode_context_parallel_size
        # A draft group can have its own head count, so resolve it per layer the
        # way the common builder does rather than off the model config.
        gathered_num_heads = (
            get_num_attention_heads_from_layers(vllm_config, layer_names)
            or vllm_config.model_config.get_num_attention_heads(parallel_config)
        ) * group_dcp_world_size
        supports_dcp_with_varlen = (
            parallel_config.cp_kv_cache_interleave_size == 1
            and AiterMLAHelper.use_gluon_verify(
                gathered_num_heads,
                2,
                vllm_config.cache_config.cache_dtype,
                group_dcp_world_size,
            )
        )
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            AiterMLAMetadata,
            supports_dcp_with_varlen=supports_dcp_with_varlen,
        )

        self.compilation_config = vllm_config.compilation_config
        self.decode_attn_out_dtype = vllm_config.model_config.dtype

        # Needed to place a verify row's causal window on this rank's KV shard.
        self.dcp_rank = get_dcp_group().rank_in_group if self.dcp_world_size > 1 else 0

        # reorder_batch_threshold is the largest query length decode can be
        # handed, and already accounts for the drafting scheme. A method-name
        # whitelist sizes unlisted drafters for qlen=1, which closes the
        # persistent gate below and makes aiter raise a KeyError mid-run.
        self._mtp_decode_qlen = self.reorder_batch_threshold or 1

        # Store the kernel block size from the spec. When kernel_block_size=1
        # (no spec-dec), behavior is identical to the original. When > 1
        # (e.g. 16 with Eagle3), we expand block-level indices into per-token
        # flat indices since the aiter kernel always uses page_size=1 internally.
        self.kernel_block_size = kv_cache_spec.block_size
        num_dcp_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
        self._dcp_verify_graph_max_kv_seq_len = (
            cdiv(vllm_config.model_config.max_model_len, num_dcp_partitions)
            * self.cp_kv_cache_interleave_size
        )

        # In the flat view (.view(-1,1,1,H)), each token is its own page,
        # so max_num_pages_per_req = max_model_len regardless of
        # kernel_block_size.
        max_num_pages_per_req = vllm_config.model_config.max_model_len
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        # Preparing persistent buffers
        # TODO: we can disambiguate between decode and mixed-prefill decode here
        # so we can only use the persistent buffer if a cudagraph is actually
        # being used.

        # paged_kv_last_page_len is always 1s (the aiter kernel always sees
        # page_size=1 after .view(-1,1,1,H) flattening), so we create it
        # once and reuse slices in both eager and cudagraph modes.
        self.paged_kv_last_page_len = torch.ones(
            max_num_reqs, dtype=torch.int32, device=device
        )

        # Persistent buffer for paged_kv_indices to avoid blocking boolean mask
        # indexing (block_table_tensor[mask]) which has data-dependent output size.
        self.paged_kv_indices = torch.zeros(
            max_num_pages, dtype=torch.int32, device=device
        )

        from aiter import dtypes, get_mla_metadata_info_v1

        # Decode kernels consume the DCP-gathered query heads.
        self._decode_num_heads = self.num_heads * self.dcp_world_size
        # Keep metadata sizing consistent with the padded tensor shape passed
        # to mla_decode_fwd, including native 24-head AITER builds.
        self._num_attention_heads = AiterMLAHelper.get_actual_mla_num_heads(
            self._decode_num_heads
        )
        kv_cache_dtype_str = getattr(vllm_config.cache_config, "cache_dtype", "auto")
        if kv_cache_dtype_str in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            kv_cache_dtype_str = "fp8"
            kv_dtype = dtypes.fp8
        else:
            kv_dtype = {
                torch.float16: dtypes.fp16,
                torch.bfloat16: dtypes.bf16,
            }[kv_cache_spec.dtype]
        # _build_decode needs the cache dtype to pick the decode kernel; keep
        # the normalized string instead of dropping it at the end of __init__.
        self._kv_cache_dtype_str = kv_cache_dtype_str
        # MLAAttention quantizes decode Q to FP8 before calling this backend
        # whenever the KV cache is FP8 and supports_quant_query_input is true.
        q_dtype = (
            dtypes.fp8 if kv_cache_dtype_str == "fp8" else self.decode_attn_out_dtype
        )
        # Persist for get_mla_metadata_v1 (decode build): omitting these causes
        # wrong split/reduce metadata for the gfx950 fp8 nhead=32 fold path.
        self._mla_q_dtype = q_dtype
        self._mla_kv_dtype = kv_dtype
        (
            (work_meta_data_size, work_meta_data_type),
            (work_indptr_size, work_indptr_type),
            (work_info_set_size, work_info_set_type),
            (reduce_indptr_size, reduce_indptr_type),
            (reduce_final_map_size, reduce_final_map_type),
            (reduce_partial_map_size, reduce_partial_map_type),
        ) = get_mla_metadata_info_v1(
            max_num_reqs,
            self._mtp_decode_qlen,
            self._num_attention_heads,
            q_dtype,
            kv_dtype,
            is_sparse=False,
            fast_mode=True,
        )
        self._mla_work_meta_data = torch.empty(
            work_meta_data_size, dtype=work_meta_data_type, device=device
        )
        self._mla_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_type, device=device
        )
        self._mla_work_info_set = torch.empty(
            work_info_set_size, dtype=work_info_set_type, device=device
        )
        self._mla_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_type, device=device
        )
        self._mla_reduce_final_map = torch.empty(
            reduce_final_map_size, dtype=reduce_final_map_type, device=device
        )
        self._mla_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_type,
            device=device,
        )

        # FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads.
        self._fp8_prefill_enabled = (
            _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
        )
        if self._fp8_prefill_enabled:
            max_prefill_qlen = min(
                vllm_config.model_config.max_model_len,
                vllm_config.scheduler_config.max_num_batched_tokens,
            )
            self._init_fp8_prefill_ps_buffers(
                max_num_reqs,
                max_prefill_qlen,
                vllm_config.scheduler_config.max_num_batched_tokens,
                device,
            )

        # Persistent buffers for the verify flatten's per-row paged-KV view. A
        # captured graph reads the addresses handed to it at capture time, and
        # the view's page count varies per step, so it cannot be a fresh
        # allocation. Only sized when a verify can actually take the flatten.
        self._verify_row_indptr: torch.Tensor | None = None
        self._verify_row_page_table: torch.Tensor | None = None
        self._verify_row_lens: torch.Tensor | None = None
        self._dcp_verify_block_table: torch.Tensor | None = None
        self._dcp_verify_qo_indptr: torch.Tensor | None = None
        self._graph_seq_lens: torch.Tensor | None = None
        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            self.qo_indptr = torch.zeros(
                max_num_reqs + 1, dtype=torch.int32, device=device
            )

            # Full graphs require a stable address after uniform-MTP padding.
            self._graph_seq_lens = torch.zeros(
                max_num_reqs, dtype=torch.int32, device=device
            )

            if AiterMLAHelper.use_gluon_verify(
                self._decode_num_heads,
                self._mtp_decode_qlen,
                self._kv_cache_dtype_str,
                self.dcp_world_size,
            ):
                max_verify_rows = max_num_reqs * self._mtp_decode_qlen
                self._verify_row_lens = torch.zeros(
                    max_verify_rows, dtype=torch.int32, device=device
                )
                if self.dcp_world_size > 1:
                    segmented_page_size = _segmented_mla_page_size(
                        self.kernel_block_size
                    )
                    max_local_pages = cdiv(
                        self._dcp_verify_graph_max_kv_seq_len,
                        segmented_page_size,
                    )
                    self._dcp_verify_block_table = torch.zeros(
                        (max_verify_rows, max_local_pages),
                        dtype=torch.int32,
                        device=device,
                    )
                    self._dcp_verify_qo_indptr = torch.arange(
                        max_verify_rows + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    self._verify_row_indptr = torch.zeros(
                        max_verify_rows + 1, dtype=torch.int32, device=device
                    )
                    self._verify_row_page_table = torch.zeros(
                        max_verify_rows * max_num_pages_per_req,
                        dtype=torch.int32,
                        device=device,
                    )

    def _init_fp8_prefill_ps_buffers(
        self,
        max_num_reqs: int,
        max_prefill_qlen: int,
        max_num_batched_tokens: int,
        device: torch.device,
    ) -> None:
        """Pre-allocate persistent buffers for FP8 MLA prefill PS metadata.

        Uses ``get_ps_metadata_info_v1`` with max values so the buffers are
        large enough for any batch.  ``get_ps_metadata_v1`` fills them
        per-batch in ``build()``.  The FP8 prefill forward path also uses the
        global workspace manager for per-call scratch, so reserve its maximum
        shape here before the workspace manager is locked after warmup.

        Args:
            max_num_reqs: Maximum number of concurrent requests.
            max_prefill_qlen: Maximum Q-length for a single request in one
                prefill batch.  Should be ``min(max_model_len,
                max_num_batched_tokens)`` — a single request never exceeds
                ``max_model_len`` tokens, nor the per-batch token budget.
            max_num_batched_tokens: Maximum number of tokens scheduled in one
                batch.  The ``final_lse`` scratch is sized by ``total_q`` (the
                summed Q-length over all prefill requests in the batch), which
                is bounded by this budget rather than by a single request's
                ``max_prefill_qlen`` — concurrent requests can sum to more than
                ``max_model_len`` when ``max_model_len < max_num_batched_tokens``.
            device: Target device for the buffers.
        """
        from aiter import get_ps_metadata_info_v1

        # After kv_b_proj decompression, K has num_heads heads (same as Q).
        # So gqa_ratio=1 and num_head_k=num_heads for the PS kernel.
        num_head_k = self.num_heads
        v_head_dim = self.mla_dims.v_head_dim
        # gqa_ratio = 1
        # qlen_granularity = _FP8_PREFILL_TILE_Q // max(gqa_ratio, 1)
        qlen_granularity = _FP8_PREFILL_TILE_Q

        (
            (work_metadata_size, work_metadata_dtype),
            (work_indptr_size, work_indptr_dtype),
            (work_info_size, work_info_dtype),
            (reduce_indptr_size, reduce_indptr_dtype),
            (reduce_final_map_size, reduce_final_map_dtype),
            (reduce_partial_map_size, reduce_partial_map_dtype),
        ) = get_ps_metadata_info_v1(
            batch_size=max_num_reqs,
            num_head_k=num_head_k,
            max_qlen=max_prefill_qlen,
            qlen_granularity=qlen_granularity,
        )

        self.fp8_ps_work_metadata = torch.empty(
            work_metadata_size, dtype=work_metadata_dtype, device=device
        )
        self.fp8_ps_work_indptr = torch.empty(
            work_indptr_size, dtype=work_indptr_dtype, device=device
        )
        self.fp8_ps_work_info = torch.empty(
            *work_info_size, dtype=work_info_dtype, device=device
        )
        self.fp8_ps_reduce_indptr = torch.empty(
            reduce_indptr_size, dtype=reduce_indptr_dtype, device=device
        )
        self.fp8_ps_reduce_final_map = torch.empty(
            *reduce_final_map_size, dtype=reduce_final_map_dtype, device=device
        )
        self.fp8_ps_reduce_partial_map = torch.empty(
            reduce_partial_map_size,
            dtype=reduce_partial_map_dtype,
            device=device,
        )

        from vllm.v1.worker.workspace import current_workspace_manager

        max_num_partial_tiles = reduce_partial_map_size
        current_workspace_manager().get_simultaneous(
            (
                (max_num_partial_tiles * _FP8_PREFILL_TILE_Q, num_head_k, v_head_dim),
                torch.float32,
            ),
            (
                (max_num_partial_tiles * _FP8_PREFILL_TILE_Q, num_head_k),
                torch.float32,
            ),
            ((max_num_batched_tokens, num_head_k), torch.float32),
        )

        logger.info(
            "FP8 MLA prefill PS buffers allocated "
            "(max_batch=%d, max_qlen=%d, num_head_k=%d)",
            max_num_reqs,
            max_prefill_qlen,
            num_head_k,
        )

    def _build_fp8_prefill_ps_metadata(
        self,
        metadata: AiterMLAMetadata,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> None:
        """Build per-batch FP8 MLA prefill PS metadata and attach to *metadata*.

        Called from ``build()`` when prefill tokens are present and
        FP8 MLA prefill is enabled (auto-detected via
        ``_fp8_mla_prefill_supported()``).
        """
        from aiter import get_ps_metadata_v1

        prefill = metadata.prefill
        # Caller (build()) only invokes this when prefill tokens exist, so
        # metadata.prefill is guaranteed non-None.  Assert to narrow for mypy.
        assert prefill is not None
        qo_indptr = prefill.query_start_loc
        kv_indptr = qo_indptr  # new tokens: KV length == Q length

        # Reuse the existing CPU view of query_start_loc instead of forcing a
        # device->host copy.  Prefill batches sit at the tail of the request
        # list, so we slice from num_decodes onwards and rebase to zero, the
        # same transform the parent build applies on device tensors.
        num_decodes = metadata.num_decodes
        qsl_cpu = common_attn_metadata.query_start_loc_cpu
        qo_indptr_cpu = (qsl_cpu[num_decodes:] - qsl_cpu[num_decodes]).to(torch.int32)
        kv_indptr_cpu = qo_indptr_cpu.clone()
        seq_lens_cpu = (qo_indptr_cpu[1:] - qo_indptr_cpu[:-1]).to(torch.int32)

        num_head_k = self.num_heads
        # gqa_ratio = 1
        # qhead_granularity = max(gqa_ratio, 1)
        # qlen_granularity = _FP8_PREFILL_TILE_Q // qhead_granularity
        gqa_ratio = 1
        qhead_granularity = 1
        qlen_granularity = _FP8_PREFILL_TILE_Q
        kvlen_granularity = 128
        block_size = 1  # non-paged: each "page" is one token

        get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_cpu,
            gqa_ratio,
            num_head_k,
            self.fp8_ps_work_metadata,
            self.fp8_ps_work_indptr,
            self.fp8_ps_work_info,
            self.fp8_ps_reduce_indptr,
            self.fp8_ps_reduce_final_map,
            self.fp8_ps_reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=True,
        )

        total_prefill_tokens = int(qo_indptr_cpu[-1].item())
        kv_indices = torch.arange(
            total_prefill_tokens, device=qo_indptr.device, dtype=torch.int32
        )

        # The actual number of active partial tiles for this batch is the
        # final value of reduce_indptr.  Resolving it here (during metadata
        # build) keeps it off the per-layer forward path where a sync would
        # break CUDA Graph capture.  Using the device-side reduce_indptr is
        # acceptable since build is allowed to incur an occasional sync.
        num_partial_tiles = int(self.fp8_ps_reduce_indptr[-1].item())

        # Attach PS metadata to the metadata object so forward_mha can read it.
        metadata.fp8_prefill_qo_indptr = qo_indptr
        metadata.fp8_prefill_kv_indptr = kv_indptr
        metadata.fp8_prefill_kv_indices = kv_indices
        metadata.fp8_prefill_work_indptr = self.fp8_ps_work_indptr
        metadata.fp8_prefill_work_info_set = self.fp8_ps_work_info
        metadata.fp8_prefill_reduce_indptr = self.fp8_ps_reduce_indptr
        metadata.fp8_prefill_reduce_final_map = self.fp8_ps_reduce_final_map
        metadata.fp8_prefill_reduce_partial_map = self.fp8_ps_reduce_partial_map
        metadata.fp8_prefill_max_q_len = prefill.max_query_len
        metadata.fp8_prefill_num_partial_tiles = num_partial_tiles

    def _build_verify_row_view(
        self,
        qlen: int,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Flatten a non-DCP verify into one causal paged-KV row per token."""
        assert self.dcp_world_size == 1
        num_reqs = paged_kv_indptr.numel() - 1
        device = paged_kv_indptr.device
        seq_lens = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        offsets = torch.arange(qlen, device=device, dtype=seq_lens.dtype)
        row_lens = (
            (seq_lens.unsqueeze(1) - (qlen - 1) + offsets).clamp_min_(0).flatten()
        )
        row_indptr = torch.cat([paged_kv_indptr.new_zeros(1), row_lens.cumsum(0)]).to(
            torch.int32
        )
        total_pages = int(row_indptr[-1].item())
        row_starts = row_indptr[:-1].to(torch.int64).repeat_interleave(row_lens)
        within_row = (
            torch.arange(total_pages, device=device, dtype=torch.int64) - row_starts
        )
        row_req = torch.arange(num_reqs, device=device).repeat_interleave(qlen)
        src = paged_kv_indptr[row_req].to(torch.int64).repeat_interleave(row_lens)
        row_page_table = paged_kv_indices[src + within_row]

        if self._verify_row_page_table is not None:
            assert self._verify_row_indptr is not None
            assert self._verify_row_lens is not None
            assert total_pages <= self._verify_row_page_table.numel(), (
                f"the verify's per-row view needs {total_pages} pages but the "
                f"cudagraph buffer holds {self._verify_row_page_table.numel()}"
            )
            num_rows = row_lens.numel()
            self._verify_row_lens[:num_rows].copy_(row_lens, non_blocking=True)
            self._verify_row_indptr[: num_rows + 1].copy_(row_indptr, non_blocking=True)
            self._verify_row_page_table[:total_pages].copy_(
                row_page_table, non_blocking=True
            )
            row_lens = self._verify_row_lens[:num_rows]
            row_indptr = self._verify_row_indptr[: num_rows + 1]
            # The whole buffer, not a slice: a captured batch fixes the row count
            # but not the page count, and the kernel reads the table through
            # row_indptr rather than through its length.
            row_page_table = self._verify_row_page_table

        # Captured split counts must remain valid when replayed with shorter rows.
        min_row_kv_len = (
            1
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            else int(row_lens.min())
        )
        return row_indptr, row_page_table, row_lens, min_row_kv_len

    def _build_dcp_verify_row_view(
        self,
        qlen: int,
        block_table: torch.Tensor,
        dcp_tot_seq_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Build committed-shard rows for segmented DCP verification."""
        assert self.dcp_world_size > 1
        num_reqs = dcp_tot_seq_lens.numel()
        committed = (dcp_tot_seq_lens - qlen).clamp(min=0)
        per_req_lens = get_dcp_local_seq_lens(
            committed,
            self.dcp_world_size,
            self.dcp_rank,
            self.cp_kv_cache_interleave_size,
        )
        row_lens = per_req_lens.repeat_interleave(qlen)
        num_rows = row_lens.numel()
        max_kv_seq_len = (
            self._dcp_verify_graph_max_kv_seq_len
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs()
            else max(1, int(row_lens.max().item()))
        )
        page_size = _segmented_mla_page_size(self.kernel_block_size)
        pages_per_block = self.kernel_block_size // page_size
        max_local_pages = cdiv(max_kv_seq_len, page_size)
        if self._dcp_verify_block_table is not None:
            assert self._dcp_verify_qo_indptr is not None
            assert self._verify_row_lens is not None
            assert max_local_pages <= self._dcp_verify_block_table.shape[1]
            row_block_table = self._dcp_verify_block_table[:num_rows]
            self._verify_row_lens[:num_rows].copy_(row_lens, non_blocking=True)
            row_lens = self._verify_row_lens[:num_rows]
            qo_indptr = self._dcp_verify_qo_indptr[: num_rows + 1]
        else:
            row_block_table = torch.empty(
                (num_rows, max_local_pages),
                dtype=torch.int32,
                device=block_table.device,
            )
            qo_indptr = torch.arange(
                num_rows + 1,
                dtype=torch.int32,
                device=block_table.device,
            )
        self._fill_dcp_verify_page_table(
            row_block_table,
            block_table,
            num_reqs,
            qlen,
            pages_per_block,
            max_local_pages,
        )
        return row_block_table, row_lens, qo_indptr, max_kv_seq_len

    def _fill_dcp_verify_page_table(
        self,
        row_block_table: torch.Tensor,
        block_table: torch.Tensor,
        num_reqs: int,
        qlen: int,
        pages_per_block: int,
        max_local_pages: int,
    ) -> None:
        if block_table.is_cuda:
            _expand_dcp_verify_subpages_kernel[(num_reqs,)](
                row_block_table,
                block_table,
                row_block_table.stride(0),
                block_table.stride(0),
                qlen=qlen,
                pages_per_block=pages_per_block,
                max_local_pages=max_local_pages,
                BLOCK=64,
            )
            return
        max_local_blocks = cdiv(max_local_pages, pages_per_block)
        subpage_offsets = torch.arange(
            pages_per_block,
            dtype=block_table.dtype,
            device=block_table.device,
        )
        per_req_page_table = (
            block_table[:num_reqs, :max_local_blocks, None] * pages_per_block
            + subpage_offsets
        ).flatten(1)[:, :max_local_pages]
        row_block_table.copy_(per_req_page_table.repeat_interleave(qlen, dim=0))

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> AiterMLADecodeMetadata:
        device = self.device
        num_reqs = seq_lens_device.size(0)
        qo_len = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_qo_len = qo_len.max().item()
        padded_mtp_qo_len = self._uniform_padded_mtp_qo_len(
            qo_len, max_qo_len, num_decode_tokens
        )
        if padded_mtp_qo_len > 0:
            max_qo_len = padded_mtp_qo_len
        pad_uniform_mtp = padded_mtp_qo_len > 0

        seq_lens_for_kernel = seq_lens_device
        if pad_uniform_mtp:
            qo_lens_device = (
                query_start_loc_device[1 : num_reqs + 1]
                - query_start_loc_device[:num_reqs]
            ).to(torch.int32)
            seq_lens_for_kernel = torch.where(
                qo_lens_device > 0,
                seq_lens_for_kernel,
                seq_lens_for_kernel.new_full((), max_qo_len),
            )

        if self._graph_seq_lens is not None:
            self._graph_seq_lens[:num_reqs].copy_(
                seq_lens_for_kernel, non_blocking=True
            )
            seq_lens_for_kernel = self._graph_seq_lens[:num_reqs]

        # The aiter kernel always operates with page_size=1 (the wrapper
        # flattens kv_buffer). last_page_len is always 1.
        paged_kv_last_page_len = self.paged_kv_last_page_len[:num_reqs]

        # indptr: cumsum of seq_lens (one page per token in the flat view)
        paged_kv_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                seq_lens_for_kernel.cumsum(dim=0, dtype=torch.int32),
            ]
        )
        use_gluon_decode = AiterMLAHelper.use_gluon_decode(
            self._decode_num_heads,
            int(max_qo_len),
            self._kv_cache_dtype_str,
            self.dcp_world_size,
        )
        use_gluon_verify = AiterMLAHelper.use_gluon_verify(
            self._decode_num_heads,
            int(max_qo_len),
            self._kv_cache_dtype_str,
            self.dcp_world_size,
        )
        skip_paged_kv_expand = use_gluon_verify and self.dcp_world_size > 1

        if not skip_paged_kv_expand:
            if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
                self.paged_kv_indices.fill_(-1)

            # Expand block_table entries into per-token flat indices.
            # When kernel_block_size=1, this degrades to a direct copy (identical
            # to the original _copy_page_indices_kernel).
            # When kernel_block_size=K>1, block_table entry b covering K tokens
            # gets expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
            _expand_page_indices_kernel[(num_reqs,)](
                self.paged_kv_indices,
                block_table_tensor,
                block_table_tensor.stride(0),
                paged_kv_indptr,
                seq_lens_for_kernel,
                KERNEL_BLOCK_SIZE=self.kernel_block_size,
                BLOCK_SIZE=1024,
            )
        paged_kv_indices = self.paged_kv_indices

        if self.compilation_config.cudagraph_mode.has_full_cudagraphs():
            self.paged_kv_indptr[: 1 + num_reqs].copy_(
                paged_kv_indptr, non_blocking=True
            )
            self.paged_kv_indptr[1 + num_reqs :].fill_(paged_kv_indptr[-1])
            paged_kv_indptr = self.paged_kv_indptr[: 1 + num_reqs]

            if pad_uniform_mtp:
                qo_indptr_src = AiterMLAHelper.qo_indptr_for_uniform_qlen(
                    num_reqs, int(max_qo_len), device
                )
            else:
                qo_indptr_src = query_start_loc_device[: 1 + num_reqs]
            self.qo_indptr[: 1 + num_reqs].copy_(qo_indptr_src, non_blocking=True)
            self.qo_indptr[1 + num_reqs :] = qo_indptr_src[-1]
            qo_indptr = self.qo_indptr[: 1 + num_reqs]

        else:
            if max_qo_len == 1:
                qo_indptr = AiterMLAHelper.qo_indptr_for_uniform_qlen(
                    num_reqs, 1, device
                )
            else:
                if pad_uniform_mtp:
                    qo_indptr = AiterMLAHelper.qo_indptr_for_uniform_qlen(
                        num_reqs, int(max_qo_len), device
                    )
                else:
                    qo_indptr = query_start_loc_device[: 1 + num_reqs]

        # Only ASM routes consume persistent scheduling metadata.
        has_persistent_metadata = False
        # Uniform verify with qlen in {2,3} can ride the qlen=4 ASM LSE-PS
        # entry via leading pad rows (Gluon verify skips this path).
        kernel_max_qo_len = max_qo_len
        asm_kernel_qlen = AiterMLAHelper.asm_lse_ps_kernel_qlen(int(max_qo_len))
        can_asm_qlen_pad = (
            not use_gluon_decode
            and not use_gluon_verify
            and asm_kernel_qlen is not None
            and asm_kernel_qlen > max_qo_len
            and (pad_uniform_mtp or torch.all(qo_len == max_qo_len))
        )
        if can_asm_qlen_pad:
            assert asm_kernel_qlen is not None
            kernel_max_qo_len = asm_kernel_qlen
            qo_indptr = AiterMLAHelper.qo_indptr_for_uniform_qlen(
                num_reqs, kernel_max_qo_len, device
            )
        use_persistent_metadata = (
            not use_gluon_decode
            and not use_gluon_verify
            # A padded rank has no bf16 persistent kernel past qlen 4 where the
            # gfx950 fold is absent; the non-persistent entry covers it. fp8
            # keeps the schedule -- its fold rejects non-persistent outright.
            and (
                self._decode_num_heads >= AiterMLAHelper._AITER_MIN_MLA_HEADS
                or kernel_max_qo_len <= AiterMLAHelper._ASM_PADDED_MAX_PS_QLEN
                or is_quantized_kv_cache(self._kv_cache_dtype_str)
            )
            and max_qo_len >= 1
            and max_qo_len <= self._mtp_decode_qlen
        )
        if use_persistent_metadata:
            from aiter import get_mla_metadata_v1

            uni_qo_len = (
                kernel_max_qo_len
                if can_asm_qlen_pad
                or pad_uniform_mtp
                or torch.all(qo_len == max_qo_len)
                else -1
            )
            get_mla_metadata_v1(
                qo_indptr,
                paged_kv_indptr,
                paged_kv_last_page_len,
                self._num_attention_heads,
                1,
                True,
                self._mla_work_meta_data,
                self._mla_work_info_set,
                self._mla_work_indptr,
                self._mla_reduce_indptr,
                self._mla_reduce_final_map,
                self._mla_reduce_partial_map,
                page_size=1,
                kv_granularity=16,
                max_seqlen_qo=kernel_max_qo_len,
                uni_seqlen_qo=uni_qo_len,
                fast_mode=True,
                dtype_q=self._mla_q_dtype,
                dtype_kv=self._mla_kv_dtype,
            )
            has_persistent_metadata = True

        row_indptr = row_page_table = row_lens = None
        min_row_kv_len = 1
        dcp_verify_block_table = dcp_verify_qo_indptr = None
        dcp_verify_max_kv_seq_len = 1
        if use_gluon_verify:
            if self.dcp_world_size > 1:
                assert dcp_tot_seq_lens_device is not None
                (
                    dcp_verify_block_table,
                    row_lens,
                    dcp_verify_qo_indptr,
                    dcp_verify_max_kv_seq_len,
                ) = self._build_dcp_verify_row_view(
                    int(max_qo_len),
                    block_table_tensor,
                    dcp_tot_seq_lens_device,
                )
            else:
                (
                    row_indptr,
                    row_page_table,
                    row_lens,
                    min_row_kv_len,
                ) = self._build_verify_row_view(
                    int(max_qo_len),
                    paged_kv_indptr,
                    paged_kv_indices,
                )

        attn_metadata = AiterMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_for_kernel,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            qo_indptr=qo_indptr,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
            max_qo_len=max_qo_len,
            kernel_max_qo_len=(
                kernel_max_qo_len if kernel_max_qo_len != max_qo_len else None
            ),
            verify_row_indptr=row_indptr,
            verify_row_page_table=row_page_table,
            verify_row_lens=row_lens,
            verify_min_kv_seq_len=min_row_kv_len,
            dcp_verify_block_table=dcp_verify_block_table,
            dcp_verify_qo_indptr=dcp_verify_qo_indptr,
            dcp_verify_max_kv_seq_len=dcp_verify_max_kv_seq_len,
            use_gluon_decode=use_gluon_decode,
            attn_out_dtype=self.decode_attn_out_dtype,
            has_persistent_metadata=has_persistent_metadata,
        )

        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> AiterMLAMetadata:
        attn_metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )
        if (
            attn_metadata.decode is not None
            and attn_metadata.decode.has_persistent_metadata
        ):
            attn_metadata.work_meta_data = self._mla_work_meta_data
            attn_metadata.work_indptr = self._mla_work_indptr
            attn_metadata.work_info_set = self._mla_work_info_set
            attn_metadata.reduce_indptr = self._mla_reduce_indptr
            attn_metadata.reduce_final_map = self._mla_reduce_final_map
            attn_metadata.reduce_partial_map = self._mla_reduce_partial_map
        if (
            self._fp8_prefill_enabled
            and attn_metadata.prefill is not None
            and attn_metadata.prefill.chunked_context is None
        ):
            self._build_fp8_prefill_ps_metadata(attn_metadata, common_attn_metadata)
        return attn_metadata


@triton.jit
def _expand_page_indices_kernel(
    page_indices,
    block_table,
    block_table_stride,
    cu_num_tokens,
    seq_lens,
    KERNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Expand block table entries into per-token flat page indices.

    The aiter MLA kernel always operates with page_size=1 internally
    (kv_buffer is flattened via .view(-1, 1, 1, H)). This kernel converts
    block-level indices from the block table into individual token positions
    in the flattened KV buffer.

    When KERNEL_BLOCK_SIZE=1: block_idx=t, offset=0, flat=block_id
    (equivalent to a direct copy -- no regression from the original kernel).

    When KERNEL_BLOCK_SIZE=K: block table entry b (covering K tokens)
    is expanded to flat indices b*K, b*K+1, ..., b*K+(K-1).
    """
    req_idx = tl.program_id(0)
    row_ptr = block_table + req_idx * block_table_stride
    start_idx = tl.load(cu_num_tokens + req_idx)
    num_tokens = tl.load(seq_lens + req_idx)

    offset = tl.arange(0, BLOCK_SIZE)
    for i in tl.range(0, num_tokens, BLOCK_SIZE):
        token_offsets = i + offset
        mask = token_offsets < num_tokens

        # Which block in the block table does this token belong to?
        block_idx = token_offsets // KERNEL_BLOCK_SIZE
        # Offset within that block
        offset_in_block = token_offsets % KERNEL_BLOCK_SIZE

        # Load the block ID from the block table
        block_ids = tl.load(row_ptr + block_idx, mask=mask)

        # Compute flat index in the flattened kv_buffer
        flat_indices = block_ids * KERNEL_BLOCK_SIZE + offset_in_block

        tl.store(
            page_indices + start_idx + token_offsets,
            flat_indices,
            mask=mask,
        )


@triton.jit
def _expand_dcp_verify_subpages_kernel(
    out_ptr,
    block_table_ptr,
    out_stride0,
    block_table_stride,
    qlen: tl.constexpr,
    pages_per_block: tl.constexpr,
    max_local_pages: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write 128-token subpage IDs into the DCP verify block table."""
    req_idx = tl.program_id(0)
    page_off = tl.arange(0, BLOCK)
    for start in tl.range(0, max_local_pages, BLOCK):
        pages = start + page_off
        mask = pages < max_local_pages
        block_idx = pages // pages_per_block
        sub = pages % pages_per_block
        block_ids = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_idx,
            mask=mask,
            other=0,
        )
        page_ids = block_ids * pages_per_block + sub
        for q in tl.static_range(qlen):
            tl.store(
                out_ptr + (req_idx * qlen + q) * out_stride0 + pages,
                page_ids,
                mask=mask,
            )


class AiterMLAHelper:
    """
    AITER MLA persistent (asm) decode requires a multiple of 16 heads. Unaligned
    head counts through 128 are padded to the next multiple of 16 by tiling the
    query heads and slicing to the padded size. Native H24 AITER builds bypass
    that padding. Small divisors of 16 retain the existing repeat_interleave and
    strided-unpad behavior. Native and aligned counts pass through without
    copies.
    """

    _AITER_MIN_MLA_HEADS: Final = 16
    _AITER_MAX_PADDED_MLA_HEADS: Final = 128
    # Largest qlen the padded gqa=16 asm decode has a bf16 persistent kernel
    # for. Above it only the non-persistent qseqlen=8 entry exists, and the
    # fold that reaches a persistent one is gfx950-only.
    _ASM_PADDED_MAX_PS_QLEN: Final = 4
    # bf16 gqa=16 ASM PS exposes return_lse only for qseqlen in {1, 4}.
    # Uniform qlen in {2, 3} pads up to 4 with leading dummy rows so each
    # real row keeps its causal window (ASM windows are right-aligned).
    _ASM_LSE_PS_PAD_QLEN: Final = 4
    _AITER_UNSUPPORTED_HEADS: ClassVar[tuple[int, ...]] = ()

    @staticmethod
    def asm_lse_ps_kernel_qlen(logical_qlen: int) -> int | None:
        """Map logical qlen to an LSE-capable ASM PS ``max_seqlen_qo``.

        ``1``/``4`` are native; ``2``/``3`` pad to ``4``; otherwise ``None``.
        """
        if logical_qlen == 1:
            return 1
        if 1 < logical_qlen <= AiterMLAHelper._ASM_LSE_PS_PAD_QLEN:
            return AiterMLAHelper._ASM_LSE_PS_PAD_QLEN
        return None

    @staticmethod
    def pad_uniform_q_to_kernel_qlen(
        q: torch.Tensor,
        logical_qlen: int,
        kernel_qlen: int,
    ) -> torch.Tensor:
        """Prepend per-request zero rows so ``[R*logical, H, D]`` becomes
        ``[R*kernel, H, D]``. Leading pads preserve ASM right-aligned causal
        windows; callers must widen ``qo_indptr`` and
        :meth:`unpad_uniform_rows` on output/LSE.
        """
        if kernel_qlen == logical_qlen:
            return q
        if kernel_qlen < logical_qlen:
            raise ValueError(
                f"kernel_qlen ({kernel_qlen}) < logical_qlen ({logical_qlen})"
            )
        if q.shape[0] % logical_qlen != 0:
            raise ValueError(
                f"Q rows {q.shape[0]} not divisible by logical_qlen={logical_qlen}"
            )
        num_reqs = q.shape[0] // logical_qlen
        pad = kernel_qlen - logical_qlen
        q_blk = q.view(num_reqs, logical_qlen, *q.shape[1:])
        zeros = q.new_zeros((num_reqs, pad, *q.shape[1:]))
        return torch.cat([zeros, q_blk], dim=1).reshape(
            num_reqs * kernel_qlen, *q.shape[1:]
        )

    @staticmethod
    def unpad_uniform_rows(
        x: torch.Tensor,
        logical_qlen: int,
        kernel_qlen: int,
    ) -> torch.Tensor:
        """Drop leading pad rows from :meth:`pad_uniform_q_to_kernel_qlen`."""
        if kernel_qlen == logical_qlen:
            return x
        if x.shape[0] % kernel_qlen != 0:
            raise ValueError(
                f"rows {x.shape[0]} not divisible by kernel_qlen={kernel_qlen}"
            )
        num_reqs = x.shape[0] // kernel_qlen
        return (
            x.view(num_reqs, kernel_qlen, *x.shape[1:])[:, kernel_qlen - logical_qlen :]
            .reshape(num_reqs * logical_qlen, *x.shape[1:])
            .contiguous()
        )

    @staticmethod
    def qo_indptr_for_uniform_qlen(
        num_reqs: int,
        qlen: int,
        device: torch.device,
        dtype: torch.dtype = torch.int32,
    ) -> torch.Tensor:
        """Build ``[0, qlen, 2*qlen, ..., num_reqs*qlen]``."""
        return torch.arange(
            0,
            (num_reqs + 1) * qlen,
            step=qlen,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def check_num_heads_validity(num_heads: int):
        assert AiterMLAHelper.is_valid_num_heads(num_heads), (
            "ROCM AITER MLA requires a positive multiple of 16 heads, or an "
            "unaligned head count up to 128 (padded to the next multiple of "
            f"16), but got {num_heads}.\n"
            f"Try adjusting tensor_parallel_size value."
        )

    @staticmethod
    def is_valid_num_heads(num_heads: int) -> bool:
        return (
            num_heads > 0
            and num_heads not in AiterMLAHelper._AITER_UNSUPPORTED_HEADS
            and (
                num_heads <= AiterMLAHelper._AITER_MAX_PADDED_MLA_HEADS
                or num_heads % AiterMLAHelper._AITER_MIN_MLA_HEADS == 0
            )
        )

    @staticmethod
    def get_actual_mla_num_heads(num_heads: int) -> int:
        if num_heads == 24 and _aiter_mla_native_h24_supported():
            return num_heads
        m = AiterMLAHelper._AITER_MIN_MLA_HEADS
        return -(-num_heads // m) * m

    @staticmethod
    def get_mla_padded_q(num_heads: int, q: torch.Tensor) -> torch.Tensor:
        m = AiterMLAHelper.get_actual_mla_num_heads(num_heads)
        if num_heads == m:
            return q
        if m % num_heads == 0:
            return q.repeat_interleave(m // num_heads, dim=1)
        # Non-divisor head counts cannot be padded by repeat_interleave. Tile
        # the query heads and slice to exactly m. MLA attention is independent
        # per query head over the shared KV, so padding heads cannot affect
        # heads [0:num_heads]; they are sliced back off the output.
        reps = -(-m // num_heads)  # ceil(m / num_heads)
        # Slicing a tiled tensor yields a non-contiguous view. The asm decode
        # reads q as packed [tokens, m, head_dim], so materialize it.
        return q.repeat(1, reps, 1)[:, :m, :].contiguous()

    @staticmethod
    def get_mla_unpadded_o(num_heads: int, o: torch.Tensor) -> torch.Tensor:
        return AiterMLAHelper._get_mla_unpadded_heads(num_heads, o)

    @staticmethod
    def _get_mla_unpadded_heads(num_heads: int, tensor: torch.Tensor) -> torch.Tensor:
        m = AiterMLAHelper.get_actual_mla_num_heads(num_heads)
        if num_heads == m:
            return tensor
        if m % num_heads == 0:
            return tensor[:, :: m // num_heads, ...]
        return tensor[:, :num_heads, ...]

    @staticmethod
    def get_mla_unpadded_lse(num_heads: int, lse: torch.Tensor) -> torch.Tensor:
        return AiterMLAHelper._get_mla_unpadded_heads(num_heads, lse)

    @staticmethod
    def _gluon_max_heads(dcp_world_size: int) -> int:
        if dcp_world_size > 1:
            return _gluon_mla_max_bh16_heads()
        return AiterMLAHelper._AITER_MIN_MLA_HEADS - 1

    @staticmethod
    def use_gluon_decode(
        num_heads: int, max_qo_len: int, kv_cache_dtype: str, dcp_world_size: int = 1
    ) -> bool:
        if max_qo_len != 1:
            return False
        if num_heads > AiterMLAHelper._gluon_max_heads(dcp_world_size):
            return False
        # The available FP8 Gluon kernel supports only batch size one.
        if is_quantized_kv_cache(kv_cache_dtype):
            return False
        mode = _aiter_mla_small_head_mode()
        if mode == "asm":
            return False
        gluon_supported = _gluon_mla_decode_supported()
        if mode == "gluon":
            return gluon_supported
        return AiterMLAHelper._AITER_MIN_MLA_HEADS % num_heads == 0 and gluon_supported

    @staticmethod
    def use_gluon_verify(
        num_heads: int,
        max_qo_len: int,
        kv_cache_dtype: str,
        dcp_world_size: int = 1,
    ) -> bool:
        """Whether multi-token verification uses a decode-kernel path."""
        if max_qo_len <= 1:
            return False
        if dcp_world_size > 1:
            return _segmented_mla_decode_supported()
        if is_quantized_kv_cache(kv_cache_dtype):
            return False
        if num_heads > AiterMLAHelper._gluon_max_heads(dcp_world_size):
            return False
        if not _gluon_mla_decode_supported():
            return False
        return _aiter_mla_small_head_mode() != "asm"


class AiterMLAImpl(MLACommonImpl[AiterMLAMetadata]):
    # Both decode paths can hand back a natural-log softmax LSE for the DCP
    # cross-rank merge: Gluon takes return_lse, and the asm kernel returns one
    # through aiter's native entry point (the vLLM custom-op wrapper drops it).
    can_return_lse_for_decode: bool = True
    supports_dcp_verify_window: ClassVar[bool] = True

    @property
    def _decode_num_heads(self) -> int:
        """Return the query-head count after DCP gathering."""
        return self.num_heads * self.dcp_world_size

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )
        AiterMLAHelper.check_num_heads_validity(num_heads)
        AiterMLAHelper.check_num_heads_validity(self._decode_num_heads)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "Aiter MLA does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        from aiter import flash_attn_varlen_func

        self.flash_attn_varlen_func = flash_attn_varlen_func

        # FP8 MLA prefill kernel imports (lazy, only when enabled).
        # Auto-enabled on gfx950 when AITER ships the kernels.
        # FP8 MLA prefill (kn_mla_reduce_v1) only supports 16-aligned heads.
        self._fp8_prefill_enabled = (
            _fp8_mla_prefill_supported() and self.num_heads % 16 == 0
        )
        if self._fp8_prefill_enabled:
            from aiter import mla_prefill_ps_asm_fwd, mla_reduce_v1

            self._mla_prefill_ps_asm_fwd = mla_prefill_ps_asm_fwd
            self._mla_reduce_v1 = mla_reduce_v1

    def _flash_attn_varlen_diff_headdims(
        self, q, k, v, return_softmax_lse=False, softmax_scale=None, **kwargs
    ):
        output = self.flash_attn_varlen_func(  # type: ignore[call-arg]
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            return_lse=return_softmax_lse,
            **kwargs,
        )

        return output

    def _mla_fp8_prefill_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        out: torch.Tensor,
    ) -> None:
        """Run FP8 MLA prefill via mla_prefill_ps_asm_fwd + mla_reduce_v1.

        Q, K, V are already decompressed (post-kv_b_proj), so K and V have
        ``num_heads`` heads (same as Q) and gqa_ratio=1.  Writes the
        result in-place to ``out``, which is the [total_q, nhead * v_head_dim]
        output buffer supplied by ``forward_mha``; no extra allocation or
        copy is required.
        """
        from vllm.platforms import current_platform
        from vllm.v1.worker.workspace import current_workspace_manager

        fp8_dtype = current_platform.fp8_dtype()
        total_q = q.shape[0]
        nhead = self.num_heads
        v_head_dim = self.v_head_dim
        tile_q = _FP8_PREFILL_TILE_Q

        # The FP8 ASM kernel expects FP8 inputs; the q_scale/k_scale/v_scale
        # parameters select per-tensor dequant scales.  Q/K/V arrive as
        # bf16 from kv_b_proj, so cast here (one_scale=1.0 disables scaling).
        if q.dtype != fp8_dtype:
            q = q.to(fp8_dtype)
        if k.dtype != fp8_dtype:
            k = k.to(fp8_dtype)
        if v.dtype != fp8_dtype:
            v = v.to(fp8_dtype)

        one_scale = torch.ones((), dtype=torch.float32, device=q.device)

        # num_partial_tiles is resolved during metadata build to avoid an
        # in-forward .item() sync that would prevent CUDA Graph capture.
        # forward_mha gates the FP8 path on fp8_prefill_qo_indptr being set,
        # and the builder always sets every fp8_prefill_* field together, so
        # num_partial_tiles is non-None here.
        num_partial_tiles = attn_metadata.fp8_prefill_num_partial_tiles
        assert num_partial_tiles is not None

        # Reuse the caller's output buffer to skip the per-call alloc + copy.
        # The ASM and reduce kernels both write to a [total_q, nhead, v_head_dim]
        # view, which aliases the [total_q, nhead * v_head_dim] storage of out.
        out_3d = out.view(total_q, nhead, v_head_dim)

        # Per-call scratch (logits, attn_lse, final_lse) is served from the
        # workspace manager so allocator churn in the prefill hot path is
        # bounded after warmup, matching the pattern in PR #41002.
        logits, attn_lse, final_lse = current_workspace_manager().get_simultaneous(
            ((num_partial_tiles * tile_q, nhead, v_head_dim), torch.float32),
            ((num_partial_tiles * tile_q, nhead), torch.float32),
            ((total_q, nhead), torch.float32),
        )

        # Phase 1: persistent-scheduling assembly prefill kernel.
        self._mla_prefill_ps_asm_fwd(
            q,
            k,
            v,
            attn_metadata.fp8_prefill_qo_indptr,
            attn_metadata.fp8_prefill_kv_indptr,
            attn_metadata.fp8_prefill_kv_indices,
            attn_metadata.fp8_prefill_work_indptr,
            attn_metadata.fp8_prefill_work_info_set,
            attn_metadata.fp8_prefill_max_q_len,
            self.scale,
            True,  # is_causal
            logits,
            attn_lse,
            out_3d,
            one_scale,
            one_scale,
            one_scale,
        )

        # Phase 2: reduction across KV splits.
        self._mla_reduce_v1(
            logits,
            attn_lse,
            attn_metadata.fp8_prefill_reduce_indptr,
            attn_metadata.fp8_prefill_reduce_final_map,
            attn_metadata.fp8_prefill_reduce_partial_map,
            tile_q,
            # num_kv_splits added by ROCm/aiter#3391; 0 selects the kernel
            # default max(cu_num, 0) == cu_num, matching pre-#3391 behavior.
            0,
            out_3d,
            final_lse,
        )

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        """Dispatch prefill to the FP8 ASM kernel when available.

        Falls back to the parent (``flash_attn_varlen_func``) when FP8
        MLA prefill is disabled, PS metadata is missing, or chunked
        context requires two-pass merge.

        The annotation uses the base ``MLACommonMetadata`` to honour LSP
        with ``MLACommonImpl.forward_mha``; the AITER builder always
        produces ``AiterMLAMetadata`` instances at runtime, so we narrow
        with ``isinstance`` before reading the AITER-specific FP8 fields.
        """
        if (
            not self._fp8_prefill_enabled
            or not isinstance(attn_metadata, AiterMLAMetadata)
            or attn_metadata.fp8_prefill_qo_indptr is None
        ):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        has_context = prefill_metadata.chunked_context is not None

        if has_context:
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                attn_metadata,
                k_scale,
                output,
                output_scale,
            )

        assert output_scale is None, (
            "fused FP8 output not supported by the AITER FP8 MLA prefill path"
        )

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        self._mla_fp8_prefill_attn(q, k, v, attn_metadata, output)

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_mqa(
            q, kv_c_and_k_pe_cache, attn_metadata, layer, k_window=None
        )

    def _merge_rank0_in_hand_window(
        self,
        o: torch.Tensor,
        lse: torch.Tensor,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_window: torch.Tensor | None,
        qlen: int,
        out_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k_window is None or self.dcp_rank != 0:
            return o, lse
        out_b, lse_b = _dense_causal_mla_attn(
            q_nope,
            q_pe,
            k_window,
            self.scale,
            qlen,
            self.kv_lora_rank,
        )
        return _lse_combine_natural(o, lse, out_b, lse_b, out_dtype)

    def _forward_segmented_dcp_verify(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        row_lens: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        decode: AiterMLADecodeMetadata,
        layer: AttentionLayer,
        k_window: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run segmented shard attention and merge the rank-0 causal window."""
        assert decode.dcp_verify_block_table is not None
        assert decode.dcp_verify_qo_indptr is not None
        assert decode.max_qo_len is not None
        q_mla = torch.cat([q_nope, q_pe], dim=-1)
        page_size = _segmented_mla_page_size(kv_c_and_k_pe_cache.shape[1])
        # skip_reduce with NUM_SEGMENTS>1 does not write `out`; reuse q_mla as
        # the unused pointer so we do not allocate a second kv_lora_rank buffer.
        segment_partials = _get_segmented_mla_decode()(
            q_mla,
            kv_c_and_k_pe_cache.view(
                -1,
                page_size,
                1,
                kv_c_and_k_pe_cache.shape[-1],
            ),
            q_mla,
            decode.dcp_verify_qo_indptr,
            row_lens,
            decode.dcp_verify_max_kv_seq_len,
            decode.dcp_verify_block_table,
            self.scale,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            causal=True,
            q_descale=None,
            kv_descale=layer._k_scale,
            skip_reduce=True,
        )
        assert isinstance(segment_partials, tuple) and len(segment_partials) == 3, (
            "AITER segmented MLA verify must return segment partials "
            "when skip_reduce=True."
        )
        segm_output, segm_max, segm_expsum = segment_partials
        o, lse = reduce_mla_segment_partials(
            segm_output,
            segm_max,
            segm_expsum,
            row_lens,
            page_size,
            decode.attn_out_dtype,
        )
        return self._merge_rank0_in_hand_window(
            o,
            lse,
            q_nope,
            q_pe,
            k_window,
            int(decode.max_qo_len),
            decode.attn_out_dtype,
        )

    def forward_mqa_with_dcp_verify_window(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
        k_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self._forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer, k_window)

    def _forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: AiterMLAMetadata,
        layer: AttentionLayer,
        k_window: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        decode = attn_metadata.decode
        assert decode.max_qo_len is not None
        assert decode.paged_kv_indptr is not None
        assert decode.paged_kv_indices is not None
        if decode.use_gluon_decode:
            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            B, num_q_heads, _ = q_nope.shape
            o = torch.empty(
                B,
                num_q_heads,
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )
            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            mla_gluon = _get_mla_gluon()
            need_lse = self.dcp_world_size > 1
            gluon_ret = mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_buffer,
                o=o,
                page_table=decode.paged_kv_indices,
                seq_info=decode.paged_kv_indptr,
                sm_scale=self.scale,
                k_pe=None,
                kv_pe_offset=self.kv_lora_rank,
                use_2d_view=False,
                kv_scale=1.0,
                min_kv_seq_len=decode.min_kv_seq_len,
                return_lse=need_lse,
            )
            lse = gluon_ret[1] if isinstance(gluon_ret, tuple) else None
            if need_lse:
                assert lse is not None, (
                    "aiter mla_gluon(return_lse=True) returned no LSE; upgrade aiter "
                    "to a build with gluon LSE support."
                )
                lse = lse.reshape(B, num_q_heads)
            return o, lse

        # Each verification row gets its own causal KV view.
        if AiterMLAHelper.use_gluon_verify(
            self._decode_num_heads,
            int(decode.max_qo_len),
            self.kv_cache_dtype,
            self.dcp_world_size,
        ):
            row_lens = decode.verify_row_lens
            assert row_lens is not None, (
                "the verify's per-row paged-KV view is missing; the builder and "
                "the impl disagree on whether the Gluon flatten runs."
            )
            if type(q) is tuple:
                q_nope, q_pe = q
            else:
                q_nope, q_pe = torch.split(
                    q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                )
            B, num_q_heads, _ = q_nope.shape
            assert row_lens.numel() == B, (
                f"the verify has {B} query rows but the per-row view holds "
                f"{row_lens.numel()}"
            )
            # FP8 Gluon reads BF16 queries and folds the KV scale into QK.
            kv_scale = 1.0
            if is_quantized_kv_cache(self.kv_cache_dtype):
                kv_scale = getattr(layer, "_k_scale_float", 1.0)
                if q_nope.dtype != torch.bfloat16:
                    q_nope = q_nope.to(torch.bfloat16) * layer._q_scale
                    q_pe = q_pe.to(torch.bfloat16) * layer._q_scale
            if self.dcp_world_size > 1:
                return self._forward_segmented_dcp_verify(
                    q_nope,
                    q_pe,
                    row_lens,
                    kv_c_and_k_pe_cache,
                    decode,
                    layer,
                    k_window,
                )
            o = torch.empty(
                B,
                num_q_heads,
                self.kv_lora_rank,
                dtype=decode.attn_out_dtype,
                device=q_nope.device,
            )

            kv_buffer = kv_c_and_k_pe_cache.reshape(-1, kv_c_and_k_pe_cache.shape[-1])
            mla_gluon = _get_mla_gluon()
            mla_gluon(
                q_nope=q_nope,
                q_pe=q_pe,
                kv_c=kv_buffer,
                o=o,
                page_table=decode.verify_row_page_table,
                seq_info=decode.verify_row_indptr,
                sm_scale=self.scale,
                k_pe=None,
                kv_pe_offset=self.kv_lora_rank,
                use_2d_view=False,
                # A scalar because the kernel folds it into the QK temperature;
                # a tensor would arrive as a pointer.
                kv_scale=kv_scale,
                min_kv_seq_len=decode.verify_min_kv_seq_len,
                return_lse=False,
            )
            return o, None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        logical_qlen = int(decode.max_qo_len)
        kernel_qlen = int(decode.kernel_max_qo_len or logical_qlen)

        assert q.shape[1] == self._decode_num_heads, (
            "ROCM_AITER_MLA decode expected the DCP-gathered query head count "
            f"{self._decode_num_heads}, got {q.shape[1]}"
        )
        # Pad before head-padding so both share the same token axis.
        if kernel_qlen != logical_qlen:
            q = AiterMLAHelper.pad_uniform_q_to_kernel_qlen(
                q, logical_qlen, kernel_qlen
            )
            B = q.shape[0]
        mla_padded_q = AiterMLAHelper.get_mla_padded_q(self._decode_num_heads, q)
        mla_num_heads = AiterMLAHelper.get_actual_mla_num_heads(self._decode_num_heads)
        o = torch.empty(
            B,
            mla_num_heads,
            self.kv_lora_rank,
            dtype=attn_metadata.decode.attn_out_dtype,
            device=q.device,
        )
        if logical_qlen > 1 and not decode.has_persistent_metadata:
            # MTP verification can call the AITER MLA decode kernel with
            # qlen > 1. If that path is running without persistent metadata,
            # zero-fill so unwritten lanes cannot leak into logits.
            o.zero_()

        kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)

        # Build kwargs for mla_decode_fwd. Pass persistent metadata only
        # when it was successfully computed.
        mla_kwargs = dict(
            q_scale=layer._q_scale,
            kv_scale=layer._k_scale,
        )
        if attn_metadata.work_meta_data is not None:
            mla_kwargs.update(
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
            )

        lse = None
        if self.dcp_world_size > 1:
            if logical_qlen != 1:
                # DCP multi-token verification requires segmented MLA.
                raise NotImplementedError(
                    "ROCM_AITER_MLA DCP multi-token verify requires the "
                    "segmented MLA path; got max_qo_len="
                    f"{logical_qlen} on the ASM path "
                    f"(kernel_qlen={kernel_qlen})."
                )
            # The vLLM custom-op wrapper exposes only the in-place output and
            # drops aiter's final LSE, which the cross-shard merge needs, so go
            # through aiter's native entry point on the DCP path.
            _, lse = _get_aiter_mla_decode()(
                mla_padded_q,
                kv_buffer.view(-1, 1, 1, mla_padded_q.shape[-1]),
                o,
                decode.qo_indptr,
                decode.paged_kv_indptr,
                decode.paged_kv_indices,
                decode.paged_kv_last_page_len,
                kernel_qlen,
                sm_scale=self.scale,
                return_lse=True,
                **mla_kwargs,
            )
            assert lse is not None, (
                "aiter mla_decode_fwd(return_lse=True) returned no LSE; upgrade "
                "aiter to a build with decode LSE support."
            )
        else:
            rocm_aiter_ops.mla_decode_fwd(
                mla_padded_q,
                kv_buffer,
                o,
                self.scale,
                decode.qo_indptr,
                kernel_qlen,
                decode.paged_kv_indptr,
                decode.paged_kv_indices,
                decode.paged_kv_last_page_len,
                **mla_kwargs,
            )

        output = AiterMLAHelper.get_mla_unpadded_o(self._decode_num_heads, o)
        if kernel_qlen != logical_qlen:
            output = AiterMLAHelper.unpad_uniform_rows(
                output, logical_qlen, kernel_qlen
            )
        if lse is not None:
            lse = AiterMLAHelper.get_mla_unpadded_lse(self._decode_num_heads, lse)
            if kernel_qlen != logical_qlen:
                lse = AiterMLAHelper.unpad_uniform_rows(lse, logical_qlen, kernel_qlen)
        return output, lse
