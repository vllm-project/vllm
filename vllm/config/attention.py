# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal

from pydantic import Field, field_validator

from vllm.config.utils import config
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

IndexerKVDType = Literal["auto", "bf16", "fp8", "mxfp4", "nvfp4"]
MiniMaxM3MSADecodeBackend = Literal["triton", "cutlass"]


@config
class HiSparseConfig:
    """Configuration for HiSparse sparse-MLA KV offloading."""

    device_buffer_size: int | None = Field(default=None, gt=0)
    """Total per-request GPU hot-buffer rows, including the newest-token slot.

    Defaults to one top-k per decode query plus one top-k of LRU slack. The
    physical allocation is rounded up to the GPU cache block size selected
    from the active backends.
    """

    eager_host_mirror: bool = True
    """Mirror decode-written KV rows to the host pool during the forward so
    page spills complete without moving data. When disabled, decode rows stay
    resident-only and evicted pages are copied to host at spill time. Prefill
    rows are always mirrored during the forward."""


@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    minimax_m3_msa_decode_backend: MiniMaxM3MSADecodeBackend = "triton"
    """Sparse decode kernel used by the MiniMax M3 MSA backend."""

    backend_per_kind: dict[str, AttentionBackendEnum] = field(default_factory=dict)
    """Per-KV-cache-group attention backend overrides, keyed by
    `KVCacheSpecKind` (e.g. `{"mla_attention": "FLASHINFER_MLA",
    "sliding_window_mla": "TRITON_MLA"}`). This lets a model that splits its
    layers across multiple KV-cache groups (e.g. interleaved full and
    sliding-window attention) use a different backend per group.

    An entry overrides `backend` for layers of the matching kind; kinds not
    listed fall back to `backend` (or automatic selection). A selected backend
    that is invalid for that kind raises at startup."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force vllm to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    tokenspeed_mla_min_split_kv: int = Field(default=1, ge=1, le=256)
    """Minimum KV splits for TokenSpeed MLA decode. The default of 1 preserves
    automatic scheduling. Larger values request a split floor and may increase
    workspace memory. Only applies to the TokenSpeed MLA backend."""

    tq_max_kv_splits_for_cuda_graph: int = 32
    """TurboQuant max NUM_KV_SPLITS for cuda graph decode.
    Fixes the split count so grid dimensions are constant across captures,
    and buffers can be pre-allocated to avoid inflating the memory estimate."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    mla_prefill_backend: MLAPrefillBackendEnum | None = None
    """MLA prefill backend to use. If None, will be selected automatically.
    Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    trtllm_ragged_backend: Literal["trtllm-gen", "cute-dsl"] = "trtllm-gen"
    """Kernel backend for cached-context TRTLLM_RAGGED MLA prefill.
    CuTe DSL requires FP8 queries for head dimension 192 and does not support
    head dimension 256. New-token prefill always uses TRTLLM."""

    indexer_kv_dtype: IndexerKVDType | str = "auto"
    """Data type for the sparse-attention indexer K cache. "auto" picks the
    model's default (bf16 for MiniMax M3, fp8 for the DeepSeek sparse
    indexer, mxfp4 for DeepSeek-V4 on ROCm CDNA4 with aiter). Quantized
    formats (fp8, mxfp4, nvfp4) require indexer kernel support in the
    backend."""

    indexer_sparse_logits: bool | None = None
    """DeepSeek V4.1 two-level indexer: score only the candidate blocks with
    DeepGEMM's sparse MQA-logits kernels instead of computing dense logits over
    the whole context and masking them. Requires an SM100-class GPU (FP8 or
    MXFP4 indexer cache), DeepGEMM >= 2.8 and the DeepSelect top-k extension
    (the top-k runs on the kernels' bf16 logits), and no decode/prefill context
    parallelism. The sparse path costs O(candidate blocks) per query regardless
    of context length. None (default) enables it wherever these requirements
    hold; True forces it (and errors where they do not); False disables it.
    On ROCm gfx950 it runs aiter's paged MXFP4 MQA-logits kernel over the
    candidate pool instead, with fp32 logits, and keeps the masked dense walk
    for steps whose contexts are too short for the pool to pay; it is not
    enabled by default there."""

    indexer_tp_shard_min_rows: int | None = None
    """Deprecated and has no effect: TP row sharding of the sparse-attention
    indexer was replaced by `sparse_mla_shard_q`. It will be removed in
    v0.34.0."""

    sparse_mla_shard_q: bool = True
    """Under tensor parallelism, shard sparse-MLA (DSA) prefill by token
    instead of by head: each TP rank runs the indexer and the sparse attention
    for its slice of the tokens with all heads, and all-to-all exchanges move
    the query and the attention output between the layouts. Applies to steps
    with at least `sparse_mla_shard_q_min_tokens` prefill tokens on the
    FlashInfer sparse MLA backend, or on FlashMLA sparse with an FP8 DeepSeek
    KV cache, without DCP, PCP or HiSparse. A mixed step's decode tokens stay
    sharded by head."""

    sparse_mla_shard_q_min_tokens: int = Field(default=8192, ge=1)
    """Minimum number of prefill tokens in a step for `sparse_mla_shard_q` to
    apply."""

    minimax_m3_indexer_cp: bool = False
    """MiniMax M3 on ROCm with tensor_parallel_size > 1: score the sparse
    indexer context-parallel at decode. Each rank reads 1/TP of the index
    cache, scores every index head on it, and per-head top-k candidates are
    exchanged and merged, so the selected blocks match the default path.
    `indexer_kv_dtype` picks the kernels that do it: a bf16 index cache ("auto"
    or "bf16") runs the Triton CP indexer, an fp8 e4m3 one the AITER MSA CP
    indexer on gfx950. This shards indexer *work* and not the KV cache, so it
    is not `decode_context_parallel_size` and requires that to be 1."""

    hisparse_config: HiSparseConfig | None = None
    """HiSparse host-resident KV configuration. Setting this enables experimental
    Model Runner V2-only HiSparse sparse-MLA decode hot-buffering. It is inferred
    with defaults when HiSparseConnector is configured (directly or via
    MultiConnector); set it explicitly only to tune its fields."""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    sparse_mla_force_mqa: bool = False
    """Force sparse MLA to use forward_mqa for all requests, including prefill.
    When False (default), pure prefill batches use forward_mha when implemented.
    Set to True to always use the MQA path."""

    sparse_mla_prefill_workspace_factor: int | None = Field(default=None, ge=2)
    """Sparse MLA prefill workspace size, in multiples of `max_model_len`.
    FlashMLA sparse gathers a prefill step's context KV into this workspace and
    runs attention in more chunks when the step holds more. The sparse indexer's
    prefill K buffer is sized to fit in the same memory. Lower values return
    memory to the KV cache at long `max_model_len`; the minimum of 2 leaves room
    for a full-length request's context-parallel shard. If None, 5 when
    `max_model_len` is at most 256K tokens, else 2."""

    flex_attn_block_m: int | None = None
    """Triton kernel BLOCK_M tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_block_n: int | None = None
    """Triton kernel BLOCK_N tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_q_block_size: int | None = None
    """Logical Q block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_m.
    If None, uses 16 for paged KV attention on PyTorch >= 2.9, and 128
    for encoder-only attention or older PyTorch versions."""

    flex_attn_kv_block_size: int | None = None
    """Logical KV block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_n.
    If None, uses the KV cache block size for paged KV attention on
    PyTorch >= 2.9, and 128 for encoder-only attention or older PyTorch
    versions."""

    def __post_init__(self) -> None:
        if self.indexer_tp_shard_min_rows is not None:
            logger.warning(
                "indexer_tp_shard_min_rows is deprecated and has no effect "
                "since sparse_mla_shard_q replaced indexer TP row sharding. "
                "It will be removed in v0.34.0."
            )
        msa_aliases: dict[AttentionBackendEnum, MiniMaxM3MSADecodeBackend] = {
            AttentionBackendEnum.CUTLASS_MSA: "cutlass",
            AttentionBackendEnum.TRITON_MSA: "triton",
        }
        if self.backend in msa_aliases:
            self.minimax_m3_msa_decode_backend = msa_aliases[self.backend]
            # The alias selects only MiniMax's sparse decode kernel. Dense
            # layers still use the platform's normal automatic backend.
            self.backend = None
        # Checked here rather than at indexer selection, which the two CP
        # backends reach separately: bf16 goes to the Triton indexer and fp8
        # to the AITER MSA one, so neither ever sees the other's dtype and
        # neither can speak for the dtypes that have no CP indexer at all.
        if self.minimax_m3_indexer_cp and self.indexer_kv_dtype not in (
            "auto",
            "bf16",
            "fp8",
            "fp8_e4m3",
        ):
            raise ValueError(
                "minimax_m3_indexer_cp requires a bf16 or fp8 e4m3 index cache "
                "(indexer_kv_dtype 'auto', 'bf16', 'fp8' or 'fp8_e4m3'), got "
                f"{self.indexer_kv_dtype!r}"
            )

    def resolve_indexer_kv_dtype(
        self, default: IndexerKVDType | str
    ) -> IndexerKVDType | str:
        """Resolve `indexer_kv_dtype`, substituting `default` for "auto"."""
        if self.indexer_kv_dtype == "auto":
            return default
        return self.indexer_kv_dtype

    def compute_hash(self) -> str:
        """Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, set())
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @field_validator("mla_prefill_backend", mode="before")
    @classmethod
    def validate_mla_prefill_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `mla_prefill_backend` enum type from string."""
        if isinstance(value, str):
            return MLAPrefillBackendEnum[value.upper()]
        return value

    @field_validator("backend_per_kind", mode="before")
    @classmethod
    def validate_backend_per_kind_before(cls, value: Any) -> Any:
        """Parse the `backend_per_kind` map from strings.

        Keys must be valid `KVCacheSpecKind` values; values are parsed like
        `backend` (enum name, case-insensitive).
        """
        from vllm.v1.kv_cache_interface import KVCacheSpecKind

        if not isinstance(value, dict):
            return value
        valid_kinds = {kind.value for kind in KVCacheSpecKind}
        parsed: dict[str, AttentionBackendEnum] = {}
        for kind, backend in value.items():
            if kind not in valid_kinds:
                raise ValueError(
                    f"Unknown KV cache group kind '{kind}' in "
                    f"backend_per_kind. Valid kinds are: "
                    f"{', '.join(sorted(valid_kinds))}."
                )
            if isinstance(backend, str):
                backend = AttentionBackendEnum[backend.upper()]
            parsed[kind] = backend
        return parsed
