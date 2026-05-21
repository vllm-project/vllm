# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DeepseekV4 MLA Attention Layer
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeepseekV2Config, DeepseekV3Config

import vllm.envs as envs
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.utils.deep_gemm import fp8_einsum
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.ops.deepseek_v4_ops import (
    build_flashinfer_mixed_sparse_indices,
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
    fused_indexer_q_rope_quant,
    fused_inv_rope_fp8_quant,
    fused_q_kv_rmsnorm,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import (
        DeepseekSparseSWAMetadata,
    )

from vllm.config import (
    CacheConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import PluggableLayer
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.deepseek_compressor import DeepseekCompressor
from vllm.model_executor.layers.layernorm import LayerNorm, RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import (
    QuantFP8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_trtllm_batch_decode_sparse_mla_dsv4_raw
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    DeepseekV4FlashMLASparseBackend,
    FlashMLASparseBackend,
    FlashMLASparseMetadata,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV4IndexerBackend,
    get_max_prefill_buffer_size,
)
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache
from vllm.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec
from vllm.v1.worker.workspace import current_workspace_manager

logger = init_logger(__name__)

_FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_flashinfer_dsv4_workspace_by_device: dict[torch.device, torch.Tensor] = {}
FlashInferSparseIndexMetadata = tuple[
    torch.Tensor,  # compressed KV cache consumed by FlashInfer.
    torch.Tensor,  # query_start_loc.
    torch.Tensor,  # query_start_loc_cpu.
    torch.Tensor,  # seq_lens.
    torch.Tensor,  # sparse_indices.
    torch.Tensor,  # sparse_topk_lens.
]

# Prefill is processed in fixed-size chunks; this bounds the bf16 kv-gather
# workspace allocated at _forward_prefill (and the matching profile-time
# reservation in attention_impl's dummy-run branch).
PREFILL_CHUNK_SIZE = 4


def _normalize_dsv4_kv_cache_dtype(
    cache_config: CacheConfig | None,
) -> str:
    kv_cache_dtype = cache_config.cache_dtype if cache_config is not None else "auto"
    if kv_cache_dtype in ("fp8", "fp8_e4m3"):
        assert cache_config is not None
        cache_config.cache_dtype = "fp8_ds_mla"
        return "fp8_ds_mla"
    if kv_cache_dtype == "fp8_inc":
        assert cache_config is not None
        cache_config.cache_dtype = "fp8_per_tensor"
        return "fp8_per_tensor"
    return kv_cache_dtype


def _dsv4_kv_cache_torch_dtype(
    kv_cache_dtype: str,
    vllm_config: VllmConfig,
) -> torch.dtype:
    if kv_cache_dtype == "fp8_ds_mla":
        return torch.uint8
    if kv_cache_dtype in ("fp8_per_tensor", "fp8_inc"):
        return torch.float8_e4m3fn
    if kv_cache_dtype == "bfloat16":
        return torch.bfloat16
    if kv_cache_dtype == "auto":
        dtype = kv_cache_dtype_str_to_dtype(kv_cache_dtype, vllm_config.model_config)
        if dtype == torch.bfloat16:
            return dtype
    raise ValueError(
        "DeepSeek V4 FlashInfer sparse MLA supports only BF16 or per-tensor "
        f"FP8 E4M3 KV cache; got kv_cache_dtype={kv_cache_dtype}. Use "
        "`bfloat16`/`auto` for BF16, `fp8_per_tensor` for per-tensor FP8, or "
        "`fp8`/`fp8_ds_mla` for the legacy UE8M0 FlashMLA path."
    )


def _get_flashinfer_dsv4_workspace(device: torch.device) -> torch.Tensor:
    workspace = _flashinfer_dsv4_workspace_by_device.get(device)
    if workspace is None:
        workspace = torch.zeros(
            _FLASHINFER_DSV4_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=device,
        )
        _flashinfer_dsv4_workspace_by_device[device] = workspace
    return workspace


@dataclass
class DeepseekV4MLAModules:
    """Modules used in DeepseekV4 MLA."""

    vllm_config: VllmConfig
    fused_wqa_wkv: torch.nn.Module
    q_norm: torch.nn.Module
    wq_b: torch.nn.Module
    kv_norm: torch.nn.Module
    wo_a: torch.nn.Module
    wo_b: torch.nn.Module
    attn_sink: torch.nn.Module
    rotary_emb: torch.nn.Module
    indexer: torch.nn.Module | None
    indexer_rotary_emb: torch.nn.Module
    topk_indices_buffer: torch.Tensor | None
    aux_stream_list: list[torch.cuda.Stream] | None = None


# --8<-- [start:multi_head_latent_attention]
@PluggableLayer.register("deepseek_v4_multi_head_latent_attention")
class DeepseekV4MultiHeadLatentAttentionWrapper(PluggableLayer):
    """Pluggable MLA layer which allows OOT backends to add
    custom implementations of the outer MLA layer (including rope & o_proj).
    Note that currently oot platforms can still use CustomOp.register_oot to
    replace MLA layer entirely, although we use PluggableLayer to register
    this layer now.

    This class takes positions and hidden_states as input.
    The input tensors can either contain prefill tokens or decode tokens.
    The class does the following:

    1. MLA Preprocess.
    2. Perform multi-head attention to prefill tokens and
       multi-query attention to decode tokens separately.
    3. Return the output tensor.
    """

    # --8<-- [end:multi_head_latent_attention]

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        o_lora_rank: int | None,
        mla_modules: DeepseekV4MLAModules,
        window_size: int,
        compress_ratio: int | None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_local_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale

        # FlashMLA sparse kernel only supports 64 or 128 heads; pad up to the
        # next supported size. Must match DeepseekV4MLAAttention.padded_heads.
        if num_heads <= 64:
            self.padded_heads = 64
        elif num_heads <= 128:
            self.padded_heads = 128
        else:
            raise ValueError(
                f"DeepseekV4 attention does not support {num_heads} heads "
                "(must be <= 128)."
            )

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.window_size = window_size
        self.compress_ratio = compress_ratio if compress_ratio is not None else 1
        self.prefix = prefix

        # Extract config from vllm_config
        config = mla_modules.vllm_config.model_config.hf_config
        tp_size = get_tensor_model_parallel_world_size()

        # DeepseekV4-specific attributes (num_heads is already TP-adjusted)
        self.eps = config.rms_norm_eps
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - self.rope_head_dim
        self.n_local_groups = config.o_groups // tp_size
        self.o_lora_rank = config.o_lora_rank

        # Store projection modules
        self.fused_wqa_wkv = mla_modules.fused_wqa_wkv
        self.q_norm = mla_modules.q_norm
        self.wq_b = mla_modules.wq_b

        self.kv_norm = mla_modules.kv_norm
        self.wo_a = mla_modules.wo_a

        self._wo_a_act_quant = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            use_ue8m0=True,
        )
        # Bypass packed-for-deepgemm path — we need FP32 scales (not packed
        # INT32) so fp8_einsum can handle layout transform internally.
        self._wo_a_act_quant.use_deep_gemm_supported = False
        self.wo_b = mla_modules.wo_b

        # Pick fp8_einsum recipe based on GPU arch:
        # SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128
        # SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1
        cap = current_platform.get_device_capability()
        assert cap is not None, "DeepseekV4 attention requires a CUDA device"
        self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
        self._tma_aligned_scales = cap.major >= 10

        self.rotary_emb = mla_modules.rotary_emb
        self.indexer_rotary_emb = mla_modules.indexer_rotary_emb
        self.topk_indices_buffer = mla_modules.topk_indices_buffer

        self.indexer = mla_modules.indexer

        # Per-head RMS normalization for Q (no learnable weights)
        self.q_head_norm = RMSNorm(head_dim, eps=self.eps, has_weight=False)

        # TODO(yifan): currently hardcoded for FP8 sparse, make it more generic
        head_bytes = (
            self.nope_head_dim  # 448 fp8 NoPE
            + self.rope_head_dim * 2  # 64 bf16 RoPE
            + self.nope_head_dim // 64  # 7B scale factors
            + 1  # 1B pad
        )

        # Will be None on ROCm for now.
        self.aux_stream_list = mla_modules.aux_stream_list
        # [0]: GEMM start / post-GEMM event0. [1..3]: GEMM done events;
        # [1] doubles as post-GEMM event1. Reuse is safe: GEMM fully joins
        # before post-GEMM starts.
        self.ln_events = [torch.cuda.Event() for _ in range(4)]

        assert cache_config is not None, "DeepseekV4 attention requires cache_config"
        kv_cache_dtype = _normalize_dsv4_kv_cache_dtype(cache_config)
        kv_cache_torch_dtype = _dsv4_kv_cache_torch_dtype(
            kv_cache_dtype, mla_modules.vllm_config
        )
        if kv_cache_dtype == "fp8_ds_mla":
            logger.info_once("Using DeepSeek's fp8_ds_mla KV cache format.")
        self.swa_cache_layer = DeepseekV4SWACache(
            head_dim=self.head_dim,
            window_size=self.window_size,
            dtype=kv_cache_torch_dtype,
            prefix=f"{prefix}.swa_cache",
            cache_config=cache_config,
        )

        self.mla_attn = DeepseekV4MLAAttention(
            num_heads=self.n_local_heads,
            head_dim=self.head_dim,
            scale=self.scale,
            qk_nope_head_dim=self.nope_head_dim,
            qk_rope_head_dim=self.rope_head_dim,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            compress_ratio=self.compress_ratio,
            window_size=self.window_size,
            head_bytes=head_bytes,
            swa_cache_layer=self.swa_cache_layer,
            attn_sink=mla_modules.attn_sink,  # already padded with -inf
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            indexer=self.indexer,
            topk_indices_buffer=self.topk_indices_buffer,
        )
        # Register this layer in the compilation config's static forward context
        # This allows the custom op to retrieve the layer during execution
        compilation_config = mla_modules.vllm_config.compilation_config
        # HACK
        self.layer_name = prefix + ".deepseek_v4_multi_head_latent_attention"
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

        # Create the compressor for layers with compress_ratio > 1; after
        # creating the DeepseekV4MLAAttention layer to get its cache.
        self.compressor = None
        if self.compress_ratio > 1:
            self.compressor = DeepseekCompressor(
                vllm_config=mla_modules.vllm_config,
                compress_ratio=self.compress_ratio,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                rotate=True,
                prefix=f"{prefix}.compressor",
                k_cache_prefix=self.mla_attn.prefix,
                # Legacy FlashMLA state/KV pages need 576B alignment. The
                # FlashInfer BF16/per-tensor FP8 path shares state pages with
                # contiguous C4 KV pages, so padding would break page matching.
                state_cache_alignment=(
                    576 if self.mla_attn.kv_cache_dtype == "fp8_ds_mla" else None
                ),
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # FlashMLA requires 64/128 heads. FlashInfer full-cache modes run on
        # the actual local head count, avoiding padded Q/output work.
        num_tokens = hidden_states.shape[0]
        use_flashinfer_full_cache = (
            self.mla_attn.kv_cache_torch_dtype != torch.uint8
            and not current_platform.is_rocm()
        )
        output_heads = (
            self.n_local_heads if use_flashinfer_full_cache else self.padded_heads
        )
        o_attn = torch.empty(
            (num_tokens, output_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # Attention (inside custom op for torch.compile boundary)
        torch.ops.vllm.deepseek_v4_attention(
            hidden_states,
            positions,
            o_attn,
            self.layer_name,
        )
        o = o_attn if use_flashinfer_full_cache else o_attn[:, : self.n_local_heads, :]

        # Keep ROCm on the BF16 reference wo_a path util kernel ready.
        if current_platform.is_rocm():
            z = rocm_inv_rope_einsum(
                self.rotary_emb,
                o,
                positions,
                self.rope_head_dim,
                self.n_local_groups,
                self.o_lora_rank,
                self.wo_a,
            )
            return self.wo_b(z.flatten(1))

        # O projection: inverse RoPE + FP8 quant + einsum + wo_b
        o_fp8, o_scale = fused_inv_rope_fp8_quant(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            tma_aligned_scales=self._tma_aligned_scales,
        )

        wo_a_fp8 = self.wo_a.weight
        wo_a_scale = self.wo_a.weight_scale_inv

        z = torch.empty(
            (num_tokens, self.n_local_groups, self.o_lora_rank),
            device=o.device,
            dtype=torch.bfloat16,
        )
        torch.ops.vllm.deepseek_v4_fp8_einsum(
            o_fp8,
            o_scale,
            wo_a_fp8,
            wo_a_scale,
            z,
            "bhr,hdr->bhd",
            list(self._einsum_recipe),
        )

        return self.wo_b(z.flatten(1))

    def attn_gemm_parallel_execute(self, hidden_states) -> tuple[Any, ...]:
        aux_streams = self.aux_stream_list
        if aux_streams is not None:
            assert len(aux_streams) >= 3
            aux_streams = aux_streams[:3]

        # fused_wqa_wkv (heaviest) on default; the three lighter input GEMMs
        # on aux streams 0..2 when their owning module exists. ln_events[0]
        # is the fan-out start event; ln_events[1..3] are per-aux done events.
        # On ROCm, aux_streams is None and execute_in_parallel runs serially.
        aux_fns: list[Callable[[], Any] | None] = [None, None, None]

        if self.compressor is not None:
            # Local ref so the closure keeps a non-None type for mypy.
            compressor = self.compressor

            def compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[0] = compressor_kv_score

        if self.indexer is not None:
            indexer = self.indexer

            def indexer_weights_proj() -> torch.Tensor:
                # ReplicatedLinear returns (output, bias); bias is None.
                weights, _ = indexer.weights_proj(hidden_states)
                return weights

            def indexer_compressor_kv_score() -> torch.Tensor:
                return torch.mm(
                    hidden_states,
                    indexer.compressor.fused_wkv_wgate.weight.T,
                    out_dtype=torch.float32,
                )

            aux_fns[1] = indexer_weights_proj
            aux_fns[2] = indexer_compressor_kv_score

        def fused_wqa_wkv() -> torch.Tensor:
            # MergedColumnParallelLinear returns (output, bias); bias is None.
            qr_kv, _ = self.fused_wqa_wkv(hidden_states)
            return qr_kv

        qr_kv, (kv_score, indexer_weights, indexer_kv_score) = execute_in_parallel(
            fused_wqa_wkv,
            aux_fns,
            self.ln_events[0],
            self.ln_events[1:4],
            aux_streams,
            enable=hidden_states.shape[0]
            <= envs.VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD,
        )

        return qr_kv, kv_score, indexer_kv_score, indexer_weights

    def attention_impl(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor,  # [num_tokens, padded_heads, head_dim], written in place
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )

        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr, kv = fused_q_kv_rmsnorm(
            qr,
            kv,
            self.q_norm.weight.data,
            self.kv_norm.weight.data,
            self.eps,
        )

        # wq_b + kv_insert (+ MLA compressor when an indexer is present) ride
        # on the default stream so q stays on its consumer stream (mla_attn
        # downstream reads q on current). Indexer/compressor go on aux for
        # overlap with default's GEMM + cache write.
        if self.indexer is not None:
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            indexer = self.indexer
            assert self.compressor is not None
            compressor = self.compressor

            def wq_b_kv_insert_and_compress() -> tuple[
                torch.Tensor, torch.Tensor | None
            ]:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q_fp8 = self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                )
                compressor(kv_score, positions, self.rotary_emb)
                return q, q_fp8

            (q, q_fp8), _ = maybe_execute_in_parallel(
                wq_b_kv_insert_and_compress,
                lambda: indexer(
                    hidden_states,
                    qr,
                    indexer_kv_score,
                    indexer_weights,
                    positions,
                    self.indexer_rotary_emb,
                ),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        elif self.compressor is not None:
            # wq_b + kv_insert on default, compressor on aux.
            aux_stream = (
                self.aux_stream_list[0] if self.aux_stream_list is not None else None
            )
            compressor = self.compressor

            def wq_b_kv_insert() -> tuple[torch.Tensor, torch.Tensor | None]:
                q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
                q_fp8 = self._fused_qnorm_rope_kv_insert(
                    q, kv, positions, attn_metadata
                )
                return q, q_fp8

            (q, q_fp8), _ = maybe_execute_in_parallel(
                wq_b_kv_insert,
                lambda: compressor(kv_score, positions, self.rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                aux_stream,
            )
        else:
            # SWA-only layer: no compressor, no overlap.
            q = self.wq_b(qr).view(-1, self.n_local_heads, self.head_dim)
            q_fp8 = self._fused_qnorm_rope_kv_insert(q, kv, positions, attn_metadata)

        # Handle dummy run (no metadata).
        if not isinstance(attn_metadata, dict):
            # Reserve _forward_prefill's bf16-gather workspace; the dummy
            # run returns before mla_attn runs, so without this the shared
            # workspace locks below the real prefill size. The per-tensor FP8
            # FlashInfer path reads the cache directly and does not need it.
            sub = self.mla_attn
            if sub.kv_cache_torch_dtype != torch.float8_e4m3fn:
                swa_only = sub.compress_ratio <= 1
                N = (
                    0
                    if swa_only
                    else (sub.max_model_len + sub.compress_ratio - 1)
                    // sub.compress_ratio
                )
                M = N + sub.window_size + sub.max_num_batched_tokens
                current_workspace_manager().get_simultaneous(
                    ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
                )
            out.zero_()
            return

        q_for_attn = q_fp8 if q_fp8 is not None else q

        # Pad q only for the legacy FlashMLA path, which requires 64 or 128
        # heads. FlashInfer full-cache modes keep the actual local head count.
        if (
            q_fp8 is None
            and self.mla_attn.kv_cache_torch_dtype == torch.uint8
            and self.n_local_heads < self.padded_heads
        ):
            pad_size = self.padded_heads - self.n_local_heads
            q_for_attn = F.pad(q_for_attn, (0, 0, 0, pad_size), value=0.0)

        # MLA attention writes into the pre-allocated `out` buffer. FlashMLA
        # gets a padded-head buffer; FlashInfer full-cache modes get actual
        # local heads.
        self.mla_attn(
            q_for_attn,
            kv,
            positions,
            output=out,
        )

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: (
            dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]] | None
        ),
    ) -> torch.Tensor | None:
        if not isinstance(attn_metadata, dict):
            return None

        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Horizontally fused:
        #   Q side:  q_head_norm (per-head RMSNorm, no weight) + GPT-J RoPE
        #   KV side: GPT-J RoPE + paged cache insert. The uint8 cache keeps the
        #   legacy UE8M0 layout; BF16/FP8 caches store the full 512-wide vector.
        # kv is unchanged; mla_attn reads kv solely via swa_kv_cache.
        if swa_kv_cache.dtype == torch.uint8:
            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q,
                kv,
                swa_kv_cache_2d,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return None

        if swa_kv_cache.dtype == torch.bfloat16:
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
                q,
                kv,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.eps,
                swa_metadata.block_size,
            )
            return None

        if swa_kv_cache.dtype == torch.float8_e4m3fn:
            q_fp8 = torch.empty(
                (q.shape[0], self.n_local_heads, q.shape[-1]),
                dtype=torch.float8_e4m3fn,
                device=q.device,
            )
            torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_fp8_insert(
                q,
                kv,
                q_fp8,
                swa_kv_cache,
                swa_metadata.slot_mapping,
                positions.to(torch.int64),
                self.rotary_emb.cos_sin_cache,
                self.mla_attn._flashinfer_fp8_kv_scale,
                self.mla_attn._flashinfer_fp8_q_scale_inv,
                self.eps,
                swa_metadata.block_size,
            )
            return q_fp8

        raise AssertionError(f"Unsupported SWA KV cache dtype {swa_kv_cache.dtype}")


def deepseek_v4_attention(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.attention_impl(hidden_states, positions, out)


def deepseek_v4_attention_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_attention",
    op_func=deepseek_v4_attention,
    mutates_args=["out"],
    fake_impl=deepseek_v4_attention_fake,
)


def deepseek_v4_fp8_einsum(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    fp8_einsum(equation, (a, a_scale), (b, b_scale), out, recipe=tuple(recipe))


def deepseek_v4_fp8_einsum_fake(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    b: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    equation: str,
    recipe: list[int],
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_fp8_einsum",
    op_func=deepseek_v4_fp8_einsum,
    mutates_args=["out"],
    fake_impl=deepseek_v4_fp8_einsum_fake,
)


class DeepseekV4MLAAttention(nn.Module, AttentionLayerBase):
    # FlashMLA FP8 sparse only supports 64 or 128 heads
    SUPPORTED_HEAD_COUNTS = (64, 128)

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        compress_ratio: int,
        window_size: int,
        head_bytes: int,
        swa_cache_layer: DeepseekV4SWACache,
        attn_sink: torch.Tensor,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        # Sparse MLA Args
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        aux_stream: torch.cuda.Stream | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = 1
        self.head_dim = head_dim
        self.scale = scale
        self.window_size = window_size
        self.head_bytes = head_bytes
        self.compress_ratio = compress_ratio
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.nope_head_dim = qk_nope_head_dim
        self.rope_head_dim = qk_rope_head_dim
        self.indexer = indexer
        self.topk_indices_buffer = topk_indices_buffer

        self.prefix = prefix  # Alias for compatibility with compressor

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        # Determine padded head count for FlashMLA
        if num_heads not in self.SUPPORTED_HEAD_COUNTS:
            if num_heads < 64:
                self.padded_heads = 64
            elif num_heads < 128:
                self.padded_heads = 128
            else:
                raise ValueError(
                    f"DeepseekV4MLAAttention does not support {num_heads} heads. "
                    f"Supported: <= 128 (will be padded to 64 or 128)"
                )
        else:
            self.padded_heads = num_heads

        # Store attention sink
        assert attn_sink is not None
        self.attn_sink: torch.Tensor = attn_sink
        # Store SWA cache
        assert swa_cache_layer is not None
        self.swa_cache_layer: DeepseekV4SWACache = swa_cache_layer

        # Get vllm config for cache setup
        vllm_config = get_current_vllm_config()
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        assert issubclass(self.get_attn_backend(), FlashMLASparseBackend), (
            "DeepseekV4 requires the sparse MLA metadata/cache backend"
        )
        kv_cache_dtype = _normalize_dsv4_kv_cache_dtype(cache_config)
        self.kv_cache_torch_dtype = _dsv4_kv_cache_torch_dtype(
            kv_cache_dtype, vllm_config
        )

        self.kv_cache_dtype = kv_cache_dtype
        fp8_q_scale = 1.0
        fp8_kv_scale = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            # TODO: load the per-tensor FP8 Q and KV scales from checkpoint
            # weights. Use unit scales until the scale tensor names are wired.
            fp8_q_scale = 1.0
            fp8_kv_scale = 1.0
        self.register_buffer(
            "_flashinfer_fp8_q_scale",
            torch.tensor([fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_q_scale_inv",
            torch.tensor([1.0 / fp8_q_scale], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_flashinfer_fp8_kv_scale",
            torch.tensor([fp8_kv_scale], dtype=torch.float32),
            persistent=False,
        )
        self._flashinfer_fp8_bmm1_scale = self.scale * fp8_q_scale * fp8_kv_scale
        self._flashinfer_fp8_bmm2_scale = fp8_kv_scale

        # Register with compilation context for metadata lookup
        compilation_config = vllm_config.compilation_config
        if prefix and prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        if prefix:
            compilation_config.static_forward_context[prefix] = self

        self.kv_cache = torch.tensor([])

    def get_attn_backend(self) -> type[AttentionBackend]:
        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse_dsv4 import (
                DeepseekV4ROCMAiterMLASparseBackend,
            )

            return DeepseekV4ROCMAiterMLASparseBackend
        return DeepseekV4FlashMLASparseBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        if (
            self.compress_ratio <= 1
        ):  # SWA part. Allocated separately as DeepseekV4SWACache.
            return None
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            compress_ratio=self.compress_ratio,
            cache_dtype_str=self.kv_cache_dtype,
            # FlashMLA's legacy fp8_ds_mla layout needs 576B page alignment.
            # FlashInfer DSV4 BF16/per-tensor FP8 sparse decode treats the KV
            # pool as a flat contiguous token array, so padding would skew
            # physical sparse indices after the first page.
            alignment=576 if self.kv_cache_dtype == "fp8_ds_mla" else None,
            model_version="deepseek_v4",
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        expected_output_dtype = (
            torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
        )
        assert output.dtype == expected_output_dtype, (
            f"output buffer dtype {output.dtype} must match expected attention "
            f"output dtype {expected_output_dtype} for q dtype {q.dtype}"
        )

        if current_platform.is_rocm():
            from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse_dsv4 import (
                DeepseekV4ROCMAiterMLASparseImpl,
            )

            DeepseekV4ROCMAiterMLASparseImpl.forward(self, q, kv, positions, output)
            return

        # Get SWA and indexer metadata from forward context
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            FlashMLASparseMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation, so self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        # Split prefill and decode
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if self.kv_cache_torch_dtype != torch.uint8:
            if current_platform.is_rocm():
                raise NotImplementedError(
                    "DeepSeek V4 BF16/per-tensor FP8 FlashInfer sparse MLA "
                    "cache path is CUDA-only."
                )
            self._forward_flashinfer(
                q=q,
                kv_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output,
            )
            return

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        assert self.kv_cache_torch_dtype == torch.uint8

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            attn_sink=self.attn_sink,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            out=output.unsqueeze(1),
        )

    def _build_flashinfer_sparse_index_metadata(
        self,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
    ) -> FlashInferSparseIndexMetadata:
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens

        assert swa_metadata.seq_lens is not None
        assert swa_metadata.query_start_loc is not None
        assert swa_metadata.query_start_loc_cpu is not None
        assert swa_metadata.token_to_req_indices is not None
        assert swa_metadata.decode_swa_indices is not None
        assert swa_metadata.block_table is not None

        decode_swa_indices = swa_metadata.decode_swa_indices.reshape(
            num_decode_tokens, self.window_size
        )
        decode_compressed_topk_lens = None
        decode_compressed_indices_are_local = False
        decode_is_valid_token = None

        if swa_only:
            assert self.topk_indices_buffer is not None
            compressed_kv_cache = swa_k_cache
            decode_compressed_indices = None
            prefill_topk_indices = self.topk_indices_buffer[
                num_decode_tokens:num_tokens, :0
            ]
            compressed_block_table = None
            compressed_block_size = swa_metadata.block_size
            top_k = 0
        else:
            assert kv_cache is not None
            assert attn_metadata is not None
            compressed_kv_cache = kv_cache
            compressed_block_table = attn_metadata.block_table[:num_reqs]
            compressed_block_size = attn_metadata.block_size // self.compress_ratio

            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                if num_prefill_tokens > 0:
                    prefill_topk_indices = self.topk_indices_buffer[
                        num_decode_tokens:num_tokens
                    ]
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = self.topk_indices_buffer[:0, :0]
                    top_k = 0

                decode_compressed_indices_are_local = True
                assert swa_metadata.is_valid_token is not None
                decode_is_valid_token = swa_metadata.is_valid_token[:num_decode_tokens]
                if num_decode_tokens > 0:
                    decode_compressed_indices = self.topk_indices_buffer[
                        :num_decode_tokens
                    ]
                else:
                    # The decode-side pointers are unused when there are no
                    # decode tokens. Keep their logical width aligned with the
                    # mixed-batch case so pure-prefill steps reuse the same
                    # Triton specialization compiled during graph capture.
                    decode_compressed_indices = prefill_topk_indices[:0]
            else:
                if num_prefill_tokens > 0:
                    assert attn_metadata.c128a_prefill_topk_indices is not None
                    prefill_topk_indices = attn_metadata.c128a_prefill_topk_indices
                    top_k = prefill_topk_indices.shape[-1]
                else:
                    prefill_topk_indices = decode_swa_indices[:0, :0]
                    top_k = 0

                if num_decode_tokens > 0:
                    assert attn_metadata.c128a_global_decode_topk_indices is not None
                    assert attn_metadata.c128a_decode_topk_lens is not None
                    decode_compressed_indices = (
                        attn_metadata.c128a_global_decode_topk_indices.view(
                            num_decode_tokens, -1
                        )
                    )
                    decode_compressed_topk_lens = attn_metadata.c128a_decode_topk_lens
                    if num_prefill_tokens == 0:
                        prefill_topk_indices = decode_compressed_indices[:0, :0]
                else:
                    # As above, these decode tensors are unused for pure prefill.
                    # Preserve the C128A topk width and lens-present flag to
                    # share the mixed-batch sparse-index kernel variant.
                    decode_compressed_indices = prefill_topk_indices[:0]
                    decode_compressed_topk_lens = swa_metadata.seq_lens[:0]

        query_start_loc = swa_metadata.query_start_loc[: num_reqs + 1]
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu[: num_reqs + 1]
        seq_lens = swa_metadata.seq_lens[:num_reqs]
        assert seq_lens.dtype == torch.int32
        sparse_indices, sparse_topk_lens = build_flashinfer_mixed_sparse_indices(
            decode_swa_indices,
            decode_compressed_indices,
            decode_compressed_topk_lens,
            prefill_topk_indices[:num_prefill_tokens],
            query_start_loc,
            seq_lens,
            swa_metadata.token_to_req_indices[:num_tokens],
            swa_metadata.block_table[:num_reqs],
            swa_metadata.block_size,
            compressed_block_table,
            compressed_block_size,
            self.window_size,
            self.compress_ratio,
            top_k,
            decode_compressed_indices_are_local=decode_compressed_indices_are_local,
            decode_is_valid_token=decode_is_valid_token,
        )
        return (
            compressed_kv_cache,
            query_start_loc,
            query_start_loc_cpu,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        )

    def _forward_flashinfer(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.kv_cache_torch_dtype in (torch.bfloat16, torch.float8_e4m3fn)
        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_reqs = num_decodes + num_prefills
        num_tokens = num_decode_tokens + num_prefill_tokens
        if num_tokens == 0:
            return

        flashinfer_sparse_metadata = self._build_flashinfer_sparse_index_metadata(
            kv_cache=kv_cache,
            swa_k_cache=swa_k_cache,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
        )
        (
            compressed_kv_cache,
            query_start_loc,
            query_start_loc_cpu,
            seq_lens,
            sparse_indices,
            sparse_topk_lens,
        ) = flashinfer_sparse_metadata

        # CUDA graph execution can pad q/output past the scheduled token count.
        # The FlashInfer DSV4 launcher validates sparse_indices against real
        # tokens, so keep the tensors restricted to the scheduled token range.
        query = q[:num_tokens]
        output = output[:num_tokens]
        bmm1_scale: float | torch.Tensor = self.scale
        bmm2_scale: float | torch.Tensor = 1.0
        if self.kv_cache_torch_dtype == torch.float8_e4m3fn:
            assert query.dtype == torch.float8_e4m3fn
            bmm1_scale = self._flashinfer_fp8_bmm1_scale
            bmm2_scale = self._flashinfer_fp8_bmm2_scale
        else:
            assert query.dtype == torch.bfloat16
            query = query.contiguous()

        workspace = _get_flashinfer_dsv4_workspace(q.device)

        if num_decode_tokens > 0:
            decode_query_start_loc = query_start_loc[: num_decodes + 1]
            decode_query_start_loc_cpu = query_start_loc_cpu[: num_decodes + 1]
            decode_query_lens_cpu = (
                decode_query_start_loc_cpu[1:] - decode_query_start_loc_cpu[:-1]
            )
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4_raw(
                query=query[:num_decode_tokens],
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=sparse_indices[:num_decode_tokens],
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=sparse_topk_lens[:num_decode_tokens],
                seq_lens=seq_lens[:num_decodes],
                out=output[:num_decode_tokens],
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=decode_query_start_loc,
                max_q_len=int(decode_query_lens_cpu.max().item()),
            )

        if num_prefill_tokens > 0:
            assert swa_metadata.prefill_query_start_loc is not None
            prefill_query_start_loc = swa_metadata.prefill_query_start_loc
            prefill_query_start_loc_cpu = query_start_loc_cpu[
                num_decodes : num_reqs + 1
            ]
            prefill_query_lens_cpu = (
                prefill_query_start_loc_cpu[1:] - prefill_query_start_loc_cpu[:-1]
            )
            prefill_query = query[num_decode_tokens:num_tokens]
            prefill_output = output[num_decode_tokens:num_tokens]
            prefill_sparse_indices = sparse_indices[num_decode_tokens:num_tokens]
            prefill_sparse_topk_lens = sparse_topk_lens[num_decode_tokens:num_tokens]
            flashinfer_trtllm_batch_decode_sparse_mla_dsv4_raw(
                query=prefill_query,
                swa_kv_cache=swa_k_cache,
                workspace_buffer=workspace,
                sparse_indices=prefill_sparse_indices,
                compressed_kv_cache=compressed_kv_cache,
                sparse_topk_lens=prefill_sparse_topk_lens,
                seq_lens=seq_lens[num_decodes:num_reqs],
                out=prefill_output,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                sinks=self.attn_sink,
                cum_seq_lens_q=prefill_query_start_loc,
                max_q_len=int(prefill_query_lens_cpu.max().item()),
            )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            flash_mla_sparse_fwd(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_length=combined_lens,
                out=output[query_start:query_end],
            )


class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # compress_ratio=1 for V3.2, >1 for DeepseekV4; both use the same cache layout.
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            compress_ratio=self.compress_ratio,
            # DeepseekV4 aligns indexer pages to FlashMLA's 576B so they can pack with
            # the indexer's compressor state cache. V3.2 keeps the legacy layout.
            alignment=576,
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config,
        hidden_size: int,
        q_lora_rank: int,
        quant_config: QuantizationConfig | None,
        cache_config: CacheConfig | None,
        topk_indices_buffer: torch.Tensor | None,
        compress_ratio: int = 1,
        prefix: str = "",
    ):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = config
        self.quant_config = quant_config
        # self.indexer_cfg = config.attn_module_list_cfg[0]["attn_index"]
        self.topk_tokens = config.index_topk
        self.n_head = config.index_n_heads  # 64
        self.head_dim = config.index_head_dim  # 128
        self.rope_dim = config.qk_rope_head_dim  # 64
        self.q_lora_rank = q_lora_rank  # 1536
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache
        logger.info_once(
            "Using %s indexer cache for Lightning Indexer.",
            "MXFP4" if self.use_fp4_kv else "FP8",
        )

        # no tensor parallel, just replicated
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.head_dim * self.n_head,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.wq_b",
        )
        self.weights_proj = ReplicatedLinear(
            hidden_size,
            self.n_head,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.weights_proj",
        )
        self.k_norm = LayerNorm(self.head_dim, eps=1e-6)
        self.softmax_scale = self.head_dim**-0.5

        self.scale_fmt = "ue8m0"
        self.quant_block_size = 128  # TODO: get from config
        self.topk_indices_buffer = topk_indices_buffer

        self.max_model_len = (
            vllm_config.model_config.max_model_len // self.compress_ratio
        )
        self.prefix = prefix

        self.max_total_seq_len = (
            get_max_prefill_buffer_size(vllm_config) // self.compress_ratio
        )

        assert cache_config is not None, "Deepseek V4 indexer requires cache_config"
        # NOTE(yifan): FP8 indxer cache use the same layout as V3.2:
        # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
        # For FP4 indexer cache, we still allocate the same amount of memory as FP8,
        # but only use the first half of the memory.
        k_cache_head_dim = self.head_dim + self.head_dim // self.quant_block_size * 4
        self.k_cache = DeepseekV4IndexerCache(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = DeepseekCompressor(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )

        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        compressed_kv_score: torch.Tensor,
        indexer_weights: torch.Tensor,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> torch.Tensor:
        # ReplicatedLinear returns (output, bias); bias is None.
        q, _ = self.wq_b(qr)
        q = q.view(-1, self.n_head, self.head_dim)
        k = self.compressor(compressed_kv_score, positions, rotary_emb)
        q_quant, weights = fused_indexer_q_rope_quant(
            positions,
            q,
            rotary_emb.cos_sin_cache,
            indexer_weights,
            self.softmax_scale,
            self.n_head**-0.5,
            use_fp4=self.use_fp4_kv,
        )
        return self.indexer_op(hidden_states, q_quant, k, weights)
