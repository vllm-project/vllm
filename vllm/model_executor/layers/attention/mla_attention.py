# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
# MLA Common Components

This file implements common components for MLA implementations.

First we define:

Sq      as Q sequence length
Skv     as KV sequence length

MLA has two possible ways of computing, a data-movement friendly approach and a
compute friendly approach, we generally want to use the compute friendly
approach for "prefill" (i.e. the ratio Sq / Skv is "small", is near 1)
and the data-movement friendly approach for "decode" (i.e. the ratio
Sq / Skv is "large").

NOTE what we deem small and large is currently determined by if its labelled
prefill or decode by the scheduler, but this is something we should probably
tune.

Main reference: DeepseekV2 paper, and FlashInfer Implementation
(https://arxiv.org/abs/2405.04434 and https://github.com/flashinfer-ai/flashinfer/pull/551).

Deepseek's MLA attention works the following way:
* Use a single latent vector to represent the per-token entry of the KV cache.
* For decode (i.e. the memory friendly approach) the attention "simulates" a
multi-head attention, while the compute is similar to multi-query attention.

Below is example of both paths assuming batchsize = 1

## More Extent Definitions:

C           Context length, `Skv - Sq`
H           hidden size
N           number of attention heads
Lq          latent dimension for Q              1536 in DSV3
Lkv         latent dimension for K/V            512 in DSV3
P           nope dimension, no rope.            128 in DSV3
R           rope dimension, goes through rope.  64 in DSV3
V           V head dim.                         128 in DSV3

## Vector/Matrix Definitions

h_t         hidden states (input to attention)  shape [Sq, H]
q_c         latent/compressed Q                 shape [Sq, Lq]
q_nope      uncompressed Q (no-rope)            shape [Sq, N, P]
q_pe        uncompressed Q (rope)               shape [Sq, N, R]
kv_c        latent/compressed KV                shape [Skv, Lkv]
k_pe        decoupled k position embeddings     shape [Skv, R]
new_kv_c    new kv_c from current iter          shape [Sq, Lkv]
new_k_pe    new k_pe from current iter          shape [Sq, R]
cache_kv_c  cached k_c from previous iters      shape [C, Lkv]
cache_k_pe  cached k_pe from previous iters     shape [C, R]
W_DQ        project h_t to q_c                  shape [H, Lq]
W_UQ        project q_c to q_nope               shape [Lq, N * P]
W_QR        project q_c to q_pe                 shape [Lq, N * R]
W_DKV       project h_t to kv_c                 shape [H, Lkv]
W_UK        project kv_c to k_nope              shape [Lkv, N, P]
W_KR        project h_t to k_pe                 shape [H, R]
W_UV        project kv_c to v                   shape [Lkv, N, V]
W_O         project v to h_t                    shape [N * V, H]


## Compute Friendly Approach (i.e. "forward_mha"):

q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(Sq, N, P)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)
k_nope   = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v        = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R
//           V headdim = V
//      spda_o shape [Sq, N, V]
spda_o = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([k_nope, k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    v
)
return spda_o @ W_O

NOTE: in the actual code,
    `kv_b_proj` is [W_UK; W_UV] concatenated per head
    `q_b_proj` is [W_UQ; W_QR] concatenated per head
    `out_proj` is W_O


## Data-Movement Friendly Approach (i.e. "forward_mqa"):

Runtime
q_c      = h_t @ W_DQ
q_nope   = (q_c @ W_UQ).view(-1, N, P)
ql_nope  = einsum("snh,lnh->snl", q, W_UK)
q_pe     = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c = h_t @ W_DKV
new_k_pe = RoPE(h_t @ W_KR)
kv_c     = torch.cat([new_kv_c, cache_kv_c], dim=0)
k_pe     = torch.cat([new_k_pe, cache_k_pe], dim=0)

// MQA with QK headdim = Lkv + R
//           V headdim = Lkv
//      spda_o shape [Sq, N, Lkv]
// NOTE: this is less compute-friendly since Lkv > P
//       but is more data-movement friendly since its MQA vs MHA
spda_o = scaled_dot_product_attention(
    torch.cat([ql_nope, q_pe], dim=-1),
    torch.cat([kv_c, k_pe], dim=-1),
    kv_c
)

o = einsum("snl,lnv->snv", spda_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ self.num_heads @ W_O


## Chunked Prefill

For chunked prefill we want to use the compute friendly algorithm. We are
assuming sufficiently large Sq / Skv ratio, in the future may want to switch to
the data-movement friendly approach if the chunk (i.e. `Sq`) is small.

However, the compute-friendly approach can potentially run out of memory if Skv
is large due to: `k_nope = (kv_c @ W_UK).view(Skv, N, P)`

To mitigate this, we chunk the computation of attention with respect to the
current context (i.e. `cache_kv_c` and `cache_k_pe`) so that we can used a
fixed workspace size.

The chunked prefill approach is as follows:

MCC        Max chunk of context to process per iter, computed dynamically,
           used to bound the memory usage

q_c        = h_t @ W_DQ
q_nope     = (q_c @ W_UQ).view(Sq, N, P)
q_pe       = RoPE(q_c @ W_QR).view(Sq, N, R)
new_kv_c   = h_t @ W_DKV
new_k_pe   = RoPE(h_t @ W_KR)
new_k_nope = (new_kv_c @ W_UK.view(Lkv, N * P)).view(Sq, N, P)
new_v      = (new_kv_c @ W_UV.view(Lkv, N * V)).view(Sq, N, V)

// MHA between queries and new KV
//     with QK headdim = P + R
//           V headdim = V
//    curr_o   shape [Sq, N, V]
//    curr_lse shape [N, Sq], this is just order FA returns
curr_o, curr_lse = scaled_dot_product_attention(
    torch.cat([q_nope, q_pe], dim=-1),
    torch.cat([new_k_nope, new_k_pe.unsqueeze(1).expand(-1, N, -1)], dim=-1),
    new_v,
    casual=True,
    return_softmax_lse=True
)

// Compute attention with the already existing context
for chunk_idx in range(cdiv(C, MCC)):
    chunk_start  = chunk_idx * MCC
    chunk_end    = min(chunk_start + MCC, C)
    Sc           = chunk_end - chunk_start
    cache_kv_c_chunk   = cache_kv_c[chunk_start:chunk_end]
    cache_k_pe_chunk   = cache_k_pe[chunk_start:chunk_end]
    cache_k_nope_chunk = (cache_kv_c_chunk @ W_UK).view(-1, N, P)
    cache_v_chunk      = (cache_kv_c_chunk @ W_UV).view(-1, N, V)

    chunk_o, chunk_lse = scaled_dot_product_attention(
        torch.cat([q_nope, q_pe], dim=-1),
        torch.cat([cache_k_nope_chunk,
                   cache_k_pe_chunk.unsqueeze(1).expand(-1, N, -1)],
                   dim=-1),
        cache_v_chunk,
        casual=False,
        return_softmax_lse=True
    )

    curr_o, curr_lse = merge_attn_states(
        suffix_output=curr_o,
        suffix_lse=curr_lse,
        prefix_output=chunk_o,
        prefix_lse=chunk_lse,
    )

return curr_o @ W_O
"""

import functools
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Generic, TypeVar, cast

import torch
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
)
from vllm.distributed.parallel_state import (
    get_dcp_group,
    is_global_first_rank,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.attention.attention import (
    _init_kv_cache_quant,
    get_attention_context,
    set_default_quant_scales,
    should_load_quant_weights,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    get_and_maybe_dequant_weights,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.math_utils import cdiv, round_down
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    is_quantized_kv_cache,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MLAAttentionImpl,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.prefill import (
    MLAPrefillBackend,
    get_mla_prefill_backend,
)
from vllm.v1.attention.backends.utils import (
    get_dcp_local_seq_lens,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
)

logger = init_logger(__name__)

_FP8_DTYPE = current_platform.fp8_dtype()


def _detect_output_quant_key(
    output: torch.Tensor,
    output_scale: torch.Tensor | None,
    output_block_scale: torch.Tensor | None,
    output_dim: int,
) -> QuantKey | None:
    """Detect the output quantization key from fusion pass parameters.

    Returns the appropriate QuantKey, or None if no quantization is needed.
    Detection is based on output dtype and which scale tensors are present.
    """
    if output_scale is None and output_block_scale is None:
        return None
    if output_block_scale is not None:
        if output.dtype == _FP8_DTYPE:
            # Per-group FP8 uses block scales only, not a separate output_scale
            assert output_scale is None
            # Infer group size from scale shape
            num_groups = output_block_scale.shape[-1]
            group_size = output_dim // num_groups
            if group_size == 128:
                return kFp8Dynamic128Sym
            elif group_size == 64:
                return kFp8Dynamic64Sym
            else:
                raise ValueError(
                    f"Unsupported group FP8 group_size={group_size} "
                    f"(output_dim={output_dim}, num_groups={num_groups}). "
                    f"Only group_size 128 and 64 are supported."
                )
        # output_scale None implies MXFP4, not supported
        assert output_scale is not None
        return kNvfp4Dynamic
    return kFp8StaticTensorSym


class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class

    This class takes query, and compressed key/value tensors as input.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        scale: float,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int | None,
        kv_lora_rank: int,
        kv_b_proj: ColumnParallelLinear,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_backend: type[AttentionBackend] | None = None,
        use_sparse: bool = False,
        indexer: object | None = None,
        **extra_impl_args,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.kv_b_proj = kv_b_proj
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix
        self.indexer = indexer

        self.num_kv_heads = 1
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False
        self.quant_config = quant_config

        dtype = torch.get_default_dtype()
        if attn_backend is not None:
            assert attn_backend.is_mla(), (
                f"MLAAttention: attn_backend must be an MLA backend, "
                f"got {attn_backend.get_name()} instead"
            )
            self.attn_backend = attn_backend
        else:
            self.attn_backend = get_attn_backend(
                self.head_size,
                dtype,
                kv_cache_dtype,
                use_mla=True,
                use_sparse=use_sparse,
                num_heads=self.num_heads,
            )

        # FlashMLA Sparse Attention fp8 backend uses "fp8_ds_mla" kv-cache format
        # Automatically convert fp8 kv-cache format to "fp8_ds_mla"
        if (
            self.attn_backend.get_name() == "FLASHMLA_SPARSE"
            and is_quantized_kv_cache(kv_cache_dtype)
            and kv_cache_dtype != "fp8_ds_mla"
        ):
            assert cache_config is not None
            cache_config.cache_dtype = "fp8_ds_mla"
            kv_cache_dtype = "fp8_ds_mla"
            logger.info_once(
                "Using DeepSeek's fp8_ds_mla KV cache format. To use standard "
                "fp8 kv-cache format, please set `--attention-backend "
                "FLASHINFER_MLA_SPARSE`"
            )

        if (
            self.attn_backend.get_name() == "FLASHINFER_MLA_SPARSE"
            and is_quantized_kv_cache(kv_cache_dtype)
        ):
            logger.info_once(
                "Using standard fp8 KV cache format. To use DeepSeek's fp8_ds_mla "
                "KV cache format, please set `--attention-backend FLASHMLA_SPARSE`"
            )

        # Initialize KV cache quantization attributes
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        _init_kv_cache_quant(self, quant_config, prefix)

        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and envs.VLLM_BATCH_INVARIANT
            and (
                self.attn_backend.get_name() == "TRITON_MLA"
                or self.attn_backend.get_name() == "FLASHINFER"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for TRITON_MLA / FLASHINFER "
                "with batch invariance, as it is not yet supported.",
            )
            cache_config.enable_prefix_caching = False

        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an MLAAttentionImpl subclass
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.scale,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=self.kv_cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
            # MLA Args
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            qk_head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            kv_b_proj=kv_b_proj,
            indexer=indexer,
            **extra_impl_args,
        )
        self.q_pad_num_heads = getattr(self.impl, "q_pad_num_heads", None)
        self.use_direct_call = not current_platform.opaque_attention_op()

        vllm_config = get_current_vllm_config()
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        prefill_backend_cls = get_mla_prefill_backend(vllm_config)
        self.prefill_backend = prefill_backend_cls(
            num_heads=self.num_heads,
            scale=self.scale,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            vllm_config=vllm_config,
        )

        self.kv_cache = torch.tensor([])

        self.use_sparse = use_sparse

        _vllm_config = get_current_vllm_config_or_none()
        self.dcp_a2a = (
            _vllm_config is not None
            and _vllm_config.parallel_config.decode_context_parallel_size > 1
            and _vllm_config.parallel_config.dcp_comm_backend == "a2a"
        )

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()

        # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
        self.is_aiter_triton_fp4_bmm_enabled = (
            rocm_aiter_ops.is_fp4bmm_enabled()
            and hasattr(self.kv_b_proj, "weight")
            and self.kv_b_proj.weight.dtype == torch.bfloat16
        )

        # Attributes for forward_impl method
        self._vllm_config = get_current_vllm_config()
        self._chunked_prefill_workspace_size: int | None = None
        self._decode_concat_quant_fp8_op = _DecodeConcatQuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            compile_native=True,
        )
        self._quant_fp8_op = QuantFP8(
            static=True,
            group_shape=GroupShape.PER_TENSOR,
            compile_native=True,
        )

    @property
    def chunked_prefill_workspace_size(self) -> int:
        if self._chunked_prefill_workspace_size is None:
            self._chunked_prefill_workspace_size = (
                MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
                    self._vllm_config
                )
            )
        return self._chunked_prefill_workspace_size

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                q,
                kv_c_normed,
                k_pe,
                _encode_layer_name(self.layer_name),
            )

        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata_raw = forward_context.attn_metadata
            attn_metadata: MLACommonMetadata
            if isinstance(attn_metadata_raw, dict):
                attn_metadata = attn_metadata_raw[self.layer_name]  # type: ignore[assignment]
            elif isinstance(attn_metadata_raw, list):
                # list[dict[str, AttentionMetadata]]: used in speculative decoding
                # where [0] is the base-model (non-speculative) metadata dict.
                attn_metadata = attn_metadata_raw[0][self.layer_name]  # type: ignore[assignment]
            else:
                attn_metadata = attn_metadata_raw
            self_kv_cache = self.kv_cache
            slot_mapping = forward_context.slot_mapping

            assert isinstance(slot_mapping, dict), (
                f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
            )
            self.impl.do_kv_cache_update(  # type: ignore[attr-defined]
                kv_c_normed,
                k_pe,
                self_kv_cache,
                slot_mapping.get(self.layer_name),
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.forward_impl(
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                output=output,
            )
            return output
        else:
            encoded = _encode_layer_name(self.layer_name)
            kv_cache_dummy_dep = torch.ops.vllm.unified_mla_kv_cache_update(
                kv_c_normed,
                k_pe,
                encoded,
                self.kv_cache_dtype,
                self._k_scale,
            )
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            torch.ops.vllm.unified_mla_attention_with_output(
                q,
                kv_c_normed,
                k_pe,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
            return output

    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
        quant_scale_ue8m0: bool | None = None,
        quant_col_major: bool | None = None,
        quant_tma_aligned: bool | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        quant_key = _detect_output_quant_key(
            output, output_scale, output_block_scale, self.num_heads * self.v_head_dim
        )
        if quant_key is not None:
            # The fusion pass has allocated output with quantized dtype
            # (FP8 or uint8 for FP4). We can't write into it directly,
            # so we swap in a temp buffer for computation, then quantize
            # into the real output at the end.
            # NOTE(carlyou): this is temporary until kernels support fp8 output
            quant_output = output
            output = torch.empty(
                output.shape[0],
                self.num_heads * self.v_head_dim,
                dtype=q.dtype,
                device=output.device,
            )

        if attn_metadata is None:
            # During the profile run try to simulate to worse case output size
            # for `self.kv_b_proj(kv_c_normed)` in `_compute_prefill_context`
            # since this can be large
            _ = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.num_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                ),
                device=k_c_normed.device,
                dtype=k_c_normed.dtype,
            )

            # The zero fill is required when used with DP + EP
            # to ensure all ranks within a DP group compute the
            # same expert outputs.
            if quant_key is not None:
                return quant_output.fill_(0)
            return output.fill_(0)

        if self.impl.dcp_world_size == -1:
            self.impl.dcp_world_size = get_dcp_group().world_size

        fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        if fp8_attention and self.kv_cache_dtype != "fp8_ds_mla":
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        # Sparse MLA impls only support forward_mqa (decode-style attention)
        is_sparse_impl = isinstance(self.impl, SparseMLAAttentionImpl)

        if is_sparse_impl:
            num_mqa_tokens = q.size(0)
            num_mha_tokens = 0
        else:
            assert (
                attn_metadata.num_decodes is not None
                and attn_metadata.num_prefills is not None
                and attn_metadata.num_decode_tokens is not None
            )
            num_mqa_tokens = attn_metadata.num_decode_tokens
            num_mha_tokens = q.size(0) - num_mqa_tokens

        if num_mha_tokens > 0:
            self.impl.forward_mha(  # type: ignore[attr-defined]
                q[num_mqa_tokens:],
                k_c_normed[num_mqa_tokens:],
                k_pe[num_mqa_tokens:],
                kv_cache,
                attn_metadata,
                self._k_scale,
                output=output[num_mqa_tokens:],
            )

        if num_mqa_tokens > 0:
            mqa_q = q[:num_mqa_tokens]
            mqa_output_slice = output[:num_mqa_tokens]

            mqa_q_nope, mqa_q_pe = mqa_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            mqa_q_nope = mqa_q_nope.transpose(0, 1)

            if self.q_pad_num_heads is not None:
                B, N, L = mqa_q_pe.shape
                mqa_pe_padded = mqa_q_pe.new_empty((B, self.q_pad_num_heads, L))
                mqa_pe_padded.resize_((B, N, L))
                mqa_pe_padded.copy_(mqa_q_pe)
                mqa_q_pe = mqa_pe_padded

            if self.is_aiter_triton_fp4_bmm_enabled:
                from aiter.ops.triton.batched_gemm_a16wfp4 import batched_gemm_a16wfp4

                mqa_ql_nope = batched_gemm_a16wfp4(
                    mqa_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    transpose_bm=True,
                    prequant=True,
                    y_scale=self._q_scale if fp8_attention else None,
                )
            elif self.is_aiter_triton_fp8_bmm_enabled:
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                mqa_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                    mqa_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    transpose_bm=True,
                )
            else:
                # Pads the head_dim if necessary (for the underlying kernel)
                N, B, P = mqa_q_nope.shape
                _, _, L = self.W_UK_T.shape

                if self.q_pad_num_heads is not None:
                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                    mqa_ql_nope.resize_((N, B, L))
                else:
                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                torch.bmm(mqa_q_nope, self.W_UK_T, out=mqa_ql_nope)

                # Convert from (N, B, L) to (B, N, L)
                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

            if fp8_attention and self.impl.supports_quant_query_input:
                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]
                assert mqa_ql_nope.shape[1] == mqa_q_pe.shape[1]
                mqa_q = self._decode_concat_quant_fp8_op(
                    mqa_ql_nope, mqa_q_pe, self._q_scale
                )
            else:
                mqa_q = (mqa_ql_nope, mqa_q_pe)
            if self.impl.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate mqa_ql_nope and mqa_q_pe -> (B, N, L + P)
                mqa_q = torch.cat(mqa_q, dim=-1)
                # mqa_q do allgather in head dim.
                mqa_q = get_dcp_group().all_gather(mqa_q, dim=1)

            # call decode attn
            if not is_sparse_impl:
                assert attn_metadata.decode is not None
            attn_out, lse = self.impl.forward_mqa(mqa_q, kv_cache, attn_metadata, self)  # type: ignore[attr-defined]

            # correct dcp attn_out with lse.
            if self.impl.dcp_world_size > 1:
                if self.dcp_a2a:
                    attn_out = dcp_a2a_lse_reduce(
                        attn_out,
                        lse,
                        get_dcp_group(),
                        is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                    )
                else:
                    attn_out = cp_lse_ag_out_rs(
                        attn_out,
                        lse,
                        get_dcp_group(),
                        is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                    )

            # v_up projection
            self._v_up_proj(attn_out, out=mqa_output_slice)

        if quant_key is not None:
            # Quantize the BF16 computation result into the quantized output
            actual = output[:num_actual_toks]
            if quant_key == kNvfp4Dynamic:
                # NVFP4: two FP4 values packed into one uint8
                assert output_block_scale is not None
                fp4_data, fp4_scales = ops.scaled_fp4_quant(actual, output_scale)
                quant_output[:num_actual_toks].copy_(fp4_data)
                output_block_scale[: fp4_scales.shape[0]].copy_(fp4_scales)
            elif quant_key in (kFp8Dynamic128Sym, kFp8Dynamic64Sym):
                # Per-group FP8
                assert output_block_scale is not None
                assert quant_group_size is not None, (
                    "Group FP8 output quant requested but "
                    "quant_group_size not passed through custom op"
                )
                finfo = torch.finfo(_FP8_DTYPE)
                torch.ops._C.per_token_group_fp8_quant(
                    actual,
                    quant_output[:num_actual_toks],
                    output_block_scale[:num_actual_toks],
                    quant_group_size,
                    1e-10,  # eps
                    finfo.min,
                    finfo.max,
                    quant_scale_ue8m0,
                    quant_col_major,
                    quant_tma_aligned,
                )
            elif quant_key == kFp8StaticTensorSym:
                # Static FP8 quantization
                fp8_data, _ = self._quant_fp8_op(actual, output_scale)
                quant_output[:num_actual_toks].copy_(fp8_data)
            else:
                raise ValueError(f"Unsupported quant_key: {quant_key}")
            return quant_output

        return output_padded

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(
            self.kv_b_proj, out_dtype=act_dtype
        ).T

        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        ), (
            f"{kv_b_proj_weight.shape=}, "
            f"{self.kv_lora_rank=}, "
            f"{self.num_heads=}, "
            f"{self.qk_nope_head_dim=}, "
            f"{self.v_head_dim=}"
        )
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # If kv_b_proj_weight is unquantized, quantize it to mxfp4 if supported
        if self.is_aiter_triton_fp4_bmm_enabled:
            from vllm.model_executor.layers.quantization.quark.utils import (
                quark_quantize_weight_to_mxfp4,
            )

            self.W_K, self.W_K_scale = quark_quantize_weight_to_mxfp4(W_UK)
            # Convert from (L, N, P) to (N, L, P)
            self.W_K = self.W_K.transpose(0, 1)
            self.W_K_scale = self.W_K_scale.transpose(0, 1)

            self.W_V, self.W_V_scale = quark_quantize_weight_to_mxfp4(
                W_UV.permute(1, 2, 0)
            )
        elif self.is_aiter_triton_fp8_bmm_enabled:
            W_K = W_UK.transpose(0, 1)  # 16 512 128
            W_V = W_UV.permute(1, 2, 0)  # 16 128 512
            self.W_K, self.W_K_scale = dynamic_per_batched_tensor_quant(
                W_K, dtype=current_platform.fp8_dtype()
            )
            self.W_V, self.W_V_scale = dynamic_per_batched_tensor_quant(
                W_V, dtype=current_platform.fp8_dtype()
            )

            # The kernel operates on non-padded inputs. Hence, pre-compiling
            # triton kernel to avoid runtime compilation for unseen batch sizes
            # Pre-compile for batch sizes 1 to 1024 to cover most use-cases.
            # On DS-R1, this step adds roughly 50s to the model loading time.
            max_batch_size = 1024  # [ToDo] Find the optimal upper limit
            pre_compilation_list = list(range(1, max_batch_size + 1))
            if is_global_first_rank():
                pre_compilation_list = tqdm(
                    pre_compilation_list,
                    desc="[Aiter Triton] Pre-compiling fp8 BMM kernel",
                    total=max_batch_size,
                )

            for m in pre_compilation_list:
                x = torch.empty(
                    (self.W_K.shape[0], m, self.W_K.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_K.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_K, self.W_K_scale, group_size=128, transpose_bm=True
                )

                x = torch.empty(
                    (self.W_V.shape[0], m, self.W_V.shape[2]),
                    dtype=torch.bfloat16,
                    device=self.W_V.device,
                )
                rocm_aiter_ops.triton_fp8_bmm(
                    x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True
                )
        else:
            # Convert from (L, N, V) to (N, L, V)
            self.W_UV = W_UV.transpose(0, 1)
            # Convert from (L, N, P) to (N, P, L)
            self.W_UK_T = W_UK.permute(1, 2, 0)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def calc_kv_scales(
        self, q: torch.Tensor, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> None:
        """Optional scale calculation for MLA inputs.

        Mirrors Attention.calc_kv_scales. Not all MLA backends require this
        """
        # Use safe defaults if ranges are not present
        q_range = getattr(self, "q_range", torch.tensor(1.0))
        k_range = getattr(self, "k_range", torch.tensor(1.0))
        v_range = getattr(self, "v_range", torch.tensor(1.0))

        self._q_scale.copy_(torch.abs(q).max() / q_range)
        # kv_c_normed is the compressed KV representation; use it for k/v
        kv_abs_max = torch.abs(kv_c_normed).max()
        self._k_scale.copy_(kv_abs_max / k_range)
        self._v_scale.copy_(kv_abs_max / v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        self.calculate_kv_scales = False

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, vllm_config.model_config
        )
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=kv_cache_dtype,
            cache_dtype_str=vllm_config.cache_config.cache_dtype,
        )

    def _v_up_proj(self, x: torch.Tensor, out: torch.Tensor):
        # Convert from (B, N, L) to (N, B, L)
        x = x.view(-1, self.num_heads, self.kv_lora_rank).transpose(0, 1)
        out = out.view(-1, self.num_heads, self.v_head_dim)
        if self.is_aiter_triton_fp4_bmm_enabled:
            out = rocm_aiter_ops.batched_gemm_a16wfp4(
                x,
                self.W_V,
                self.W_V_scale,
                out,
                transpose_bm=True,
                prequant=True,
                y_scale=None,
            )
            x = out.view(-1, self.num_heads * self.v_head_dim)
        elif self.is_aiter_triton_fp8_bmm_enabled:
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            x = rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
            )
        else:
            # Convert from (B, N * V) to (N, B, V)
            out = out.transpose(0, 1)

            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            torch.bmm(x, self.W_UV, out=out)  # Reuse "out" to make it "hot"

            # Convert from (N, B, V) to (B, N * V)
            out_new = out.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

            # Adjust output buffer shape back to the original (B, N * V)
            N, B, V = out.shape
            out.resize_((B, N * V))
            out.copy_(out_new)  # Copy result


def unified_mla_kv_cache_update(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            kv_c_normed,
            k_pe,
            kv_cache,
            layer_slot_mapping,
            kv_cache_dtype,
            k_scale,
        )

    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


def unified_mla_kv_cache_update_fake(
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(0, device=kv_c_normed.device, dtype=kv_c_normed.dtype)


direct_register_custom_op(
    op_name="unified_mla_kv_cache_update",
    op_func=unified_mla_kv_cache_update,
    fake_impl=unified_mla_kv_cache_update_fake,
)


@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, layer, kv_cache, _ = get_attention_context(layer_name)
    layer.forward_impl(
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
        quant_group_size=quant_group_size,
        quant_scale_ue8m0=quant_scale_ue8m0,
        quant_col_major=quant_col_major,
        quant_tma_aligned=quant_tma_aligned,
    )


def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
    quant_group_size: int | None = None,
    quant_scale_ue8m0: bool | None = None,
    quant_col_major: bool | None = None,
    quant_tma_aligned: bool | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_mla_attention_with_output",
    op_func=unified_mla_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_mla_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
    tags=(torch.Tag.flexible_layout,),
)


class QueryLenSupport(Enum):
    """Defines the level of query length support for an attention backend's
    decode pipeline.

    - SINGLE_ONLY: Decode pipeline only supports single-token queries
                   (query_len=1)
    - UNIFORM: Decode pipeline supports uniform multi-token queries
               (all requests must have same query_len > 1)
    - VARLEN: Decode pipeline supports variable-length queries
              (mixed query lengths in same batch)
    """

    SINGLE_ONLY = "single_only"
    UNIFORM = "uniform"
    VARLEN = "varlen"


def dynamic_per_batched_tensor_quant(
    x: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn
):
    DTYPE_MAX = torch.finfo(dtype).max
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-10)
    scale = DTYPE_MAX / amax
    x_scl_sat = (x * scale).clamp(min=-DTYPE_MAX, max=DTYPE_MAX)
    return x_scl_sat.to(dtype).contiguous(), scale.float().reciprocal()


@CustomOp.register(
    "mla_decode_concat_quant_fp8",
    dynamic_arg_dims={"decode_ql_nope": 0, "decode_q_pe": 0},
)
class _DecodeConcatQuantFP8(QuantFP8):
    """
    QuantFP8 variant that concatenates decode_ql_nope and decode_q_pe before
    quantization. When disabled, forward_native is compiled via torch.compile,
    fusing cat/reshape/quant/view together.
    """

    def _make_forward(quant_fn):  # noqa: N805
        """Factory to create forward methods that concat before quantization."""

        def forward(
            self,
            decode_ql_nope: torch.Tensor,
            decode_q_pe: torch.Tensor,
            scale: torch.Tensor,
            scale_ub: torch.Tensor | None = None,
        ) -> torch.Tensor:
            decode_q0 = torch.cat((decode_ql_nope, decode_q_pe), dim=-1)
            decode_q_flat = decode_q0.reshape(decode_q0.shape[0], -1)
            decode_q, _ = quant_fn(self, decode_q_flat, scale, scale_ub)
            return decode_q.view(decode_q0.shape)

        return forward

    forward_native = _make_forward(QuantFP8.forward_native)  # type: ignore[arg-type]
    forward_cuda = _make_forward(QuantFP8.forward_cuda)  # type: ignore[arg-type]
    forward_hip = _make_forward(QuantFP8.forward_hip)  # type: ignore[arg-type]


class MLACommonBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MLACommonMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            # MLA kernels require contiguous per-layer KV cache views.
            # Identity permutation keeps num_layers first in physical
            # layout, signaling cross-layer allocation is unsupported.
            return (0, 1, 2, 3)
        return (0, 1, 2)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [320, 576]

    @classmethod
    def is_mla(cls) -> bool:
        return True


@dataclass
class MLACommonPrefillMetadata:
    """Prefill Specific Metadata"""

    @dataclass
    class ChunkedContextMetadata:
        # New for MLA (compared to FlashAttention)
        # For handling chunked prefill
        cu_seq_lens: torch.Tensor
        starts: torch.Tensor
        seq_tot: list[int]
        max_seq_lens: list[int]
        seq_lens: torch.Tensor
        workspace: torch.Tensor
        token_to_seq: torch.Tensor
        chunk_total_token: list[int]

        # for mla DCP
        padded_local_chunk_seq_lens: list[list[int]] | None = None
        local_context_lens_allranks: list[list[int]] | None = None
        padded_local_cu_seq_lens: torch.Tensor | None = None
        cu_seq_lens_lst: list[list[int]] | None = None
        chunk_size: int | None = None
        prefill_tokens_with_context: int | None = None

    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    max_query_len: int
    chunked_context: ChunkedContextMetadata | None = None
    q_data_type: torch.dtype | None = None
    output_dtype: torch.dtype | None = None
    prefill_backend: MLAPrefillBackend | None = None


@dataclass
class MLACommonDecodeMetadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    dcp_tot_seq_lens: torch.Tensor | None


D = TypeVar("D", bound=MLACommonDecodeMetadata)


@dataclass
class MLACommonMetadata(AttentionMetadata, Generic[D]):
    """Metadata for MLACommon.

    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    # New for MLA (compared to FlashAttention)
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    num_prefills: int

    # The dimension of the attention heads
    head_dim: int | None = None

    prefill: MLACommonPrefillMetadata | None = None
    decode: D | None = None

    def __post_init__(self):
        if self.head_dim is not None and not MLACommonBackend.supports_head_size(
            self.head_dim
        ):
            raise ValueError(f"Head dimension {self.head_dim} is not supported by MLA.")


M = TypeVar("M", bound=MLACommonMetadata)
A = TypeVar("A", bound=AttentionMetadata)


@dataclass
class MLADims:
    q_lora_rank: int | None
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int


def get_mla_dims(model_config: ModelConfig) -> MLADims:
    hf_text_config = model_config.hf_text_config

    # Check if this is a DeepseekV4 config (uses unified head_dim + rope_head_dim)
    if hasattr(hf_text_config, "compress_ratios"):
        # DeepseekV4 style config: unified head_dim with rope_head_dim
        head_dim = hf_text_config.head_dim
        rope_head_dim = hf_text_config.qk_rope_head_dim
        return MLADims(
            q_lora_rank=hf_text_config.q_lora_rank,
            kv_lora_rank=head_dim,
            qk_nope_head_dim=head_dim - rope_head_dim,
            qk_rope_head_dim=rope_head_dim,
            v_head_dim=head_dim,
        )

    # DeepseekV2/V3 style config
    return MLADims(
        q_lora_rank=getattr(hf_text_config, "q_lora_rank", None),
        kv_lora_rank=hf_text_config.kv_lora_rank,
        qk_nope_head_dim=hf_text_config.qk_nope_head_dim,
        qk_rope_head_dim=hf_text_config.qk_rope_head_dim,
        v_head_dim=hf_text_config.v_head_dim,
    )


@functools.cache
def backend_supports_prefill_query_quantization() -> bool:
    """Check if the selected MLA prefill backend supports query quantization.

    Currently supported backends:
    - FlashInfer
    - TRT-LLM Ragged

    Not supported:
    - FlashAttention (FA3/FA4)
    - Non-GB200 devices (FP8 prefill requires device capability 100)
    """
    # FP8 prefill query quantization requires GB200 (device capability 100)
    # for the necessary FP8 kernels at the moment.
    if not current_platform.is_device_capability_family(100):
        return False

    from vllm.config import get_current_vllm_config
    from vllm.v1.attention.backends.mla.prefill import get_mla_prefill_backend

    vllm_config = get_current_vllm_config()
    backend_cls = get_mla_prefill_backend(vllm_config)
    return backend_cls.get_name() in (
        "FLASHINFER",
        "TRTLLM_RAGGED",
        "TOKENSPEED_MLA",
    )


class MLACommonMetadataBuilder(AttentionMetadataBuilder[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    # Defines the level of query length support for this backend.
    # - SINGLE_ONLY: Only single-token queries (no spec decode support)
    # - UNIFORM: Supports uniform multi-token queries (spec decode with uniform lengths)
    # - VARLEN: Supports variable-length queries (spec decode with mixed lengths)
    # If set to UNIFORM or VARLEN, this will increase `reorder_batch_threshold` when
    # speculative decoding is enabled.
    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.SINGLE_ONLY

    # The threshold for reordering the batch into decode and prefill requests.
    # If > 1, the batch will be reordered such that requests with
    # query length <= threshold are classified as decode requests.
    # Use `query_len_support` (above) to set this automatically
    # when speculative decoding is enabled.
    reorder_batch_threshold: int = 1

    @staticmethod
    def determine_chunked_prefill_workspace_size(vllm_config: VllmConfig) -> int:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        chunked_prefill_workspace_size = min(
            # Try for 8 full length request or at least 4 pages per-request
            max(
                8 * model_config.max_model_len,
                4 * scheduler_config.max_num_seqs * cache_config.block_size,
            ),
            # For long-context models try not to over-allocate limiting
            # kv-cache space, limiting it to 64k tokens,
            # which would result in the workspace being:
            #   2*(576)*(64*1024) = 144mb
            # (assuming 576 MLA head dim, and fp16)
            # which would result in up-projected context being
            #   2*(192*128)*(64*1024) = 3gb
            # (assuming 192 QK head dim, 128 heads, and fp16)
            64 * 1024,
        )

        # Enforce that we enough for at least 1 page per request
        chunked_prefill_workspace_size = max(
            chunked_prefill_workspace_size,
            scheduler_config.max_num_seqs * cache_config.block_size,
        )

        return chunked_prefill_workspace_size

    @staticmethod
    def determine_prefill_query_data_type(
        vllm_config: VllmConfig,
        model_dtype: torch.dtype,
    ) -> torch.dtype:
        """
        Determine the query data type for prefill queries.
        Return FP8 dtype if cache is FP8 and prefill query quantization
        is enabled, else model dtype.
        """
        use_fp8 = (
            is_quantized_kv_cache(vllm_config.cache_config.cache_dtype)
            and vllm_config.attention_config.use_prefill_query_quantization
            and backend_supports_prefill_query_quantization()
        )

        if use_fp8:
            fp8_dtype = current_platform.fp8_dtype()
            logger.info_once("FP8 prefill attention enabled: query data type is FP8")
            return fp8_dtype
        elif vllm_config.attention_config.use_prefill_query_quantization:
            logger.info_once(
                "Unable to perform FP8 prefill attention when"
                " use_prefill_query_quantization is enabled. Please"
                " ensure that --kv-cache-dtype is set to fp8 and your prefill"
                " backend is compatible with FP8 attention.",
            )
            return model_dtype
        elif (
            is_quantized_kv_cache(vllm_config.cache_config.cache_dtype)
            and backend_supports_prefill_query_quantization()
        ):
            logger.warning_once(
                "FP8 KV cache is enabled but prefill queries are not "
                "quantized to FP8. For long-context workloads (ISL >= 4K), "
                "enabling FP8 prefill attention can significantly optimize "
                "prefill latency. To enable, add: "
                '--attention-config \'{"use_prefill_query_quantization"'
                ": true}'",
            )

        return model_dtype

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
        metadata_cls: type[M] | None = None,
        supports_dcp_with_varlen: bool = False,
    ):
        self.metadata_cls = (
            metadata_cls if metadata_cls is not None else MLACommonMetadata
        )
        self.kv_cache_spec = kv_cache_spec
        self.model_config = vllm_config.model_config
        parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.vllm_config = vllm_config
        self.device = device

        self.num_heads = self.model_config.get_num_attention_heads(parallel_config)
        self.mla_dims = get_mla_dims(self.model_config)
        self.aot_schedule = current_platform.is_cuda()

        self.kv_cache_spec = kv_cache_spec
        self.q_data_type = self.determine_prefill_query_data_type(
            vllm_config, self.model_config.dtype
        )

        try:
            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0
        self.dcp_local_block_size = parallel_config.cp_kv_cache_interleave_size
        self.dcp_virtual_block_size = self.dcp_local_block_size * self.dcp_world_size
        self.cp_kv_cache_interleave_size = parallel_config.cp_kv_cache_interleave_size

        # Don't try to access the runner on AMD
        if self.aot_schedule:
            self.page_size = self.kv_cache_spec.block_size

        self.chunked_prefill_workspace_size = (
            self.determine_chunked_prefill_workspace_size(vllm_config)
        )

        if self.dcp_world_size > 1:
            # Note(hc): The local kvcache is incomplete when DCP is triggered,
            # an additional kvcache allgather across the DCP group is therefore
            # required, so the workspace has to be enlarged by 1/DCP relative
            # to the original TP allocation.
            assert self.chunked_prefill_workspace_size % self.dcp_world_size == 0
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size
                    + self.chunked_prefill_workspace_size // self.dcp_world_size,
                    self.model_config.get_head_size(),
                ),
                dtype=self.model_config.dtype,
                device=device,
            )
        else:
            self.chunked_prefill_workspace = torch.empty(
                (
                    self.chunked_prefill_workspace_size,
                    self.model_config.get_head_size(),
                ),
                dtype=self.q_data_type,
                device=device,
            )

        self._prefill_backend = self.compilation_config.static_forward_context[
            layer_names[0]
        ].prefill_backend

        supports_spec_decode = self.query_len_support != QueryLenSupport.SINGLE_ONLY
        self._init_reorder_batch_threshold(
            self.reorder_batch_threshold, supports_spec_decode, supports_dcp_with_varlen
        )

        if self.query_len_support == QueryLenSupport.SINGLE_ONLY:
            assert self.reorder_batch_threshold == 1, (
                f"reorder_batch_threshold must be 1 when query_len_support is "
                f"SINGLE_ONLY, got {self.reorder_batch_threshold}"
            )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_device: torch.Tensor,
        max_seq_len: int,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> MLACommonDecodeMetadata:
        return MLACommonDecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
            dcp_tot_seq_lens=dcp_tot_seq_lens_device,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MLA.
        """
        m = common_attn_metadata
        assert m.num_reqs <= (m.num_actual_tokens * self.reorder_batch_threshold), (
            "MLA only supports decode-only full CUDAGraph capture. "
            "Make sure all cudagraph capture sizes <= max_num_seq."
        )

        assert m.max_query_len <= self.reorder_batch_threshold  # decode only

        return self.build(0, m)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> M:
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.device
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        query_start_loc = common_attn_metadata.query_start_loc
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens = common_attn_metadata.seq_lens
        dcp_local_seq_lens = common_attn_metadata.dcp_local_seq_lens

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.reorder_batch_threshold,
                require_uniform=(self.query_len_support != QueryLenSupport.VARLEN),
            )
        )

        assert num_decodes + num_prefills == num_reqs
        assert num_decode_tokens + num_prefill_tokens == num_tokens

        prefill_metadata = None
        if num_prefills > 0:
            reqs_start = num_decodes  # prefill_start

            # Upper bound is exact for prefill rows (no D2H sync).
            seq_lens_cpu = common_attn_metadata.seq_lens_cpu_upper_bound
            assert seq_lens_cpu is not None
            prefill_query_lens_cpu = (
                query_start_loc_cpu[reqs_start + 1 : num_reqs + 1]
                - query_start_loc_cpu[reqs_start:num_reqs]
            )
            context_lens_cpu = (
                seq_lens_cpu[reqs_start:num_reqs] - prefill_query_lens_cpu
            )
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = (
                query_start_loc[reqs_start:] - query_start_loc[reqs_start]
            )
            prefill_query_start_loc_cpu = (
                query_start_loc_cpu[reqs_start:] - query_start_loc_cpu[reqs_start]
            )

            chunked_context_metadata = None
            if max_context_len_cpu > 0:
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                max_context_chunk = (
                    self.chunked_prefill_workspace_size // num_prefills_with_context_cpu
                )

                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_and_maybe_dequant_cache` kernel
                    # cannot handle `context_chunk_starts` that are not aligned
                    # to page_size
                    max_context_chunk = round_down(max_context_chunk, self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like
                #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = (
                    torch.arange(num_chunks, dtype=torch.int32)
                    .unsqueeze(1)
                    .expand(-1, num_prefills)
                    * max_context_chunk
                )
                chunk_ends = torch.min(
                    context_lens_cpu.unsqueeze(0), chunk_starts + max_context_chunk
                )
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(
                    num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                )
                torch.cumsum(
                    chunk_seq_lens, dim=1, out=cu_seq_lens_cpu[:, 1:], dtype=torch.int32
                )
                chunk_total_token = cu_seq_lens_cpu[:, -1]

                max_token_num_over_chunk = chunk_total_token.max().item()
                token_to_seq_tensor_cpu = torch.zeros(
                    [num_chunks, max_token_num_over_chunk], dtype=torch.int32
                )
                range_idx = torch.arange(num_prefills, dtype=torch.int32)
                for i in range(num_chunks):
                    chunk_token_to_seq_tensor = torch.repeat_interleave(
                        range_idx, chunk_seq_lens[i]
                    )
                    chunk_len = chunk_token_to_seq_tensor.shape[0]
                    token_to_seq_tensor_cpu[i, :chunk_len] = chunk_token_to_seq_tensor

                if self.dcp_world_size > 1:
                    local_context_lens_allranks = get_dcp_local_seq_lens(
                        context_lens_cpu,
                        self.dcp_world_size,
                        None,
                        self.dcp_local_block_size,
                    )
                    # Note(qcs): The max local context lengths
                    # padded to `dcp_local_block_size`.
                    padded_local_context_lens_cpu: torch.Tensor = (
                        cdiv(
                            context_lens_cpu,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    # Note(hc): The above max_context_chunk already enforces
                    # block_size alignment, DCP just need the block_size can
                    # be divisible by dcp_world_size, because DCP use
                    # cp_gather_cache which not require `cp_chunk_starts`
                    # aligned to page_size.
                    assert max_context_chunk % self.dcp_world_size == 0
                    padded_local_max_context_chunk_across_ranks = (
                        cdiv(
                            max_context_chunk,
                            self.dcp_virtual_block_size,
                        )
                        * self.dcp_local_block_size
                    )
                    local_chunk_starts = (
                        torch.arange(num_chunks, dtype=torch.int32)
                        .unsqueeze(1)
                        .expand(-1, num_prefills)
                        * padded_local_max_context_chunk_across_ranks
                    )
                    local_chunk_ends = torch.min(
                        padded_local_context_lens_cpu.unsqueeze(0),
                        local_chunk_starts
                        + padded_local_max_context_chunk_across_ranks,
                    )
                    padded_local_chunk_seq_lens = (
                        local_chunk_ends - local_chunk_starts
                    ).clamp(min=0)

                    padded_local_cu_chunk_seq_lens_cpu = torch.zeros(
                        num_chunks, num_prefills + 1, dtype=torch.int32, pin_memory=True
                    )
                    torch.cumsum(
                        padded_local_chunk_seq_lens,
                        dim=1,
                        out=padded_local_cu_chunk_seq_lens_cpu[:, 1:],
                        dtype=torch.int32,
                    )

                prefill_tokens_with_context = None
                if num_prefills_with_context_cpu > 0:
                    prefill_tokens_with_context = prefill_query_start_loc_cpu[
                        num_prefills_with_context_cpu
                    ].item()
                _ChunkedMetadata = MLACommonPrefillMetadata.ChunkedContextMetadata
                if self.dcp_world_size > 1:
                    chunked_context_metadata = _ChunkedMetadata(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=local_chunk_starts.to(device, non_blocking=True),
                        seq_tot=padded_local_chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token.tolist(),
                        workspace=self.chunked_prefill_workspace,
                        padded_local_chunk_seq_lens=padded_local_chunk_seq_lens.tolist(),
                        local_context_lens_allranks=local_context_lens_allranks.tolist(),
                        padded_local_cu_seq_lens=padded_local_cu_chunk_seq_lens_cpu.to(
                            device, non_blocking=True
                        ),
                        cu_seq_lens_lst=cu_seq_lens_cpu.tolist(),
                        chunk_size=padded_local_max_context_chunk_across_ranks,
                        prefill_tokens_with_context=prefill_tokens_with_context,
                    )
                else:
                    chunked_context_metadata = _ChunkedMetadata(
                        cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                        starts=chunk_starts.to(device, non_blocking=True),
                        seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                        max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                        seq_lens=chunk_seq_lens,
                        token_to_seq=token_to_seq_tensor_cpu.to(
                            device, non_blocking=True
                        ),
                        chunk_total_token=chunk_total_token,
                        workspace=self.chunked_prefill_workspace,
                        prefill_tokens_with_context=prefill_tokens_with_context,
                    )

                assert (
                    max(chunked_context_metadata.max_seq_lens)
                    <= self.chunked_prefill_workspace_size
                )

            prefill_metadata = MLACommonPrefillMetadata(
                block_table=block_table_tensor[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
                output_dtype=self.model_config.dtype,
                q_data_type=self.q_data_type,
                prefill_backend=self._prefill_backend,
            )

            self._prefill_backend.prepare_metadata(prefill_metadata)

        decode_metadata = None
        if num_decodes > 0:
            dcp_tot_seq_lens_device = None
            if self.dcp_world_size > 1:
                dcp_tot_seq_lens_device = seq_lens[:num_decodes]
                seq_lens = dcp_local_seq_lens

                # After DCP distribution, the maximum number of tokens for any rank is
                # ceil(L / (N * I)) * I, where L is max_seq_len, N is dcp_world_size,
                # and I is cp_kv_cache_interleave_size.
                # This eliminates GPU->CPU sync while minimizing workspace
                # over-allocation.
                num_partitions = self.dcp_world_size * self.cp_kv_cache_interleave_size
                max_seq_len = (
                    (max_seq_len + num_partitions - 1) // num_partitions
                ) * self.cp_kv_cache_interleave_size

            decode_metadata = self._build_decode(
                block_table_tensor=block_table_tensor[:num_decodes, ...],
                seq_lens_device=seq_lens[:num_decodes],
                max_seq_len=max_seq_len,
                query_start_loc_cpu=query_start_loc_cpu[: num_decodes + 1],
                query_start_loc_device=query_start_loc[: num_decodes + 1],
                num_decode_tokens=num_decode_tokens,
                dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
            )

        attn_metadata = self.metadata_cls(
            num_reqs=common_attn_metadata.num_reqs,
            max_query_len=common_attn_metadata.max_query_len,
            max_seq_len=max_seq_len,
            num_actual_tokens=num_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )

        return attn_metadata  # type: ignore[return-value]


def reorg_kvcache(
    allgatered_kv_c_normed: torch.Tensor,
    allgatered_k_pe: torch.Tensor,
    padded_local_chunk_seq_lens_lst: list[int],
    local_context_lens_allranks: list[list[int]],
    sum_seq_len: int,
    max_seq_len: int,
    chunk_size: int,
    chunk_idx: int,
    toks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    reorg and unpad kvcache after cp local gather to tp layout for attn kernel.
    e.g.
    allgatered_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T1_0, T1_1, ...,
                              T0_4, T0_5, pad, pad, T1_2, pad, ...]
    -> reorganized_kv_c_normed = [T0_0, T0_1, T0_2, T0_3, T0_4, T0_5,
                                  T1_0, T1_1, T1_2, ...]
    Args:
        padded_local_chunk_seq_lens_lst: local chunk context lengths
            under current CP rank.
        local_context_lens_allranks: local context lengths on each CP rank.
        sum_seq_len: the sum of cp_chunk_seq_lens_lst.
        max_seq_len: the max value of cp_chunk_seq_lens_lst.
        chunk_size: the local padded max context chunk from
            chunked_context_metadata building.
        chunk_idx: chunk idx of chunked_prefill.
        toks: the number of tokens for local gather cache.
    """
    kv_c_segments = []
    k_pe_segments = []
    src_token_idx = 0
    max_seq_len_check = 0
    for padded_local_chunk_seq_len, local_context_lens in zip(
        padded_local_chunk_seq_lens_lst, local_context_lens_allranks
    ):
        cur_seq_len = 0
        for rank, local_context_len in enumerate(local_context_lens):
            # Note(qcs): We split the context into multiple chunks,
            # depending on the size of the workspace.
            # local_context in dcp0:   |-----------------|
            # local_context in dcp1:   |--------------|
            # n*padded_local_chunk:    |-----|-----|-----|
            # local_chunk_len in dcp1: |-----|-----|--|
            # so we need update the last chunk length in dcp1.
            local_chunk_len = min(
                max(0, local_context_len - chunk_idx * chunk_size),
                padded_local_chunk_seq_len,
            )
            if local_chunk_len != 0:
                kv_c_segment = allgatered_kv_c_normed[
                    rank * toks + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                k_pe_segment = allgatered_k_pe[
                    rank * toks + src_token_idx : rank * toks
                    + src_token_idx
                    + local_chunk_len
                ]
                kv_c_segments.append(kv_c_segment)
                k_pe_segments.append(k_pe_segment)
                cur_seq_len += local_chunk_len
        max_seq_len_check = max(max_seq_len_check, cur_seq_len)
        src_token_idx += padded_local_chunk_seq_len
    reorganized_kv_c_normed = torch.cat(kv_c_segments, dim=0)
    reorganized_k_pe = torch.cat(k_pe_segments, dim=0)
    assert reorganized_kv_c_normed.shape[0] == sum_seq_len
    assert reorganized_k_pe.shape[0] == sum_seq_len
    assert max_seq_len_check == max_seq_len
    return reorganized_kv_c_normed, reorganized_k_pe


class MLACommonImpl(MLAAttentionImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def fused_output_quant_supported(self, quant_key):
        return quant_key in (
            kFp8StaticTensorSym,
            kNvfp4Dynamic,
            kFp8Dynamic128Sym,
            kFp8Dynamic64Sym,
        )

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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        # DSV3.2 MLA Specific Arguments
        indexer: object | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported for MLA")

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads
        self.supports_quant_query_input = True
        self.is_aiter_triton_fp8_bmm_enabled = rocm_aiter_ops.is_fp8bmm_enabled()

        # Use flashinfer's optimized concat_mla_k kernel when available.
        # The kernel is optimized for DeepSeek V3 dimensions:
        # num_heads=128, nope_dim=128, rope_dim=64
        self._use_flashinfer_concat_mla_k = (
            has_flashinfer()
            and (self.num_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        )

        self.dcp_world_size: int = -1

        self.cp_kv_cache_interleave_size: int = (
            get_current_vllm_config().parallel_config.cp_kv_cache_interleave_size
        )

    def _concat_k_nope_k_pe(
        self, k_nope: torch.Tensor, k_pe: torch.Tensor
    ) -> torch.Tensor:
        """
        Efficiently concatenate k_nope and k_pe tensors along the last dimension.

        This function avoids the performance penalty of torch.cat with expanded
        non-contiguous tensors by pre-allocating the output and using direct copies.

        Args:
            k_nope: Tensor of shape [..., nope_dim]
            k_pe: Tensor to broadcast and concatenate, typically shape [..., 1, pe_dim]
                or [..., pe_dim]

        Returns:
            Tensor of shape [..., nope_dim + pe_dim]
        """
        k = torch.empty(
            (*k_nope.shape[:-1], k_nope.shape[-1] + k_pe.shape[-1]),
            dtype=k_nope.dtype,
            device=k_nope.device,
        )

        if self._use_flashinfer_concat_mla_k:
            torch.ops.vllm.flashinfer_concat_mla_k(k, k_nope, k_pe)
        else:
            # Fallback: Direct copies with efficient broadcasting
            k[..., : k_nope.shape[-1]] = k_nope
            k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None

        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        output = None
        merge_output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            if not use_fp8_prefill:
                ops.gather_and_maybe_dequant_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    token_to_seq=prefill_metadata.chunked_context.token_to_seq[i],
                    num_tokens=prefill_metadata.chunked_context.chunk_total_token[i],
                    kv_cache_dtype=self.kv_cache_dtype,
                    scale=k_scale,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )
            else:
                # FP8 path: gather cache without dequantization
                ops.cp_gather_cache(
                    src_cache=kv_c_and_k_pe_cache,
                    dst=workspace,
                    block_table=prefill_metadata.block_table,
                    cu_seq_lens=prefill_metadata.chunked_context.cu_seq_lens[i],
                    batch_size=attn_metadata.num_prefills,
                    seq_starts=prefill_metadata.chunked_context.starts[i],
                )

            # Extract kv_c_normed from workspace
            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            # When FP8 weights are used without FP8 prefill, kv_b_proj expects
            # model dtype input and will quantize internally.
            # For quantized layers (AWQ/GPTQ) that lack a .weight attribute,
            # use params_dtype which is the expected input dtype.
            _kv_b_proj_w_dtype = (
                self.kv_b_proj.weight.dtype
                if hasattr(self.kv_b_proj, "weight")
                else self.kv_b_proj.params_dtype
            )
            # For NVFP4, weights are packed uint8 — keep input in model dtype
            # since the NVFP4 linear layer quantizes internally.
            if (
                use_fp8_prefill or _kv_b_proj_w_dtype != current_platform.fp8_dtype()
            ) and _kv_b_proj_w_dtype != torch.uint8:
                kv_c_normed = kv_c_normed.to(self.kv_b_proj.weight.dtype)

            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)
            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )

            # To Do: Use epilogue of kv_b_proj to generate fp8 kv_nope.
            if use_fp8_prefill:
                kv_nope = kv_nope.to(prefill_metadata.q_data_type)
                k_pe = k_pe.to(prefill_metadata.q_data_type)
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk_idx=i,
                    q=q,
                    k=k,
                    v=v,
                )
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                if merge_output is None:
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                merge_attn_states(
                    output=merge_output,
                    output_lse=merge_output_lse,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        return output, output_lse

    def _context_parallel_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        dcp_world_size: int,
    ):
        assert k_scale is None, "DCP not support scaled kvcache now."
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.padded_local_chunk_seq_lens is not None
        assert prefill_metadata.chunked_context.local_context_lens_allranks is not None
        assert prefill_metadata.chunked_context.padded_local_cu_seq_lens is not None
        assert prefill_metadata.chunked_context.cu_seq_lens_lst is not None
        assert prefill_metadata.chunked_context.chunk_size is not None

        output = None
        merge_output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace

        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
            ops.cp_gather_cache(
                src_cache=kv_c_and_k_pe_cache,
                dst=workspace,
                block_table=prefill_metadata.block_table,
                cu_seq_lens=prefill_metadata.chunked_context.padded_local_cu_seq_lens[
                    i
                ],
                batch_size=attn_metadata.num_prefills,
                seq_starts=prefill_metadata.chunked_context.starts[i],
            )
            # workspace
            # |------- N tokens --------|--------- N*dcp_size tokens ----------|
            # |<- use for local_gather ->|<--------- use for allgather -------->|
            allgather_offset = workspace.shape[0] // (dcp_world_size + 1)
            assert allgather_offset * (dcp_world_size + 1) == workspace.shape[0]
            assert toks <= allgather_offset
            local_gathered_kvcache = workspace[:toks]
            cur_allgather_workspace = workspace[
                allgather_offset : allgather_offset * (1 + dcp_world_size)
            ]
            assert toks * dcp_world_size <= cur_allgather_workspace.shape[0]
            cur_allgather_kvcache = cur_allgather_workspace[: toks * dcp_world_size]
            cur_allgather_kvcache.copy_(
                get_dcp_group().all_gather(local_gathered_kvcache, dim=0)
            )
            assert (
                cur_allgather_kvcache.shape[-1]
                == self.kv_lora_rank + self.qk_rope_head_dim
            )
            allgatered_kv_c_normed, allgatered_k_pe = cur_allgather_kvcache.unsqueeze(
                1
            ).split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

            kv_c_normed, k_pe = reorg_kvcache(
                allgatered_kv_c_normed,
                allgatered_k_pe,
                padded_local_chunk_seq_lens_lst=prefill_metadata.chunked_context.padded_local_chunk_seq_lens[
                    i
                ],
                local_context_lens_allranks=prefill_metadata.chunked_context.local_context_lens_allranks,
                sum_seq_len=prefill_metadata.chunked_context.cu_seq_lens_lst[i][-1],
                max_seq_len=prefill_metadata.chunked_context.max_seq_lens[i],
                chunk_size=prefill_metadata.chunked_context.chunk_size,
                chunk_idx=i,
                toks=toks,
            )

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            attn_output, attn_softmax_lse = (
                prefill_metadata.prefill_backend.run_prefill_context_chunk(
                    chunk_idx=i,
                    q=q,
                    k=k,
                    v=v,
                )
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                if merge_output is None:
                    merge_output = torch.empty_like(output)
                    merge_output_lse = torch.empty_like(output_lse)
                merge_attn_states(
                    output=merge_output,
                    output_lse=merge_output_lse,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output, merge_output = merge_output, output
                output_lse, merge_output_lse = merge_output_lse, output_lse

        return output, output_lse

    def forward_mha(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert attn_metadata.prefill is not None
        assert self.dcp_world_size != -1

        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.prefill_backend is not None
        use_fp8_prefill = prefill_metadata.q_data_type == current_platform.fp8_dtype()

        # Convert q to FP8 if FP8 prefill attention is enabled
        if use_fp8_prefill:
            q = q.to(prefill_metadata.q_data_type)

        has_context = prefill_metadata.chunked_context is not None

        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        if use_fp8_prefill:
            k = k.to(prefill_metadata.q_data_type)
            v = v.to(prefill_metadata.q_data_type)

        output_prefill = prefill_metadata.prefill_backend.run_prefill_new_tokens(
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
            assert prefill_metadata.chunked_context is not None
            suffix_output, suffix_lse = output_prefill
            if self.dcp_world_size > 1:
                context_output, context_lse = (
                    self._context_parallel_compute_prefill_context(
                        q,
                        kv_c_and_k_pe_cache,
                        attn_metadata,
                        k_scale=None,
                        dcp_world_size=self.dcp_world_size,
                    )
                )
            else:
                context_output, context_lse = self._compute_prefill_context(
                    q, kv_c_and_k_pe_cache, attn_metadata, k_scale
                )

            output = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
                prefill_tokens_with_context=prefill_metadata.chunked_context.prefill_tokens_with_context,
            )
        else:
            assert isinstance(output_prefill, torch.Tensor)
            output_prefill = output_prefill.flatten(start_dim=-2)
            output.copy_(output_prefill)

    @abstractmethod
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: M,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        raise NotImplementedError
