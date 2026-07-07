# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HPC fused RoPE + QK-Norm + KV-Cache-Write (+ optional FP8 Q quant).

Decoupled from HpcAttentionImpl; extra params are passed via layer attrs.
"""

from __future__ import annotations

import importlib.util
from enum import IntEnum
from typing import Any

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.hpc.hpc_module import HpcModule
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.hpc_attn import HpcAttnMetadata
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

_hpc_rope_norm_instances: dict[str, HpcRopeNorm] = {}


class QkNormPolicy(IntEnum):
    """Order of QK-RMSNorm relative to RoPE in the fused HPC rope_norm kernel.

    The values are part of the HPC kernel ABI (passed through as ints), so they
    must stay in sync with the kernel's expectations.
    """

    # No QK-Norm: apply RoPE only.
    NONE = 0
    # Apply RoPE first, then QK-RMSNorm.
    ROPE_THEN_NORM = 1
    # Apply QK-RMSNorm first, then RoPE (e.g. HunYuan V3).
    NORM_THEN_ROPE = 2


def hpc_rope_norm_forward(
    qkv: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Top-level custom op: RoPE + QK-Norm + KV-Cache-Write + FP8 Q quant.

    Fully opaque to torch.compile (dynamo).
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata: Any = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    if attn_metadata is None:
        output.zero_()
        return

    attn_layer = forward_context.no_compile_layers[layer_name]
    # bind_kv_cache stores the per-layer KV cache as a single 5D tensor
    # (num_blocks, 2, block_size, num_kv_heads, head_size), so use it directly.
    kv_cache = attn_layer.kv_cache

    if kv_cache.numel() == 0:
        output.zero_()
        return

    assert kv_cache.dim() == 5, (
        f"Expected kv_cache to have 5 dims, got {tuple(kv_cache.shape)}"
    )

    rope_norm = _hpc_rope_norm_instances[layer_name]
    rope_norm._forward_impl(qkv, kv_cache, attn_metadata, attn_layer, output)


def hpc_rope_norm_forward_fake(
    qkv: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake impl for torch.compile trace; output is a mutated arg."""
    return


direct_register_custom_op(
    op_name="hpc_rope_norm_forward",
    op_func=hpc_rope_norm_forward,
    mutates_args=["output"],
    fake_impl=hpc_rope_norm_forward_fake,
)


@CustomOp.register("hpc_rope_norm")
class HpcRopeNorm(CustomOp, HpcModule):
    """HPC fused RoPE + QK-Norm + KV-Cache-Write (+ optional FP8 Q quant).

    Registered as a sub-module in model layers (e.g. HunYuanAttention).
    Norm weights are extracted from fallback norm modules via
    process_weights_after_loading() after all weights are loaded.

    forward() is dispatched by CustomOp framework:
    - In compiled mode: forward_cuda() calls torch.ops.vllm.hpc_rope_norm_forward
      as a splitting point — internal Python control flow is opaque
      to torch.compile and not captured by CUDA Graph.
    - In eager/native mode: forward_native() falls back to forward_cuda().
    """

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        cos_sin_cache: torch.Tensor,
        use_qk_norm: bool,
        fallback_qnorm: torch.nn.Module | None,
        fallback_knorm: torch.nn.Module | None,
        kv_cache_dtype: str,
        layer_name: str,
        qk_norm_policy: QkNormPolicy = QkNormPolicy.ROPE_THEN_NORM,
    ) -> None:
        super().__init__()
        if importlib.util.find_spec("hpc") is None:
            raise ImportError(
                "HPCRopeNorm requires the hpc module to be installed. "
                "Please install it from https://github.com/Tencent/hpc-ops"
            )

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.use_qk_norm = use_qk_norm

        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim

        # Register as a non-persistent buffer so it participates in sleep
        # level-2 save/restore (CuMemAllocator) but is excluded from the
        # checkpoint state_dict.
        self.register_buffer("cos_sin_cache", cos_sin_cache.float(), persistent=False)

        self.fallback_qnorm = fallback_qnorm
        self.fallback_knorm = fallback_knorm

        self.head_per_group = num_heads // num_kv_heads

        # Pre-allocate norm weight tensors as Parameters so they are tracked by
        # CuMemAllocator (for sleep/wake_up) and have stable addresses for CUDA
        # Graph replay. process_weights_after_loading() updates them inplace via
        # copy_() so refit does not invalidate captured graph tensor pointers.
        # Shape is [head_dim] to match the HPC kernel's q/k_norm_weight layout.
        if use_qk_norm and fallback_qnorm is not None:
            self.qnorm_weight: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(head_dim, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.qnorm_weight = None
        if use_qk_norm and fallback_knorm is not None:
            self.knorm_weight: torch.nn.Parameter | None = torch.nn.Parameter(
                torch.empty(head_dim, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.knorm_weight = None

        self.use_fp8 = "fp8" in kv_cache_dtype
        # The RMSNorm/RoPE ordering is model dependent (e.g. HunYuan V3 applies
        # QK-Norm before RoPE -> NORM_THEN_ROPE), so it is supplied by the
        # caller. When QK-Norm is disabled the policy is forced to NONE.
        self.qk_norm_policy = qk_norm_policy if use_qk_norm else QkNormPolicy.NONE

        # Register layer_name + add self to the global instance registry so the
        # module-level custom op (hpc_rope_norm_forward) can route back here.
        self.layer_name: str | None = None
        self.register_layer_name(layer_name)

        import hpc

        if self.use_fp8:
            self._quant_type = (
                hpc.QuantType.QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR.value
            )
        else:
            self._quant_type = None

    @classmethod
    def support(
        cls,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        kv_cache_dtype: str,
    ) -> bool:
        """Check whether HpcRopeNorm is supported for the given config."""
        # HpcRopeNorm is only enabled together with the HPC attention backend.
        vllm_config = get_current_vllm_config_or_none()
        if (
            vllm_config is None
            or vllm_config.attention_config.backend != AttentionBackendEnum.HPC_ATTN
        ):
            return False

        if kv_cache_dtype not in ("fp8_e4m3", "auto"):
            logger.warning_once(
                f"hpc rope_norm not support kv_cache_dtype:{kv_cache_dtype}, "
                "only support fp8_e4m3, bfloat16"
            )
            return False

        if head_dim not in (128,):
            logger.warning_once("hpc rope_norm only support head_dim == 128.")
            return False

        head_per_group = num_heads // num_kv_heads
        if head_per_group not in (4, 8):
            logger.warning_once("hpc rope_norm only support head_per_group in [4, 8].")
            return False

        logger.info_once("enable hpc rope_norm")
        return True

    def process_weights_after_loading(self, model: torch.nn.Module = None) -> None:
        """Copy norm weights (float32) from fallback norm modules inplace.

        Uses copy_() to preserve tensor addresses for CUDA Graph / refit
        compatibility. Called by the model's load_weights() after all weights
        are loaded (and generically from the model loader for DummyModelLoader
        / sleep-wake_up reload paths).
        """
        if self.use_qk_norm:
            if self.fallback_qnorm is not None and self.qnorm_weight is not None:
                self.qnorm_weight.data.copy_(self.fallback_qnorm.weight.data.float())
            if self.fallback_knorm is not None and self.knorm_weight is not None:
                self.knorm_weight.data.copy_(self.fallback_knorm.weight.data.float())

    def register_layer_name(self, layer_name: str) -> None:
        """Register layer_name and add self to the global registry.

        The global registry is needed because the bottom-level torch op
        (hpc_rope_norm_forward) is a module-level function and needs to
        route back to the correct instance via layer_name.
        """
        self.layer_name = layer_name
        _hpc_rope_norm_instances[layer_name] = self
        logger.debug(
            "[rope_norm] registered HpcRopeNorm for layer: %s",
            layer_name,
        )

    def forward_native(
        self,
        qkv: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Native fallback path: delegates to forward_cuda().

        For now, the default native path will use CUDA backend path.
        Other platforms may override via OOT registration.
        """
        return self.forward_cuda(qkv, layer_name)

    def forward_cuda(
        self,
        qkv: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """CUDA path: invoke the torch custom op as a compile splitting point."""
        num_tokens = qkv.shape[0]
        output = torch.empty(
            (num_tokens, self.num_heads, self.head_dim),
            dtype=torch.float8_e4m3fn if self.use_fp8 else qkv.dtype,
            device=qkv.device,
        )

        torch.ops.vllm.hpc_rope_norm_forward(qkv, output, layer_name)
        return output

    def _forward_impl(
        self,
        qkv: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: HpcAttnMetadata,
        attn_layer: torch.nn.Module,
        output: torch.Tensor,
    ) -> None:
        """Actual forward logic called by the custom op.

        Writes processed q into *output* and attaches extra params
        (e.g. FP8 scales) to *attn_layer* as attributes.
        """
        import hpc

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_prefill_reqs = attn_metadata.num_prefills
        num_decode_reqs = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        qkv = qkv[:num_actual_tokens]

        num_prefill_tokens = num_actual_tokens - num_decode_tokens

        # KV cache for the FP8 path is stored as uint8; view it as fp8 so the
        # rope_norm_store_kv_fp8 kernel can write quantized K/V in-place.
        if self.use_fp8:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        # Per-tensor K/V scales (shape [1]) used by the FP8 kernel.
        k_scale = attn_layer._k_scale.reshape(1)
        v_scale = attn_layer._v_scale.reshape(1)

        q_norm_weight = (
            self.qnorm_weight if self.qk_norm_policy != QkNormPolicy.NONE else None
        )
        k_norm_weight = (
            self.knorm_weight if self.qk_norm_policy != QkNormPolicy.NONE else None
        )

        # --- Prefill ---
        if num_prefill_reqs > 0:
            seq_lens_prefill = attn_metadata.seq_lens[num_decode_reqs:]
            cu_seqlens_prefill = attn_metadata.qo_indptr
            max_seqlens = attn_metadata.max_query_len
            block_table_prefill = attn_metadata.block_table_tensor[num_decode_reqs:]
            qkv_prefill = qkv[num_decode_tokens:]
            out_q_prefill = output[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]

            if self.use_fp8:
                _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
                    key_cache=kv_cache[:, 0],
                    value_cache=kv_cache[:, 1],
                    qkv=qkv_prefill,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=seq_lens_prefill,
                    q_index=cu_seqlens_prefill,
                    kvcache_indices=block_table_prefill,
                    is_prefill=True,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=max_seqlens,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=out_q_prefill,
                )
                attn_metadata.hpc_prefill_q_scale = q_scale
            else:
                hpc.rope_norm_store_kv(
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    qkv_prefill,
                    self.cos_sin_cache,
                    seq_lens_prefill,
                    cu_seqlens_prefill,
                    block_table_prefill,
                    True,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=out_q_prefill,
                    qk_norm_policy=self.qk_norm_policy,
                )

        # --- Decode ---
        if num_decode_reqs > 0:
            num_seq_kvcache = attn_metadata.seq_lens[:num_decode_reqs]
            block_table_decode = attn_metadata.block_table_tensor[:num_decode_reqs]
            qkv_decode = qkv[:num_decode_tokens]
            # Single-token decode: q_index is the per-request prefix sum
            # [0, 1, ..., num_decode_reqs].
            decode_query_len = attn_metadata.decode_query_len
            out_q_decode = output[:num_decode_tokens]

            if self.use_fp8:
                _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
                    key_cache=kv_cache[:, 0],
                    value_cache=kv_cache[:, 1],
                    qkv=qkv_decode,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=num_seq_kvcache,
                    q_index=attn_metadata.qo_indptr_decode,
                    kvcache_indices=block_table_decode,
                    is_prefill=False,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=decode_query_len,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=out_q_decode,
                )
                attn_metadata.hpc_decode_q_scale = q_scale
                if split_k_flag is not None:
                    attn_metadata.hpc_split_k_flag = split_k_flag
            else:
                hpc.rope_norm_store_kv(
                    kv_cache[:, 0],
                    kv_cache[:, 1],
                    qkv_decode,
                    self.cos_sin_cache,
                    num_seq_kvcache,
                    attn_metadata.qo_indptr_decode,
                    block_table_decode,
                    False,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=out_q_decode,
                    qk_norm_policy=self.qk_norm_policy,
                )
