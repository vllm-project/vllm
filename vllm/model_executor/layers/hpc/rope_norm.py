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
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

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
    # bind_kv_cache stores the per-layer KV cache as a single tensor
    # (num_blocks, num_kv_heads, block_size, 2 * head_size), so use it directly.
    kv_cache = attn_layer.kv_cache

    if kv_cache.numel() == 0:
        output.zero_()
        return

    assert kv_cache.dim() == 4, (
        f"Expected kv_cache to have 4 dims, got {tuple(kv_cache.shape)}"
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

        if kv_cache_dtype not in ("fp8_e4m3", "auto", "bfloat16"):
            logger.warning_once(
                f"hpc rope_norm not support kv_cache_dtype:{kv_cache_dtype}, "
                "only support fp8_e4m3, auto, bfloat16"
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

    def _kv_write_scratch(self, cache: torch.Tensor) -> torch.Tensor:
        """Throwaway dense K/V pages for the stride-aware fallback path.

        ``hpc.rope_norm_store_kv[_fp8]`` always writes the paged cache itself
        (rotated K/V *and* zero padding for the tail of every request's last
        block), addressing it densely. When the real pages are not dense those
        writes would land on unrelated blocks, so they are redirected here by
        pairing this scratch cache with an all-zero block table; the real data
        is taken out through ``out_k``/``out_v`` instead.

        Returns a (2, 1, block_size, num_kv_heads, head_dim) tensor: index 0 is
        the key scratch page, index 1 the value scratch page.
        """
        block_size = cache.shape[1]
        key = (cache.dtype, block_size, cache.device)
        if getattr(self, "_kv_scratch_key", None) != key:
            self._kv_scratch = torch.zeros(
                2,
                1,
                block_size,
                self.num_kv_heads,
                self.head_dim,
                dtype=cache.dtype,
                device=cache.device,
            )
            self._kv_scratch_key = key
        return self._kv_scratch

    @staticmethod
    def _zero_pad_last_blocks(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
    ) -> None:
        """Zero the unused tail slots of each request's last block.

        The HPC attention kernels read whole blocks, so the slots past
        ``seq_len`` must be zero. The fused op normally does this itself; on the
        fallback path its (densely addressed) padding writes are discarded, so
        the padding is reapplied here through the real, strided cache views.

        Shapes are static and no value is inspected on the host, so this stays
        CUDA-graph capturable. It is also self-contained: each request's last
        block is read, masked and written back, so slots that must be preserved
        (and requests padded in for graph capture, whose ``seq_len`` is 0) keep
        their previous contents instead of being redirected somewhere.
        """
        block_size = key_cache.shape[1]
        seq_lens = seq_lens.long()
        # seq_len == 0 only happens for graph-capture padding; clamping keeps
        # the gather in bounds and the all-False mask below makes the write a
        # no-op for those rows.
        last_block_idx = (seq_lens - 1).clamp_min(0) // block_size
        last_block_idx = last_block_idx.clamp_max(block_table.shape[1] - 1)
        last_block = block_table.gather(1, last_block_idx.unsqueeze(1)).squeeze(1)
        last_block = last_block.long()

        remainder = seq_lens % block_size
        # A block that ends exactly on the boundary has no tail to clear.
        remainder = torch.where(remainder == 0, block_size, remainder)

        offsets = torch.arange(block_size, device=key_cache.device)
        is_tail = (offsets.unsqueeze(0) >= remainder.unsqueeze(1)) & (
            seq_lens > 0
        ).unsqueeze(1)
        is_tail = is_tail[..., None, None]

        for cache in (key_cache, value_cache):
            view = cache
            if view.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                view = view.view(torch.uint8)
            blocks = view.index_select(0, last_block)
            blocks = torch.where(is_tail, blocks.new_zeros(()), blocks)
            view.index_copy_(0, last_block, blocks)

    @staticmethod
    def _scatter_kv_cache(
        cache: torch.Tensor,
        src: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Write dense per-token K (or V) into a paged cache view.

        Used on the stride-aware fallback path (see ``_forward_impl``).
        ``cache`` is a (num_blocks, block_size, num_kv_heads, head_dim) view
        with arbitrary block/token/head strides; ``src`` is the dense
        (num_tokens, num_kv_heads, head_dim) buffer produced by the fused HPC
        kernel. FP8 payloads are moved as raw bytes because ``index_copy_``
        has no FP8 kernel.

        Padded tokens carry ``PAD_SLOT_ID`` (-1); clamping sends them to the
        reserved null block instead of tripping ``index_copy_``'s bounds check.
        This keeps the op CUDA-graph safe (static shape, no host sync), unlike
        masking the padded rows out.
        """
        # (num_blocks, block_size, ...) -> (num_slots, num_kv_heads, head_dim).
        # Valid as a view for any layout whose block stride is a whole number
        # of token strides, which holds for every layout the backend accepts.
        flat = cache.flatten(0, 1)
        if flat.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            flat = flat.view(torch.uint8)
            src = src.view(torch.uint8)
        flat.index_copy_(0, slot_mapping.clamp_min(NULL_BLOCK_ID), src)

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

        # KV cache for the FP8 path is stored as uint8; view it as fp8 so the
        # rope_norm_store_kv_fp8 kernel can write quantized K/V in-place.
        # Must happen *before* deriving the K/V views, so that they (and any
        # buffer whose dtype is taken from them) carry the FP8 dtype.
        if self.use_fp8:
            kv_cache = kv_cache.view(torch.float8_e4m3fn)

        # (B, H, N, 2*hs) -> two (B, N, H, hs) K/V views, as in HpcAttentionImpl.
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_dim, dim=-1)

        # hpc.rope_norm_store_kv[_fp8] addresses its K/V caches as *dense*
        # NHD tensors and ignores their strides, so it can only write the cache
        # in place when both views are contiguous. Packed layouts (the backend's
        # LBNHC pages interleave K and V inside the state-content dim, so each
        # split view has a token stride of 2 * head_dim) must instead take the
        # rotated K/V out through out_k/out_v and be scattered separately.
        fused_kv_write = key_cache.is_contiguous() and value_cache.is_contiguous()
        if not fused_kv_write:
            logger.warning_once(
                "HpcRopeNorm: KV cache pages interleave K and V, which "
                "hpc.rope_norm_store_kv[_fp8] cannot address, so K/V are "
                "written by a separate scatter. Update hpc-ops to a build with "
                "strided KV cache support to re-enable the fully fused write."
            )
            slot_mapping = attn_metadata.slot_mapping

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_prefill_reqs = attn_metadata.num_prefills
        num_decode_reqs = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        qkv = qkv[:num_actual_tokens]

        num_prefill_tokens = num_actual_tokens - num_decode_tokens

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

            kv_dtype = key_cache.dtype
            if fused_kv_write:
                out_k = out_v = None
                kcache_arg, vcache_arg = key_cache, value_cache
                kv_indices_arg = block_table_prefill
            else:
                out_k = torch.empty(
                    num_prefill_tokens,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=kv_dtype,
                    device=qkv.device,
                )
                out_v = torch.empty_like(out_k)
                # Send the kernel's own cache writes to scratch (all block ids
                # remapped to the single scratch page).
                scratch = self._kv_write_scratch(key_cache)
                kcache_arg, vcache_arg = scratch[0], scratch[1]
                kv_indices_arg = torch.zeros_like(block_table_prefill)

            if self.use_fp8:
                _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
                    key_cache=kcache_arg,
                    value_cache=vcache_arg,
                    qkv=qkv_prefill,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=seq_lens_prefill,
                    q_index=cu_seqlens_prefill,
                    kvcache_indices=kv_indices_arg,
                    is_prefill=True,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=max_seqlens,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=out_q_prefill,
                    out_k=out_k,
                    out_v=out_v,
                )
                attn_metadata.hpc_prefill_q_scale = q_scale
            else:
                hpc.rope_norm_store_kv(
                    kcache_arg,
                    vcache_arg,
                    qkv_prefill,
                    self.cos_sin_cache,
                    seq_lens_prefill,
                    cu_seqlens_prefill,
                    kv_indices_arg,
                    True,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=out_q_prefill,
                    out_k=out_k,
                    out_v=out_v,
                    qk_norm_policy=self.qk_norm_policy,
                )

            if out_k is not None:
                self._zero_pad_last_blocks(
                    key_cache, value_cache, seq_lens_prefill, block_table_prefill
                )
                prefill_slots = slot_mapping[num_decode_tokens:num_actual_tokens]
                self._scatter_kv_cache(key_cache, out_k, prefill_slots)
                self._scatter_kv_cache(value_cache, out_v, prefill_slots)

        # --- Decode ---
        if num_decode_reqs > 0:
            num_seq_kvcache = attn_metadata.seq_lens[:num_decode_reqs]
            block_table_decode = attn_metadata.block_table_tensor[:num_decode_reqs]
            qkv_decode = qkv[:num_decode_tokens]
            # Single-token decode: q_index is the per-request prefix sum
            # [0, 1, ..., num_decode_reqs].
            decode_query_len = attn_metadata.decode_query_len
            out_q_decode = output[:num_decode_tokens]

            if fused_kv_write:
                out_k = out_v = None
                kcache_arg, vcache_arg = key_cache, value_cache
                kv_indices_arg = block_table_decode
            else:
                out_k = torch.empty(
                    num_decode_tokens,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=key_cache.dtype,
                    device=qkv.device,
                )
                out_v = torch.empty_like(out_k)
                scratch = self._kv_write_scratch(key_cache)
                kcache_arg, vcache_arg = scratch[0], scratch[1]
                kv_indices_arg = torch.zeros_like(block_table_decode)

            if self.use_fp8:
                _, q_scale, split_k_flag = hpc.rope_norm_store_kv_fp8(
                    key_cache=kcache_arg,
                    value_cache=vcache_arg,
                    qkv=qkv_decode,
                    cos_sin=self.cos_sin_cache,
                    num_seqlen_per_req=num_seq_kvcache,
                    q_index=attn_metadata.qo_indptr_decode,
                    kvcache_indices=kv_indices_arg,
                    is_prefill=False,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    quant_policy=self._quant_type,
                    max_seqlens=decode_query_len,
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    qk_norm_policy=self.qk_norm_policy,
                    out_q=out_q_decode,
                    out_k=out_k,
                    out_v=out_v,
                )
                attn_metadata.hpc_decode_q_scale = q_scale
                if split_k_flag is not None:
                    attn_metadata.hpc_split_k_flag = split_k_flag
            else:
                hpc.rope_norm_store_kv(
                    kcache_arg,
                    vcache_arg,
                    qkv_decode,
                    self.cos_sin_cache,
                    num_seq_kvcache,
                    attn_metadata.qo_indptr_decode,
                    kv_indices_arg,
                    False,  # is_prefill
                    q_norm_weight=q_norm_weight,
                    k_norm_weight=k_norm_weight,
                    out_q=out_q_decode,
                    out_k=out_k,
                    out_v=out_v,
                    qk_norm_policy=self.qk_norm_policy,
                )

            if out_k is not None:
                self._zero_pad_last_blocks(
                    key_cache, value_cache, num_seq_kvcache, block_table_decode
                )
                decode_slots = slot_mapping[:num_decode_tokens]
                self._scatter_kv_cache(key_cache, out_k, decode_slots)
                self._scatter_kv_cache(value_cache, out_v, decode_slots)

        # Signal HpcAttentionImpl that KV cache has already been written by
        # rope_norm_store_kv[_fp8] above, so it should skip its own
        # reshape_and_cache_flash. Set after the fused kernels ran (either
        # prefill or decode branch); otherwise the standard KV-write path in
        # the attention impl would kick in (which is what non-HpcRopeNorm
        # models rely on).
        attn_metadata.hpc_kv_written = True
