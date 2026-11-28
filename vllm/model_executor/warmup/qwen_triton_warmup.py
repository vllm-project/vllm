# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm up Qwen Triton kernels from the loaded model's compile keys."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)

_QWEN_MODEL_TYPES = frozenset(
    {
        "qwen3_next",
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen4_exp",
        "qwen4_exp_text",
    }
)

# Covers L=1 constexpr, non-divisible runtime L, and divisible runtime L.
_FLA_POST_CONV_WARMUP_LENGTHS = (1, 2, 16)

# The gfx1151 W4A16 Triton kernel is specialized on the tile chosen for M and on
# whether M is a multiple of 16. M <= 8 takes the HIP skinny path, so cover both
# sides of each tile boundary above it.
_W4A16_WARMUP_M = (9, 16, 17, 32, 33, 64, 65, 128, 129, 144, 1025, 2048)


@dataclass(frozen=True, slots=True)
class _QwenGDNWarmupConfig:
    h: int
    hv: int
    k: int
    v: int
    conv_kernel_size: int
    conv_state: torch.Tensor
    conv_dtype: torch.dtype
    norm_weight_dtype: torch.dtype
    norm_before_gate: bool
    norm_activation: str
    a_log: torch.Tensor
    dt_bias: torch.Tensor
    state_stride_token: int
    state_dtype: torch.dtype
    fused_spec_decode: bool = False
    # The FlashInfer prefill takes exp(g) straight from fused_post_conv_prep.
    post_conv_g_exp: bool = False

    @property
    def conv_dim(self) -> int:
        return 2 * self.h * self.k + self.hv * self.v


def _is_non_empty_tensor(value: object) -> bool:
    return isinstance(value, torch.Tensor) and value.numel() > 0


def _is_qwen_gdn_layer(module: object) -> bool:
    return all(
        hasattr(module, attr)
        for attr in (
            "num_k_heads",
            "num_v_heads",
            "head_k_dim",
            "head_v_dim",
            "conv_kernel_size",
            "tp_size",
            "kv_cache",
            "norm",
            "A_log",
            "dt_bias",
        )
    )


def _iter_qwen_gdn_layers(static_forward_context: object):
    if not isinstance(static_forward_context, dict):
        return

    for module in static_forward_context.values():
        if _is_qwen_gdn_layer(module):
            yield module


def _split_qwen_gdn_cache(kv_cache: object) -> tuple[torch.Tensor, torch.Tensor] | None:
    if isinstance(kv_cache, (list, tuple)) and len(kv_cache) >= 2:
        conv_cache, ssm_state = kv_cache[:2]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state

    if isinstance(kv_cache, torch.Tensor) and kv_cache.size(0) >= 2:
        conv_cache = kv_cache[0]
        ssm_state = kv_cache[1]
        if _is_non_empty_tensor(conv_cache) and _is_non_empty_tensor(ssm_state):
            return conv_cache, ssm_state
    return None


def _qwen_gdn_warmup_config(
    static_forward_context: object,
) -> _QwenGDNWarmupConfig | None:
    found_layer = False
    for layer in _iter_qwen_gdn_layers(static_forward_context):
        found_layer = True
        cache_tensors = _split_qwen_gdn_cache(getattr(layer, "kv_cache", None))
        if cache_tensors is None:
            continue

        conv_cache, ssm_state = cache_tensors
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_state = (
            conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        )
        tp_size = int(layer.tp_size)
        h = int(layer.num_k_heads) // tp_size
        hv = int(layer.num_v_heads) // tp_size
        norm = layer.norm

        return _QwenGDNWarmupConfig(
            h=h,
            hv=hv,
            k=int(layer.head_k_dim),
            v=int(layer.head_v_dim),
            conv_kernel_size=int(layer.conv_kernel_size),
            conv_state=conv_state,
            conv_dtype=conv_state.dtype,
            norm_weight_dtype=norm.weight.dtype,
            norm_before_gate=bool(norm.norm_before_gate),
            norm_activation=str(norm.activation),
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_stride_token=int(ssm_state.stride(0)),
            state_dtype=ssm_state.dtype,
            fused_spec_decode=bool(
                getattr(layer, "enable_fused_gdn_spec_decode", False)
            ),
            post_conv_g_exp=getattr(layer, "gdn_prefill_backend", None) == "flashinfer",
        )

    if found_layer:
        logger.info("Skipping Qwen GDN Triton warmup: no bound Qwen GDN cache found.")
    else:
        logger.info("Skipping Qwen GDN Triton warmup: no Qwen GDN layer found.")
    return None


def _warm_gated_rms_norm_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
    max_num_tokens: int,
    x_dtype: torch.dtype,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import (
        warmup_layer_norm_fwd,
    )

    # _rms_norm_gated_cuda: a row per token; RMSNormGated: a row per head.
    fused = config.fused_spec_decode
    warmup_layer_norm_fwd(
        max_num_tokens=max_num_tokens,
        rows_per_token=1 if fused else config.hv,
        groups_per_row=config.hv if fused else 1,
        group_size=config.v,
        x_dtype=x_dtype,
        weight_dtype=config.norm_weight_dtype,
        device=device,
        norm_before_gate=config.norm_before_gate,
        is_rms_norm=True,
        activation=config.norm_activation,
    )


def _warm_causal_conv1d_fwd_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID, PAD_SLOT_ID

    x_storage = torch.empty(
        (1, config.conv_dim), dtype=config.conv_dtype, device=device
    )
    x = x_storage.t()
    weight = torch.empty(
        (config.conv_dim, config.conv_kernel_size),
        dtype=config.conv_dtype,
        device=device,
    )
    cache_indices = torch.full((1,), NULL_BLOCK_ID, dtype=torch.int32, device=device)
    has_initial_state = torch.empty(1, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)

    causal_conv1d_fn(
        x,
        weight,
        None,
        config.conv_state,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        pad_slot_id=PAD_SLOT_ID,
        null_block_id=NULL_BLOCK_ID,
        metadata=None,
        validate_data=False,
    )


def _warm_fused_post_conv_kernel(
    device: torch.device, config: _QwenGDNWarmupConfig
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (  # noqa: E501
        fused_post_conv_prep,
    )

    qkv_dim = 2 * config.h * config.k + config.hv * config.v
    for length in _FLA_POST_CONV_WARMUP_LENGTHS:
        conv_output = torch.empty(
            (length, qkv_dim), dtype=config.conv_dtype, device=device
        )
        a = torch.empty((length, config.hv), dtype=config.conv_dtype, device=device)
        b = torch.empty_like(a)

        fused_post_conv_prep(
            conv_output,
            a,
            b,
            config.a_log,
            config.dt_bias,
            config.h,
            config.k,
            config.v,
            apply_l2norm=True,
            output_g_exp=config.post_conv_g_exp,
        )


def _warm_fused_sigmoid_gating_delta_rule_update_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
) -> None:
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    k = torch.empty_like(q)
    v = torch.empty((1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device)
    a = torch.empty((1, 1, config.hv), dtype=config.conv_dtype, device=device)
    b = torch.empty_like(a)
    state = torch.empty(
        (1, config.state_stride_token),
        dtype=config.state_dtype,
        device=device,
    )
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.empty((1, 1), dtype=torch.int32, device=device)
    ssm_state_indices.zero_()

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=a,
        b=b,
        dt_bias=config.dt_bias,
        q=q,
        k=k,
        v=v,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _warm_fused_gdn_decode_post_conv_mtp_kernel(
    device: torch.device,
    config: _QwenGDNWarmupConfig,
    x_dtype: torch.dtype,
    num_spec_tokens: int,
    max_num_reqs: int,
) -> None:
    """Compile the ROCm Triton MTP decode kernels for every V tile size the
    request count can select, with the layer's packed QKVZ/BA layouts so the
    stride specializations match serving."""
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_decode_post_conv_mtp import (  # noqa: E501
        fused_gdn_decode_post_conv_mtp_triton,
    )
    from vllm.utils.platform_utils import num_compute_units

    num_cus = num_compute_units(device.index)
    # Smallest request counts that select BV = 32, 64 and V (see the wrapper).
    request_counts = sorted({1, -(-num_cus // config.hv), 4 * num_cus // config.hv + 1})
    tokens_per_req = num_spec_tokens + 1
    norm_weight = torch.ones(config.v, dtype=config.norm_weight_dtype, device=device)
    state = torch.zeros(
        2 * config.state_stride_token, dtype=config.state_dtype, device=device
    ).as_strided(
        (2, config.hv, config.v, config.k),
        (config.state_stride_token, config.v * config.k, config.k, 1),
    )
    for num_reqs in request_counts:
        if num_reqs > max_num_reqs:
            continue
        num_tokens = num_reqs * tokens_per_req
        mixed_qkvz = torch.zeros(
            (num_tokens, config.conv_dim + config.hv * config.v),
            dtype=x_dtype,
            device=device,
        )
        mixed_qkv, output_gate = mixed_qkvz.split(
            [config.conv_dim, config.hv * config.v], dim=-1
        )
        ba = torch.zeros((num_tokens, 2 * config.hv), dtype=x_dtype, device=device)
        b, a = ba.chunk(2, dim=-1)
        fused_gdn_decode_post_conv_mtp_triton(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=config.a_log,
            dt_bias=config.dt_bias,
            state_indices=torch.zeros(
                (num_reqs, tokens_per_req), dtype=torch.int32, device=device
            ),
            cu_seqlens=torch.arange(
                0,
                num_tokens + 1,
                tokens_per_req,
                dtype=torch.int32,
                device=device,
            ),
            num_accepted_tokens=torch.ones(num_reqs, dtype=torch.int32, device=device),
            state=state,
            output_gate=output_gate.reshape(num_tokens, config.hv, config.v),
            norm_weight=norm_weight,
            out=torch.empty(
                (num_tokens, config.hv, config.v), dtype=x_dtype, device=device
            ),
            scale=config.k**-0.5,
            norm_eps=1e-6,
            output_gate_activation=config.norm_activation,
        )


def _warm_state_gather_scatter_kernels(
    device: torch.device, config: "_QwenGDNWarmupConfig"
) -> None:
    """Compile the gather/scatter variants serving hits.

    Triton specializes on 16-byte pointer alignment and stride == 1; mixed
    batches slice the indices/mask at `num_decodes` and spec-decode block
    tables give stride-2 indices, so warm each combination.
    """
    from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
        gather_initial_states,
    )
    from vllm.model_executor.layers.mamba.ops.scatter_states import (
        scatter_states,
    )

    state = torch.zeros(
        2, config.hv, config.v, config.k, dtype=config.state_dtype, device=device
    )
    idx_pool = torch.zeros(6, dtype=torch.int32, device=device)
    has_pool = torch.zeros(2, dtype=torch.bool, device=device)
    for idx, has in (
        (idx_pool[:1], has_pool[:1]),  # aligned, contiguous
        (idx_pool[1:2], has_pool[1:2]),  # unaligned
        (idx_pool[4:5], has_pool[1:2]),  # aligned indices, unaligned mask
        (idx_pool[::2][:1], has_pool[:1]),  # strided (spec-decode block table)
    ):
        gathered = gather_initial_states(state, idx, has)
        scatter_states(state, gathered.to(torch.float32), idx)
        scatter_states(state, gathered, idx)


def _warm_w4a16_kernels(runner: "GPUModelRunner") -> None:
    """Compile the Triton W4A16 kernel for every loaded layer shape."""
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        RDNAHybridW4A16LinearKernel,
    )

    dtype = runner.vllm_config.model_config.dtype
    m_values = [m for m in _W4A16_WARMUP_M if m <= runner.max_num_tokens]
    seen = set()
    for layer in runner.get_model().modules():
        kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
        if not isinstance(kernel, RDNAHybridW4A16LinearKernel):
            continue
        w_q, _, w_zp = kernel._get_weight_params(layer)
        key = (tuple(w_q.shape), kernel.config.group_size, w_zp is not None)
        if key in seen:
            continue
        seen.add(key)
        for m in m_values:
            x = torch.empty((m, w_q.shape[1] * 2), dtype=dtype, device=runner.device)
            kernel.apply_weights(layer, x)
    logger.info("Warmed W4A16 Triton kernels for %d layer shapes.", len(seen))


def _warm_triton_attention_kernel(runner: "GPUModelRunner") -> None:
    """Compile the unified-attention kernel for the bound hybrid KV cache."""
    from vllm.model_executor.layers.attention import Attention
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    groups = runner.kv_cache_config.kv_cache_groups
    for name, layer in runner.compilation_config.static_forward_context.items():
        if not (
            isinstance(layer, Attention)
            and layer.attn_backend.get_name() == "TRITON_ATTN"
            and layer.kv_cache_dtype == "auto"
            and layer.kv_cache.ndim == 4
        ):
            continue
        # Same K/V views as TritonAttentionImpl.forward: [B, H, N, 2 * hs].
        hs = layer.head_size
        key_cache, value_cache = layer.kv_cache[:1].transpose(1, 2).split(hs, dim=-1)
        group_id = next(i for i, g in enumerate(groups) if name in g.layer_names)
        block_table = runner.block_tables.input_block_tables[group_id][:1, :1]
        device = runner.device
        # The kernel key does not depend on the query length.
        dtype = runner.vllm_config.model_config.dtype
        q = torch.empty((1, layer.num_heads, hs), dtype=dtype, device=device)
        unified_attention(
            q=q,
            k=key_cache,
            v=value_cache,
            out=torch.empty_like(q),
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
            max_seqlen_q=1,
            seqused_k=torch.tensor([1], dtype=torch.int32, device=device),
            max_seqlen_k=1,
            softmax_scale=hs**-0.5,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0,
            q_descale=None,
            # Production passes the per-layer scales expanded per request, which
            # makes them pointer arguments rather than constexpr None.
            k_descale=layer._k_scale.expand(1, key_cache.shape[2]),
            v_descale=layer._v_scale.expand(1, key_cache.shape[2]),
        )
        logger.info(
            "Warmed Triton unified attention for block_size=%d.", key_cache.shape[1]
        )
        return


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.accelerator.synchronize(device)


@torch.inference_mode()
def qwen_triton_warmup(
    runner: "GPUModelRunner",
    model_config: "ModelConfig",
) -> None:
    """Warm Qwen GDN Triton kernels reported by the JIT monitor."""
    model_type = getattr(model_config.hf_text_config, "model_type", "") or getattr(
        model_config.hf_config, "model_type", ""
    )
    if model_type not in _QWEN_MODEL_TYPES:
        return

    device = runner.device
    logger.info("Warming up Qwen Triton kernels for model_type=%s.", model_type)
    if current_platform.is_rocm():
        from vllm.platforms.rocm import on_gfx1151

        if on_gfx1151():
            _warm_w4a16_kernels(runner)
            _warm_triton_attention_kernel(runner)

    gdn_config = _qwen_gdn_warmup_config(
        runner.compilation_config.static_forward_context
    )
    if gdn_config is None:
        return

    max_num_tokens = max(1, int(runner.max_num_tokens))
    _warm_gated_rms_norm_kernel(device, gdn_config, max_num_tokens, model_config.dtype)
    _warm_causal_conv1d_fwd_kernel(device, gdn_config)
    _warm_fused_post_conv_kernel(device, gdn_config)
    # Pooling only runs full prefills; the decode update kernel is unused.
    if not runner.is_pooling_model:
        _warm_fused_sigmoid_gating_delta_rule_update_kernel(device, gdn_config)
    _warm_state_gather_scatter_kernels(device, gdn_config)
    # ROCm runs the Triton port of the fused MTP decode kernel.
    spec_config = runner.speculative_config
    if (
        current_platform.is_rocm()
        and gdn_config.fused_spec_decode
        and spec_config is not None
    ):
        _warm_fused_gdn_decode_post_conv_mtp_kernel(
            device,
            gdn_config,
            model_config.dtype,
            spec_config.num_speculative_tokens,
            runner.max_num_reqs,
        )
    _synchronize_device(device)
