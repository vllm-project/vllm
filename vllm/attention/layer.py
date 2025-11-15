# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.attention import AttentionType
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.selector import get_attn_backend
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.multimodal import MultiModalConfig
from vllm.config.vllm import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.models.vision import get_vit_attn_backend
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backends.mla.common import reorg_kvcache
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

if current_platform.is_rocm():
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.platforms.rocm import on_gfx9
else:
    on_gfx9 = lambda *args, **kwargs: False
    rocm_aiter_ops = None


FP8_DTYPE = current_platform.fp8_dtype()
logger = init_logger(__name__)
USE_XFORMERS_OPS = None


def check_xformers_availability():
    global USE_XFORMERS_OPS
    if USE_XFORMERS_OPS is not None:
        return USE_XFORMERS_OPS

    if current_platform.is_cuda() and current_platform.has_device_capability(100):
        # Xformers FA is not compatible with B200
        USE_XFORMERS_OPS = False
    else:
        try:
            from importlib.util import find_spec

            find_spec("xformers.ops")
            USE_XFORMERS_OPS = True
        except ImportError:
            USE_XFORMERS_OPS = False

    # the warning only needs to be shown once
    if not USE_XFORMERS_OPS:
        logger.warning("Xformers is not available, falling back.")

    return USE_XFORMERS_OPS


def check_upstream_fa_availability(dtype: torch.dtype):
    if (
        dtype in (torch.float16, torch.bfloat16)
        and current_platform.is_cuda()
        and current_platform.has_device_capability(80)
    ):
        from transformers.utils import is_flash_attn_2_available

        return is_flash_attn_2_available()
    if current_platform.is_rocm():
        from importlib.util import find_spec

        return find_spec("flash_attn") is not None
    return False


def maybe_get_vit_flash_attn_backend(
    attn_backend: AttentionBackendEnum,
    use_upstream_fa: bool,
    attn_backend_override: AttentionBackendEnum | None = None,
) -> tuple[AttentionBackendEnum, Callable | None]:
    if current_platform.is_rocm():
        if envs.VLLM_ROCM_USE_AITER and envs.VLLM_ROCM_USE_AITER_MHA and on_gfx9():
            attn_backend = AttentionBackendEnum.ROCM_AITER_FA

        elif (
            check_upstream_fa_availability(torch.get_default_dtype())
            and on_gfx9()
            and attn_backend_override is None
        ):
            attn_backend = AttentionBackendEnum.FLASH_ATTN
            use_upstream_fa = True
        else:
            return AttentionBackendEnum.TORCH_SDPA, None

    elif current_platform.is_cuda():
        if (
            attn_backend != AttentionBackendEnum.FLASH_ATTN
            and check_upstream_fa_availability(torch.get_default_dtype())
        ):
            attn_backend = AttentionBackendEnum.FLASH_ATTN
            use_upstream_fa = True
    elif current_platform.is_xpu():
        assert attn_backend == AttentionBackendEnum.FLASH_ATTN, (
            "XPU platform only supports FLASH_ATTN as vision attention backend."
        )
        use_upstream_fa = False
    else:
        return AttentionBackendEnum.TORCH_SDPA, None

    if attn_backend in {
        AttentionBackendEnum.FLASH_ATTN,
        AttentionBackendEnum.ROCM_AITER_FA,
    }:
        if attn_backend == AttentionBackendEnum.ROCM_AITER_FA:
            from aiter import flash_attn_varlen_func
        else:
            if use_upstream_fa:
                from flash_attn import flash_attn_varlen_func
            else:
                from vllm.attention.utils.fa_utils import flash_attn_varlen_func
    else:
        flash_attn_varlen_func = None

    return attn_backend, flash_attn_varlen_func


def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
    kv_cache_dtype: str,
    calculate_kv_scales: bool,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
        kv_cache_dtype: The KV cache data type string.
        calculate_kv_scales: Whether to calculate KV scales dynamically.
    """
    # The default k/v_scale is set to 1.0. This is ignored
    # when kv-cache is not fp8, and should be used with
    # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
    # expect the pre-quantized k/v_scale to be loaded along
    # with the model weights.
    layer.kv_cache_dtype = kv_cache_dtype
    layer.calculate_kv_scales = calculate_kv_scales
    layer._k_scale = torch.tensor(1.0, dtype=torch.float32)
    layer._v_scale = torch.tensor(1.0, dtype=torch.float32)
    layer._q_scale = torch.tensor(1.0, dtype=torch.float32)
    layer._prob_scale = torch.tensor(1.0, dtype=torch.float32)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )
    if quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    ):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(
            self, quant_config, prefix, kv_cache_dtype, calculate_kv_scales
        )

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                block_size,
                use_mla=False,
                has_sink=self.has_sink,
            )
        else:
            self.attn_backend = attn_backend

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
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
            **extra_impl_args,
        )
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        self.use_output = self.attn_backend.accept_output_buffer
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        # for attn backends supporting query quantization
        self.query_quant = None
        if (
            self.kv_cache_dtype.startswith("fp8")
            and self.impl.supports_quant_query_input()
        ):
            self.query_quant = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(query, key, value, self.layer_name)
        output_dtype = query.dtype
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input():
                query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size)
            if self.use_direct_call:
                forward_context: ForwardContext = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata, output=output
                )
            else:
                torch.ops.vllm.unified_attention_with_output(
                    query, key, value, output, self.layer_name
                )
            return output.view(-1, hidden_size)
        else:
            if self.use_direct_call:
                forward_context = get_forward_context()
                attn_metadata = forward_context.attn_metadata
                if isinstance(attn_metadata, dict):
                    attn_metadata = attn_metadata[self.layer_name]
                self_kv_cache = self.kv_cache[forward_context.virtual_engine]
                return self.impl.forward(
                    self, query, key, value, self_kv_cache, attn_metadata
                )
            else:
                return torch.ops.vllm.unified_attention(
                    query, key, value, self.layer_name
                )

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        if self.sliding_window is not None:
            assert not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                sliding_window=self.sliding_window,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                dtype=self.kv_cache_torch_dtype,
            )


class MultiHeadAttention(nn.Module):
    """Multi-headed attention without any cache, used for ViT."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        # This has no effect, it is only here to make it easier to swap
        # between Attention and MultiHeadAttention
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Determine the attention backend
        attn_backend_override = None
        if multimodal_config is not None:
            attn_backend_override = multimodal_config.mm_encoder_attn_backend
        backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
            attn_backend_override=attn_backend_override,
        )

        # Some auto-selected backends can be upgraded
        # to upstream flash attention if available.
        # If vllm native fa is selected, we use it directly.
        use_upstream_fa = False

        self.attn_backend = (
            backend
            if backend
            in {
                AttentionBackendEnum.TORCH_SDPA,
                AttentionBackendEnum.XFORMERS,
                AttentionBackendEnum.PALLAS,
                AttentionBackendEnum.ROCM_AITER_FA,
                AttentionBackendEnum.FLASH_ATTN,
            }
            else AttentionBackendEnum.TORCH_SDPA
        )

        self.attn_backend, self._flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
                attn_backend_override=attn_backend_override,
            )
        )

        if (
            self.attn_backend == AttentionBackendEnum.XFORMERS
            and not check_xformers_availability()
        ):
            self.attn_backend = AttentionBackendEnum.TORCH_SDPA

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        # this condition is just to make sure that the
        # use_upstream_fa in the log is correct
        if (
            current_platform.is_rocm()
            and self.attn_backend == AttentionBackendEnum.FLASH_ATTN
        ):
            use_upstream_fa = True

        logger.info_once(
            f"MultiHeadAttention attn_backend: {self.attn_backend}, "
            f"use_upstream_fa: {use_upstream_fa}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        if self.is_flash_attn_backend:
            assert self._flash_attn_varlen_func is not None
            cu_seqlens_q = torch.arange(
                0, (bsz + 1) * q_len, step=q_len, dtype=torch.int32, device=query.device
            )
            cu_seqlens_k = torch.arange(
                0, (bsz + 1) * kv_len, step=kv_len, dtype=torch.int32, device=key.device
            )

            out = self._flash_attn_varlen_func(
                query.flatten(0, 1),
                key.flatten(0, 1),
                value.flatten(0, 1),
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=q_len,
                max_seqlen_k=kv_len,
                softmax_scale=self.scale,
            )
        elif self.attn_backend == AttentionBackendEnum.XFORMERS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention_forward(
                query, key, value, scale=self.scale
            )
        elif self.attn_backend == AttentionBackendEnum.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == AttentionBackendEnum.PALLAS:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            from torch_xla.experimental.custom_kernel import flash_attention

            out = flash_attention(query, key, value, sm_scale=self.scale)
            out = out.transpose(1, 2)
        else:
            # ViT attention hasn't supported this backend yet
            raise NotImplementedError(
                f"ViT attention hasn't supported {self.attn_backend} backend yet."
            )

        return out.reshape(bsz, q_len, -1)


class MLAAttention(nn.Module, AttentionLayerBase):
    """Multi-Head Latent Attention layer.

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
        self.head_size = kv_lora_rank + qk_rope_head_dim
        self.layer_name = prefix

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(
            self, quant_config, prefix, kv_cache_dtype, calculate_kv_scales
        )

        dtype = torch.get_default_dtype()
        self.attn_backend = get_attn_backend(
            self.head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla=True,
            use_sparse=use_sparse,
        )
        impl_cls = cast(type[MLAAttentionImpl], self.attn_backend.get_impl_cls())
        self.impl = impl_cls(
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

        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.kv_cache = [
            torch.tensor([])
            for _ in range(
                get_current_vllm_config().parallel_config.pipeline_parallel_size
            )
        ]

        self.use_sparse = use_sparse

        # Initialize q/k/v range constants.
        self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
        self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
        self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)

        
        self._use_compiled_split = False

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe, self.layer_name)

        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]

            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                return self.forward_impl(
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    kv_cache=self_kv_cache,
                    attn_metadata=attn_metadata,
                    output=output,
                )
            else:
                return self.forward_impl(
                    q=q,
                    kv_c_normed=kv_c_normed,
                    k_pe=k_pe,
                    kv_cache=self_kv_cache,
                    attn_metadata=attn_metadata,
                )
        else:
            if self.attn_backend.accept_output_buffer:
                output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
                torch.ops.vllm.unified_mla_attention_with_output(
                    q,
                    kv_c_normed,
                    k_pe,
                    output,
                    self.layer_name,
                )
                return output
            else:
                return torch.ops.vllm.unified_mla_attention(
                    q,
                    kv_c_normed,
                    k_pe,
                    self.layer_name,
                )

    def forward_prefill(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Prefill path orchestration in the layer."""
        assert attn_metadata.prefill is not None
        if self.impl.dcp_world_size is None:
            self.impl.dcp_world_size = get_dcp_group().world_size

        has_context = attn_metadata.prefill.chunked_context is not None

        # KV projection and splitting
        kv_nope = self.impl.kv_b_proj(k_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        # Run prefill for new tokens
        output = self.impl._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        # Handle chunked context if present
        if has_context:
            suffix_output, suffix_lse = output
            prefill_metadata = attn_metadata.prefill
            chunked_context = prefill_metadata.chunked_context

            context_output = None
            context_lse = None
            iters = len(chunked_context.seq_tot)
            workspace = chunked_context.workspace

            for i in range(iters):
                toks = chunked_context.seq_tot[i]

                if self.impl.dcp_world_size > 1:
                    # DCP path: use cp_gather_cache and allgather
                    assert self._k_scale is None or (self._k_scale == 1.0).all(), (
                        "DCP not support scaled kvcache now."
                    )
                    assert chunked_context.padded_local_chunk_seq_lens is not None
                    assert chunked_context.local_context_lens_allranks is not None
                    assert chunked_context.padded_local_cu_seq_lens is not None
                    assert chunked_context.cu_seq_lens_lst is not None

                    ops.cp_gather_cache(
                        src_cache=kv_cache,
                        dst=workspace,
                        block_table=prefill_metadata.block_table,
                        cu_seq_lens=chunked_context.padded_local_cu_seq_lens[i],
                        batch_size=attn_metadata.num_prefills,
                        seq_starts=chunked_context.starts[i],
                    )
                    # workspace
                    # |------- N tokens --------|--------- N*dcp_size tokens ----------|
                    # |<- use for loca_gather ->|<--------- use for allgather -------->|
                    allgather_offset = workspace.shape[0] // (
                        self.impl.dcp_world_size + 1
                    )
                    assert (
                        allgather_offset * (self.impl.dcp_world_size + 1)
                        == workspace.shape[0]
                    )
                    assert toks <= allgather_offset
                    local_gathered_kvcache = workspace[:toks]
                    cur_allgather_workspace = workspace[
                        allgather_offset : allgather_offset
                        * (1 + self.impl.dcp_world_size)
                    ]
                    assert (
                        toks * self.impl.dcp_world_size
                        <= cur_allgather_workspace.shape[0]
                    )
                    cur_allgather_kvcache = cur_allgather_workspace[
                        : toks * self.impl.dcp_world_size
                    ]
                    cur_allgather_kvcache.copy_(
                        get_dcp_group().all_gather(local_gathered_kvcache, dim=0)
                    )
                    assert (
                        cur_allgather_kvcache.shape[-1]
                        == self.impl.kv_lora_rank + self.qk_rope_head_dim
                    )
                    allgatered_kv_c_normed, allgatered_k_pe = (
                        cur_allgather_kvcache.unsqueeze(1).split(
                            [self.impl.kv_lora_rank, self.qk_rope_head_dim], dim=-1
                        )
                    )

                    kv_c_normed, k_pe = reorg_kvcache(
                        allgatered_kv_c_normed,
                        allgatered_k_pe,
                        padded_local_chunk_seq_lens_lst=chunked_context.padded_local_chunk_seq_lens[
                            i
                        ],
                        local_context_lens_allranks=chunked_context.local_context_lens_allranks,
                        sum_seq_len=chunked_context.cu_seq_lens_lst[i][-1],
                        max_seq_len=chunked_context.max_seq_lens[i],
                        toks=toks,
                    )
                    k_pe = k_pe.unsqueeze(1)
                else:
                    # Non-DCP path: use gather_and_maybe_dequant_cache
                    ops.gather_and_maybe_dequant_cache(
                        src_cache=kv_cache,
                        dst=workspace,
                        block_table=prefill_metadata.block_table,
                        cu_seq_lens=chunked_context.cu_seq_lens[i],
                        batch_size=attn_metadata.num_prefills,
                        kv_cache_dtype=self.impl.kv_cache_dtype,
                        scale=self._k_scale,
                        seq_starts=chunked_context.starts[i],
                    )

                    kv_c_normed = workspace[:toks][..., : self.impl.kv_lora_rank]
                    k_pe = workspace[:toks][..., self.impl.kv_lora_rank :].unsqueeze(1)

                # KV projection and splitting
                kv_nope = self.impl.kv_b_proj(kv_c_normed)[0].view(
                    -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
                )
                k_nope, v_chunk = kv_nope.split(
                    [self.qk_nope_head_dim, self.v_head_dim], dim=-1
                )

                k_chunk = torch.cat(
                    (k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1
                )

                # Run attention kernel for this chunk
                attn_output, attn_softmax_lse = self.impl._run_prefill_context_chunk(
                    prefill=prefill_metadata,
                    chunk_idx=i,
                    q=q,
                    k=k_chunk,
                    v=v_chunk,
                )

                # Merge with previous chunks
                if context_output is None:
                    context_output = attn_output
                    context_lse = attn_softmax_lse
                else:
                    output_tmp = torch.empty_like(context_output)
                    output_lse_tmp = torch.empty_like(context_lse)
                    merge_attn_states(
                        output=output_tmp,
                        output_lse=output_lse_tmp,
                        prefix_output=context_output,
                        prefix_lse=context_lse,
                        suffix_output=attn_output,
                        suffix_lse=attn_softmax_lse,
                    )
                    context_output = output_tmp
                    context_lse = output_lse_tmp

            # Merge context with new tokens
            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        # Unpad if necessary
        if self.impl._pad_v:
            output = output[..., : v.shape[-1]]

        return output.flatten(start_dim=-2)

    def forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
    ) -> torch.Tensor:
        """Decode path orchestration in the layer."""
        if self.impl.dcp_world_size is None:
            self.impl.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        # Split q into no-rope and rope parts
        decode_q_nope, decode_q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        # (B, N, P) -> (N, B, P)
        decode_q_nope = decode_q_nope.transpose(0, 1)

        if self.impl.q_pad_num_heads is not None:
            B, N, L = decode_q_pe.shape
            decode_pe_padded = decode_q_pe.new_empty((B, self.impl.q_pad_num_heads, L))
            decode_pe_padded.resize_((B, N, L))
            decode_pe_padded.copy_(decode_q_pe)
            decode_q_pe = decode_pe_padded

        if self.impl.is_aiter_triton_fp8_bmm_enabled:
            # (N, B, P) x (N, P, L) -> (N, B, L) -> (B, N, L)
            decode_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                decode_q_nope,
                self.impl.W_K,
                self.impl.W_K_scale,
                group_size=128,
                transpose_bm=True,
            )
        else:
            N, B, P = decode_q_nope.shape
            _, _, L = self.impl.W_UK_T.shape
            if self.impl.q_pad_num_heads is not None:
                decode_ql_nope = decode_q_nope.new_empty(
                    (self.impl.q_pad_num_heads, B, L)
                )
                decode_ql_nope.resize_((N, B, L))
            else:
                decode_ql_nope = decode_q_nope.new_empty((N, B, L))
            torch.bmm(decode_q_nope, self.impl.W_UK_T, out=decode_ql_nope)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)

        if fp8_attention:
            ql_nope_shape = decode_ql_nope.shape
            decode_ql_nope, _ = ops.scaled_fp8_quant(
                decode_ql_nope.reshape(
                    [ql_nope_shape[0], ql_nope_shape[1] * ql_nope_shape[2]]
                ),
                self._q_scale,
            )
            decode_ql_nope = decode_ql_nope.reshape(ql_nope_shape)
            q_pe_shape = decode_q_pe.shape
            decode_q_pe, _ = ops.scaled_fp8_quant(
                decode_q_pe.reshape([q_pe_shape[0], q_pe_shape[1] * q_pe_shape[2]]),
                self._q_scale,
            )
            decode_q_pe = decode_q_pe.reshape(q_pe_shape)

        decode_q = (decode_ql_nope, decode_q_pe)
        if self.impl.dcp_world_size > 1:
            assert not fp8_attention, "DCP not support fp8 kvcache now."
            decode_q = torch.cat(decode_q, dim=-1)
            decode_q = get_dcp_group().all_gather(decode_q, dim=1)

        attn_out, lse = self.impl._forward_decode(
            decode_q, kv_cache, attn_metadata, self
        )

        if self.impl.dcp_world_size > 1:
            attn_out = cp_lse_ag_out_rs(attn_out, lse, get_dcp_group())

        out = torch.empty(
            (q.shape[0], self.num_heads * self.v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )
        self.impl._v_up_proj(attn_out, out=out)
        return out

    def forward_impl(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        allow_compiled_split: bool = True,
    ) -> torch.Tensor:
        """Forward implementation for MLA attention."""
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLA"
            )

        # Allocate output if not provided
        if output is None:
            output = torch.empty(
                (q.shape[0], self.num_heads * self.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )

        if attn_metadata is None:
            return output.fill_(0)

        # Write KV to cache first (outside backend)
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                kv_c_normed,
                k_pe.squeeze(1) if k_pe.dim() > 2 else k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=self._k_scale,
            )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]
        prefill_k_c = kv_c_normed[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]

        # Process decode and prefill branches
        if has_prefill:
            prefill_out = self.forward_prefill(
                prefill_q, prefill_k_c, prefill_k_pe, kv_cache, attn_metadata
            )
            output[num_decode_tokens:] = prefill_out

        if has_decode:
            decode_out = self.forward_decode(decode_q, kv_cache, attn_metadata)
            output[:num_decode_tokens] = decode_out

        return output

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)

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


def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_calc_kv_scales",
    op_func=maybe_calc_kv_scales,
    mutates_args=["query", "key", "value"],
    fake_impl=maybe_calc_kv_scales_fake,
)


def get_attention_context(
    layer_name: str,
) -> tuple[dict | object | None, Attention | MLAAttention, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, and KV cache tensor for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current virtual engine

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]
    return attn_metadata, attn_layer, kv_cache


@maybe_transfer_kv_layer
def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)

    return output


def unified_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(query).contiguous()


direct_register_custom_op(
    op_name="unified_attention",
    op_func=unified_attention,
    fake_impl=unified_attention_fake,
)


@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)


@maybe_transfer_kv_layer
def unified_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    output = self.forward_impl(
        q=q,
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
    )

    return output


def unified_mla_attention_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(q).contiguous()


direct_register_custom_op(
    op_name="unified_mla_attention",
    op_func=unified_mla_attention,
    mutates_args=[],
    fake_impl=unified_mla_attention_fake,
    dispatch_key=current_platform.dispatch_key,
)


@maybe_transfer_kv_layer
def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)
    self.forward_impl(
        q=q,
        kv_c_normed=kv_c_normed,
        k_pe=k_pe,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def unified_mla_attention_with_output_fake(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_mla_attention_with_output",
    op_func=unified_mla_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_mla_attention_with_output_fake,
    dispatch_key=current_platform.dispatch_key,
)
