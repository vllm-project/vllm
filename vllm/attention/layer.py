# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm.envs as envs
from vllm.attention import AttentionType
from vllm.attention.backends.abstract import AttentionBackend, MLAAttentionImpl
from vllm.attention.backends.registry import _Backend, backend_name_to_enum
from vllm.attention.selector import get_attn_backend
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
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
from vllm.utils import GiB_bytes, direct_register_custom_op

logger = init_logger(__name__)
USE_XFORMERS_OPS = None
try:
    tag_cudagraph_unsafe = (torch._C.Tag.cudagraph_unsafe,)
except AttributeError:
    tag_cudagraph_unsafe = ()  # type: ignore[assignment]


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
    attn_backend: _Backend, use_upstream_fa: bool
) -> tuple[_Backend, Callable]:
    if (
        attn_backend != _Backend.FLASH_ATTN
        and attn_backend != _Backend.ROCM_AITER_FA
        and check_upstream_fa_availability(torch.get_default_dtype())
    ):
        attn_backend = _Backend.FLASH_ATTN
        use_upstream_fa = True

    if current_platform.is_rocm() and attn_backend == _Backend.FLASH_ATTN:
        use_upstream_fa = True

    if attn_backend in {_Backend.FLASH_ATTN, _Backend.ROCM_AITER_FA}:
        if attn_backend == _Backend.ROCM_AITER_FA:
            from aiter import flash_attn_varlen_func
        else:
            if use_upstream_fa:
                from flash_attn import flash_attn_varlen_func
            else:
                from vllm.vllm_flash_attn import flash_attn_varlen_func
    else:
        flash_attn_varlen_func = None

    return attn_backend, flash_attn_varlen_func


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

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            calculate_kv_scales = False
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )

        # The default k/v_scale is set to 1.0. This is ignored
        # when kv-cache is not fp8, and should be used with
        # kv-cache in fp8_e5m2. For kv-cache in fp8_e4m3, we
        # expect the pre-quantized k/v_scale to be loaded along
        # with the model weights.
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        # FlashAttn doesn't support quantizing the kv-cache only
        # but requires q to be quantized as well.
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        # We also keep q/k/v_scale on host (cpu) memory for attention
        # backends that require the scales to be on host instead of on device.
        # e.g. Flashinfer
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0

        # The output scale on host memory. This should be the input scale of
        # the quant op after this attention layer.
        self._o_scale_float: float | None = None

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        quant_method = (
            quant_config.get_quant_method(self, prefix=prefix) if quant_config else None
        )
        if quant_method is not None and not isinstance(
            quant_method, UnquantizedLinearMethod
        ):
            assert isinstance(quant_method, BaseKVCacheMethod)
            # TODO (mgoin): kv cache dtype should be specified in the FP8
            # checkpoint config and become the "auto" behavior
            if self.kv_cache_dtype == "fp8_e5m2":
                raise ValueError(
                    "fp8_e5m2 kv-cache is not supported with fp8 checkpoints."
                )
            # If quantization is enabled, we make "k_scale" and "v_scale"
            # parameters so that it can be loaded from the model checkpoint.
            # The k/v_scale will then be converted back to native float32
            # values after weight loading.
            self.quant_method = quant_method
            self.quant_method.create_weights(self)

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
        self.backend = backend_name_to_enum(self.attn_backend.get_name())
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        self.use_output = self.attn_backend.accept_output_buffer
        compilation_config = get_current_vllm_config().compilation_config
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
            for _ in range(
                get_current_vllm_config().parallel_config.pipeline_parallel_size
            )
        ]

        try:
            self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
            self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
            self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)
        except torch.cuda.OutOfMemoryError as e:
            logger.error("Failed to initialize attention q/k/v range constants: %s", e)
            if torch.cuda.is_available():
                logger.debug("CUDA device: %s", torch.cuda.current_device())
                logger.debug(
                    "Allocated: %.2f GiB", torch.cuda.memory_allocated() / GiB_bytes
                )
                logger.debug(
                    "Reserved: %.2f GiB", torch.cuda.memory_reserved() / GiB_bytes
                )
            raise RuntimeError(
                "Failed to initialize q/k/v range constants. "
                "This may be caused by insufficient memory to allocate "
                "kv cache."
            ) from e

        # for attn backends supporting query quantization
        self.query_quant = None
        if (
            self.kv_cache_dtype.startswith("fp8")
            and self.attn_backend.supports_quant_query_input
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
            query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.zeros(output_shape, dtype=output_dtype, device=query.device)
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
        if hasattr(self.impl, "process_weights_after_loading"):
            self.impl.process_weights_after_loading(act_dtype)

        # FlashInfer requires attention sinks to be float32
        if self.backend == _Backend.FLASHINFER and hasattr(self.impl, "sinks"):
            from vllm.v1.attention.backends.flashinfer import FlashInferImpl

            assert isinstance(self.impl, FlashInferImpl)
            if self.impl.sinks is not None and self.impl.sinks.dtype != torch.float32:
                self.impl.sinks = self.impl.sinks.to(torch.float32)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend


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
        backend = get_vit_attn_backend(head_size=head_size, dtype=dtype)

        # Some auto-selected backends can be upgraded
        # to upstream flash attention if available.
        # If vllm native fa is selected, we use it directly.
        use_upstream_fa = False

        if current_platform.is_xpu():
            # currently, only torch_sdpa is supported on xpu
            self.attn_backend = _Backend.TORCH_SDPA
        else:
            self.attn_backend = (
                backend
                if backend
                in {
                    _Backend.TORCH_SDPA,
                    _Backend.XFORMERS,
                    _Backend.PALLAS,
                    _Backend.ROCM_AITER_FA,
                    _Backend.FLASH_ATTN,
                }
                else _Backend.TORCH_SDPA
            )

        self.attn_backend, self._flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(
                self.attn_backend,
                use_upstream_fa,
            )
        )

        if self.attn_backend == _Backend.XFORMERS and not check_xformers_availability():
            self.attn_backend = _Backend.TORCH_SDPA

        self.is_flash_attn_backend = self.attn_backend in {
            _Backend.FLASH_ATTN,
            _Backend.ROCM_AITER_FA,
        }

        # this condition is just to make sure that the
        # use_upstream_fa in the log is correct
        if current_platform.is_rocm() and self.attn_backend == _Backend.FLASH_ATTN:
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
        elif self.attn_backend == _Backend.XFORMERS:
            from xformers import ops as xops

            out = xops.memory_efficient_attention_forward(
                query, key, value, scale=self.scale
            )
        elif self.attn_backend == _Backend.TORCH_SDPA:
            query, key, value = (x.transpose(1, 2) for x in (query, key, value))
            out = F.scaled_dot_product_attention(query, key, value, scale=self.scale)
            out = out.transpose(1, 2)
        elif self.attn_backend == _Backend.PALLAS:
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
        self.kv_cache_dtype = kv_cache_dtype

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

        # Align with Attention's scale attributes for MLA backends.

        self.calculate_kv_scales = calculate_kv_scales
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)
        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._prob_scale = torch.tensor(1.0, dtype=torch.float32)

        # Host-side mirrors used by some attention backends
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        self._o_scale_float: float | None = None

        self.use_sparse = use_sparse

        # Initialize q/k/v range constants.
        try:
            self.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
            self.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
            self.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)
        except torch.cuda.OutOfMemoryError:
            # Keep defaults if allocation fails; not critical for init.
            pass

    def forward(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        if self.use_direct_call:
            forward_context: ForwardContext = get_forward_context()
            attn_metadata = forward_context.attn_metadata
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]

            # Mirror Attention.forward scale calculation path
            if self.calculate_kv_scales and getattr(
                attn_metadata, "enable_kv_scales_calculation", False
            ):
                self.calc_kv_scales(q, kv_c_normed, k_pe)

            if self.attn_backend.accept_output_buffer:
                output = torch.zeros(output_shape, dtype=q.dtype, device=q.device)
                self.impl.forward(
                    self,
                    q,
                    kv_c_normed,
                    k_pe,
                    self_kv_cache,
                    attn_metadata,
                    output=output,
                )
                return output
            else:
                return self.impl.forward(
                    self, q, kv_c_normed, k_pe, self_kv_cache, attn_metadata
                )
        else:
            if self.attn_backend.accept_output_buffer:
                output = torch.zeros(output_shape, dtype=q.dtype, device=q.device)
                torch.ops.vllm.unified_mla_attention_with_output(
                    q,
                    kv_c_normed,
                    k_pe,
                    output,
                    self.layer_name,
                )
                return output
            else:
                # We can still access forward context to check calculation flag
                if self.calculate_kv_scales:
                    forward_context = get_forward_context()
                    attn_metadata = forward_context.attn_metadata
                    if isinstance(attn_metadata, dict):
                        attn_metadata = attn_metadata[self.layer_name]
                    if getattr(attn_metadata, "enable_kv_scales_calculation", False):
                        self.calc_kv_scales(q, kv_c_normed, k_pe)
                return torch.ops.vllm.unified_mla_attention(
                    q,
                    kv_c_normed,
                    k_pe,
                    self.layer_name,
                )

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


def wait_for_kv_layer_from_connector(layer_name: str):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.wait_for_layer_load(layer_name)


def maybe_save_kv_layer_to_connector(
    layer_name: str,
    kv_cache_layer: list[torch.Tensor],
):
    if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
        return

    connector = get_kv_transfer_group()

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is None:
        return
    assert isinstance(attn_metadata, dict)
    connector.save_kv_layer(layer_name, kv_cache_layer, attn_metadata[layer_name])


def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata

    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    if attn_metadata is None or not getattr(
        attn_metadata, "enable_kv_scales_calculation", False
    ):
        return

    self = forward_context.no_compile_layers[layer_name]
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


def unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
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
    tags=tag_cudagraph_unsafe,
)


def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
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

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


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
    tags=tag_cudagraph_unsafe,
)


def unified_mla_attention(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    wait_for_kv_layer_from_connector(layer_name)

    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self: MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    output = self.impl.forward(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata)

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)
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


def unified_mla_attention_with_output(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    wait_for_kv_layer_from_connector(layer_name)
    forward_context: ForwardContext = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]
    self: MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = self.kv_cache[forward_context.virtual_engine]
    self.impl.forward(
        self,
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )

    maybe_save_kv_layer_to_connector(layer_name, kv_cache)


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
