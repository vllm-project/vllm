# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer."""

from typing import TYPE_CHECKING, TypeVar, cast

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.common import MLACommonMetadata

import torch
import torch.nn as nn
from tqdm import tqdm

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.attention.ops.common import cp_lse_ag_out_rs
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.attention.selector import get_attn_backend
from vllm.attention.utils.kv_sharing_utils import validate_kv_sharing_target
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.distributed.parallel_state import get_dcp_group, is_global_first_rank
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx9
else:
    on_gfx9 = lambda *args, **kwargs: False

M = TypeVar("M", bound="MLACommonMetadata")


FP8_DTYPE = current_platform.fp8_dtype()
logger = init_logger(__name__)


def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0


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

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would
    # be an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e., should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    if should_load_quant_weights(quant_method):
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
        head_size_v: int | None = None,
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
        self.quant_config = quant_config
        self.layer_name = prefix

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(
            self,
            self.quant_config,
            self.layer_name,
            kv_cache_dtype,
            calculate_kv_scales,
        )

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.has_sink = extra_impl_args.get("sinks") is not None

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

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
                use_mm_prefix=self.use_mm_prefix,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend

        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and vllm_is_batch_invariant()
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

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
            and self.impl.supports_quant_query_input
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
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if self.use_output:
            if output_shape is None:
                # Handle both 2D [num_tokens, hidden] and
                # 3D [num_tokens, heads, head_dim] query
                num_tokens = query.shape[0]
                output_shape = torch.Size(
                    (num_tokens, self.num_heads * self.head_size_v)
                )
            output_shape = output_shape if output_shape is not None else query.shape
            output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]
            # Reshape the query, key, and value tensors.
            # NOTE(woosuk): We do this outside the custom op to minimize the
            # CPU overheads from the non-CUDA-graph regions.
            query = query.view(-1, self.num_heads, self.head_size)
            output = output.view(-1, self.num_heads, self.head_size_v)
            if key is not None:
                key = key.view(-1, self.num_kv_heads, self.head_size)
            if value is not None:
                value = value.view(-1, self.num_kv_heads, self.head_size_v)
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
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
            )


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
        self.quant_config = quant_config

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(
            self,
            self.quant_config,
            self.layer_name,
            kv_cache_dtype,
            calculate_kv_scales,
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

        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and vllm_is_batch_invariant()
            and (
                self.attn_backend.get_name() == "TRITON_MLA"
                or self.attn_backend.get_name() == "FLASHINFER"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for TRITON_MLA / FLASHINFER "
                "with batch invariance, as it is not yet supported.",
                scope="local",
            )
            cache_config.enable_prefix_caching = False

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
            use_sparse=use_sparse,
            **extra_impl_args,
        )

        # Store kv_b_proj for use in methods
        self.kv_b_proj = kv_b_proj

        # Initialize attributes from impl for direct method access
        # Use getattr since these exist on MLACommonImpl but not on abstract base
        self.is_aiter_triton_fp8_bmm_enabled: bool = getattr(
            self.impl, "is_aiter_triton_fp8_bmm_enabled", False
        )
        self.dcp_world_size: int | None = getattr(self.impl, "dcp_world_size", None)
        self.chunked_prefill_workspace_size: int = getattr(
            self.impl, "chunked_prefill_workspace_size", 0
        )
        self._pad_v: bool = getattr(self.impl, "_pad_v", False)
        self._use_fi_prefill: bool = getattr(self.impl, "_use_fi_prefill", False)

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

        # Weight matrices set by process_weights_after_loading
        # For non-aiter path:
        self.W_UV: torch.Tensor | None = None
        self.W_UK_T: torch.Tensor | None = None
        # For aiter fp8 path:
        self.W_K: torch.Tensor | None = None
        self.W_K_scale: torch.Tensor | None = None
        self.W_V: torch.Tensor | None = None
        self.W_V_scale: torch.Tensor | None = None

        # Get q_pad_num_heads from impl (set in impl's __init__)
        self.q_pad_num_heads: int | None = getattr(self.impl, "q_pad_num_heads", None)

    # Property accessors for impl methods
    @property
    def _run_prefill_context_chunk(self):
        return getattr(self.impl, "_run_prefill_context_chunk", None)

    @property
    def _run_prefill_new_tokens(self):
        return getattr(self.impl, "_run_prefill_new_tokens", None)

    def _forward_decode(self, *args, **kwargs):
        return self.impl._forward_decode(  # type: ignore[attr-defined]
            *args, **kwargs
        )

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
                self.forward_impl(
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
                return self.forward_impl(
                    self, q, kv_c_normed, k_pe, self_kv_cache, attn_metadata
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

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        from vllm.v1.attention.backends.mla.common import (
            dynamic_per_batched_tensor_quant,
        )

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute: {WEIGHT_NAMES}."
            )

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(
                    layer.input_size_per_partition,
                    dtype=act_dtype,
                    device=get_layer_weight(layer).device,
                )
                dequant_weights = layer.quant_method.apply(layer, eye, bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            return layer.weight

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
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

        if self.is_aiter_triton_fp8_bmm_enabled:
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

        # Also set weight matrices on impl for sparse backends that have
        # their own forward() using self.W_UK_T etc.
        if self.is_aiter_triton_fp8_bmm_enabled:
            self.impl.W_K = self.W_K
            self.impl.W_K_scale = self.W_K_scale
            self.impl.W_V = self.W_V
            self.impl.W_V_scale = self.W_V_scale
        else:
            self.impl.W_UV = self.W_UV
            self.impl.W_UK_T = self.W_UK_T

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

        if self.is_aiter_triton_fp8_bmm_enabled:
            out = out.view(-1, self.num_heads, self.v_head_dim)
            # Multiply + Transpose (N, B, L) x (N, L, V)->(N, B, V)->(B, N, V)
            x = rocm_aiter_ops.triton_fp8_bmm(
                x, self.W_V, self.W_V_scale, group_size=128, transpose_bm=True, YQ=out
            )
        else:
            # Convert from (B, N * V) to (N, B, V)
            out = out.view(-1, self.num_heads, self.v_head_dim).transpose(0, 1)

            # Multiply (N, B, L) x (N, L, V) -> (N, B, V)
            torch.bmm(x, self.W_UV, out=out)  # Reuse "out" to make it "hot"

            # Convert from (N, B, V) to (B, N * V)
            out_new = out.transpose(0, 1).reshape(-1, self.num_heads * self.v_head_dim)

            # Adjust output buffer shape back to the original (B, N * V)
            N, B, V = out.shape
            out.resize_((B, N * V))
            out.copy_(out_new)  # Copy result

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
        # Direct copies with efficient broadcasting
        k[..., : k_nope.shape[-1]] = k_nope
        k[..., k_nope.shape[-1] :] = k_pe
        return k

    def _compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        k_scale: torch.Tensor,
    ):
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None

        output = None
        iters = len(prefill_metadata.chunked_context.seq_tot)
        workspace = prefill_metadata.chunked_context.workspace
        for i in range(iters):
            toks = prefill_metadata.chunked_context.seq_tot[i]
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

            kv_c_normed = workspace[:toks][..., : self.kv_lora_rank]
            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)

            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

            k = self._concat_k_nope_k_pe(k_nope, k_pe)

            assert self.impl._run_prefill_context_chunk is not None
            attn_output, attn_softmax_lse = self.impl._run_prefill_context_chunk(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _context_parallel_compute_prefill_context(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        k_scale: torch.Tensor,
        dcp_world_size: int,
    ):
        from vllm.v1.attention.backends.mla.common import reorg_kvcache

        assert k_scale is None, "DCP not support scaled kvcache now."
        assert attn_metadata.prefill is not None
        prefill_metadata = attn_metadata.prefill
        assert prefill_metadata.chunked_context is not None
        assert prefill_metadata.chunked_context.padded_local_chunk_seq_lens is not None
        assert prefill_metadata.chunked_context.local_context_lens_allranks is not None
        assert prefill_metadata.chunked_context.padded_local_cu_seq_lens is not None
        assert prefill_metadata.chunked_context.cu_seq_lens_lst is not None
        assert prefill_metadata.chunked_context.chunk_size is not None

        output = None
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
            # |<- use for loca_gather ->|<--------- use for allgather -------->|
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

            assert self.impl._run_prefill_context_chunk is not None
            attn_output, attn_softmax_lse = self.impl._run_prefill_context_chunk(
                prefill=prefill_metadata,
                chunk_idx=i,
                q=q,
                k=k,
                v=v,
            )

            if output is None:
                output = attn_output
                output_lse = attn_softmax_lse
            else:
                output_tmp = torch.empty_like(output)
                output_lse_tmp = torch.empty_like(output_lse)
                merge_attn_states(
                    output=output_tmp,
                    output_lse=output_lse_tmp,
                    prefix_output=output,
                    prefix_lse=output_lse,
                    suffix_output=attn_output,
                    suffix_lse=attn_softmax_lse,
                )
                output = output_tmp
                output_lse = output_lse_tmp

        return output, output_lse

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        k_scale: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        # TODO (zyongye): Prefill function here
        assert attn_metadata.prefill is not None
        assert self.dcp_world_size is not None

        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = self._concat_k_nope_k_pe(k_nope, k_pe)

        assert self.impl._run_prefill_new_tokens is not None
        output_prefill = self.impl._run_prefill_new_tokens(
            prefill=attn_metadata.prefill,
            q=q,
            k=k,
            v=v,
            return_softmax_lse=has_context,
        )

        if has_context:
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

            # unpad if necessary
            if self._pad_v:
                context_output = context_output[..., : v.shape[-1]]
                suffix_output = suffix_output[..., : v.shape[-1]]

            output = output.view(-1, self.num_heads, self.v_head_dim)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )
        else:
            output_prefill = output_prefill[..., : v.shape[-1]].flatten(start_dim=-2)
            output.copy_(output_prefill)

    def forward_impl(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: M,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Direct Copy Paste of MLACommonImpl.forward()"""

        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for MLACommonImpl"
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
            return output.fill_(0)

        if self.dcp_world_size is None:
            self.dcp_world_size = get_dcp_group().world_size

        fp8_attention = self.kv_cache_dtype.startswith("fp8")

        num_actual_toks = attn_metadata.num_actual_tokens

        # Inputs and outputs may be padded for CUDA graphs
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        assert (
            attn_metadata.num_decodes is not None
            and attn_metadata.num_prefills is not None
            and attn_metadata.num_decode_tokens is not None
        )

        has_decode = attn_metadata.num_decodes > 0
        has_prefill = attn_metadata.num_prefills > 0
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]

        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # write the latent and rope to kv cache
        if kv_cache.numel() > 0:
            ops.concat_and_cache_mla(
                k_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype=self.kv_cache_dtype,
                scale=layer._k_scale,
            )

        if fp8_attention:
            kv_cache = kv_cache.view(current_platform.fp8_dtype())

        if has_prefill:
            self._forward_prefill(
                prefill_q,
                prefill_k_c_normed,
                prefill_k_pe,
                kv_cache,
                attn_metadata,
                layer._k_scale,
                output=output[num_decode_tokens:],
            )

        if has_decode:
            assert attn_metadata.decode is not None

            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            # Convert from (B, N, P) to (N, B, P)
            decode_q_nope = decode_q_nope.transpose(0, 1)

            if self.q_pad_num_heads is not None:
                B, N, L = decode_q_pe.shape
                decode_pe_padded = decode_q_pe.new_empty((B, self.q_pad_num_heads, L))
                decode_pe_padded.resize_((B, N, L))
                decode_pe_padded.copy_(decode_q_pe)
                decode_q_pe = decode_pe_padded

            if self.is_aiter_triton_fp8_bmm_enabled:
                # Multiply+Transpose (N, B, P)x(N, P, L)->(N, B, L)->(B, N, L)
                decode_ql_nope = rocm_aiter_ops.triton_fp8_bmm(
                    decode_q_nope,
                    self.W_K,
                    self.W_K_scale,
                    group_size=128,
                    transpose_bm=True,
                )
            else:
                # Pads the head_dim if necessary (for the underlying kernel)
                N, B, P = decode_q_nope.shape
                assert self.W_UK_T is not None
                _, _, L = self.W_UK_T.shape

                if self.q_pad_num_heads is not None:
                    decode_ql_nope = decode_q_nope.new_empty(
                        (self.q_pad_num_heads, B, L)
                    )
                    decode_ql_nope.resize_((N, B, L))
                else:
                    decode_ql_nope = decode_q_nope.new_empty((N, B, L))

                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
                torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)

                # Convert from (N, B, L) to (B, N, L)
                decode_ql_nope = decode_ql_nope.transpose(0, 1)

            if fp8_attention:
                ql_nope_shape = decode_ql_nope.shape
                q_pe_shape = decode_q_pe.shape
                assert decode_ql_nope.shape[0] == decode_q_pe.shape[0]
                assert decode_ql_nope.shape[1] == decode_q_pe.shape[1]
                decode_q_shape = (
                    ql_nope_shape[0],
                    ql_nope_shape[1],
                    ql_nope_shape[2] + q_pe_shape[2],
                )
                # Using empty and copy since torch.cat introduces significant overhead.
                decode_q0 = torch.empty(
                    decode_q_shape,
                    device=decode_ql_nope.device,
                    dtype=decode_ql_nope.dtype,
                )
                decode_q0[..., : ql_nope_shape[2]].copy_(decode_ql_nope)
                decode_q0[..., ql_nope_shape[2] :].copy_(decode_q_pe)

                decode_q, _ = ops.scaled_fp8_quant(
                    decode_q0.view(decode_q_shape[0], -1),
                    layer._q_scale,
                )
                decode_q = decode_q.view(decode_q_shape)
            else:
                decode_q = (decode_ql_nope, decode_q_pe)

            if self.dcp_world_size > 1:
                assert not fp8_attention, "DCP not support fp8 kvcache now."
                # concatenate decode_ql_nope and decode_q_pe -> (B, N, L + P)
                decode_q = torch.cat(decode_q, dim=-1)
                # decode_q do allgather in head dim.
                decode_q = get_dcp_group().all_gather(decode_q, dim=1)

            # call decode attn
            attn_out, lse = self._forward_decode(
                decode_q, kv_cache, attn_metadata, layer
            )

            # correct dcp attn_out with lse.
            if self.dcp_world_size > 1:
                attn_out = cp_lse_ag_out_rs(
                    attn_out,
                    lse,
                    get_dcp_group(),
                    is_lse_base_on_e=not getattr(self, "_use_fi_prefill", False),
                )

            # v_up projection
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])
        return output_padded


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
    output = self.forward_impl(self, q, kv_c_normed, k_pe, kv_cache, attn_metadata)

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
        self,
        q,
        kv_c_normed,
        k_pe,
        kv_cache,
        attn_metadata,
        output,
        output_scale,
        output_block_scale,
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
