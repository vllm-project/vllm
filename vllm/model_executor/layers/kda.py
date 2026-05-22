# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from einops import rearrange
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from .fla.ops.kda import (
    FusedRMSNormGated,
    chunk_kda,
    fused_kda_gate,
    fused_recurrent_kda,
)
from .linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from .mamba.abstract import MambaBase
from .mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from .mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from .quantization.base_config import QuantizationConfig

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# FlyDSL gated delta rule decode kernel (opt-in, ROCm only)
# ---------------------------------------------------------------------------
# ``aiter.ops.flydsl.linear_attention_kernels.flydsl_gdr_decode`` is a
# FlyDSL-compiled KDA decode kernel that fuses the ``fused_kda_gate`` and the
# recurrent gated-delta-rule update. It is not always available (requires a
# recent AITER + FlyDSL on ROCm), so we look it up lazily and cache the
# resolution. Gated behind ``VLLM_ROCM_USE_AITER_FLYDSL_KDA`` (default off).
_FLYDSL_KDA_RESOLVED: bool = False
_FLYDSL_KDA_KERNEL = None


def _maybe_get_flydsl_kda_kernel():
    """Return the FlyDSL gated delta rule decode kernel, or ``None``.

    Resolution is cached on first call. We guard with:
      * ``VLLM_ROCM_USE_AITER_FLYDSL_KDA`` (opt-in env flag)
      * ``current_platform.is_rocm()`` (kernel is ROCm-only)
      * ``ImportError`` fallback if AITER is too old / missing the module
    """
    global _FLYDSL_KDA_RESOLVED, _FLYDSL_KDA_KERNEL
    if _FLYDSL_KDA_RESOLVED:
        return _FLYDSL_KDA_KERNEL
    _FLYDSL_KDA_RESOLVED = True

    if not envs.VLLM_ROCM_USE_AITER_FLYDSL_KDA:
        return None
    if not current_platform.is_rocm():
        logger.debug(
            "VLLM_ROCM_USE_AITER_FLYDSL_KDA=1 set on non-ROCm platform; "
            "ignoring and falling back to FLA triton KDA decode."
        )
        return None
    try:
        from aiter.ops.flydsl.linear_attention_kernels import (  # type: ignore[import-not-found] # noqa: E501
            flydsl_gdr_decode,
        )
    except ImportError as e:
        logger.info(
            "VLLM_ROCM_USE_AITER_FLYDSL_KDA=1 but aiter.ops.flydsl."
            "linear_attention_kernels.flydsl_gdr_decode is unavailable (%s); "
            "falling back to FLA triton KDA decode.",
            e,
        )
        return None
    _FLYDSL_KDA_KERNEL = flydsl_gdr_decode
    logger.info(
        "KimiDeltaAttention decode: using FlyDSL gated delta rule kernel "
        "(aiter.ops.flydsl.linear_attention_kernels.flydsl_gdr_decode)."
    )
    return _FLYDSL_KDA_KERNEL


def _flydsl_kda_decode(
    flydsl_kernel,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g1_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    recurrent_state: torch.Tensor,
) -> torch.Tensor:
    """Run the FlyDSL KDA decode kernel and return ``core_attn_out_non_spec``.

    Input tensor shapes (matching the FLA path in ``_forward``):
      * q, k, v:        ``[1, N, H, D]`` where N = num_decode_tokens
      * g1_raw:         ``[1, N, H*D]`` (pre-gate output of ``f_b_proj``)
      * beta_raw:       ``[1, N, H]`` (pre-sigmoid output of ``b_proj``)
      * A_log:          ``[1, 1, H, 1]`` (weight tensor)
      * dt_bias:        ``[H*D]`` (weight tensor)
      * ssm_state_indices: ``[N, ...]`` cache-slot indices into
        ``recurrent_state`` (the exact layout comes from
        ``GDNAttentionMetadata``)
      * recurrent_state: ``[max_cache_slots, H, D, D]`` (mutated in-place)

    The FlyDSL kernel expects per-sequence batched tensors (``[bs, T=1, ...]``
    for decode), so we reshape the [1, N, ...] layout to [N, 1, ...]. This is
    a stride-compatible view since the outer dim is 1.

    Returns ``core_attn_out_non_spec`` of shape ``[1, N, H, D]`` matching the
    FLA path's return value (so downstream code is shape-agnostic).
    """
    _, num_tokens, num_heads, head_dim = q.shape

    # [1, N, H, D] -> [N, 1, H, D]
    q_fly = q.reshape(num_tokens, 1, num_heads, head_dim)
    k_fly = k.reshape(num_tokens, 1, num_heads, head_dim)
    v_fly = v.reshape(num_tokens, 1, num_heads, head_dim)
    # [1, N, H*D] -> [N, 1, H*D]
    a_fly = g1_raw.reshape(num_tokens, 1, num_heads * head_dim)
    # [1, N, H] -> [N, 1, H]
    b_fly = beta_raw.reshape(num_tokens, 1, num_heads)

    # The FlyDSL kernel expects ``indices`` of shape ``[bs]`` (one cache slot
    # per decoded sequence). ``ssm_state_indices`` from GDN attention metadata
    # may be laid out as ``[bs, 1]`` or ``[bs]`` depending on the decode path;
    # normalize to ``[bs]`` and cast to int32 as expected by FlyDSL.
    idx = ssm_state_indices
    if idx.dim() > 1:
        idx = idx[..., 0]
    idx = idx[:num_tokens].to(dtype=torch.int32)

    out = torch.empty(
        num_tokens, 1, num_heads, head_dim, device=q.device, dtype=q.dtype
    )
    flydsl_kernel(
        query=q_fly,
        key=k_fly,
        value=v_fly,
        a=a_fly,
        b=b_fly,
        dt_bias=dt_bias,
        A_log=A_log.view(-1),
        indices=idx,
        state=recurrent_state,
        out=out,
        use_qk_l2norm=True,
        need_shuffle_state=False,
    )
    return out.reshape(1, num_tokens, num_heads, head_dim)


def kda_attention(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        q_proj_states=q_proj_states,
        k_proj_states=k_proj_states,
        v_proj_states=v_proj_states,
        g1_raw=g1_raw,
        beta_raw=beta_raw,
        core_attn_out=core_attn_out,
    )


def kda_attention_fake(
    q_proj_states: torch.Tensor,
    k_proj_states: torch.Tensor,
    v_proj_states: torch.Tensor,
    g1_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_attention",
    op_func=kda_attention,
    mutates_args=["core_attn_out"],
    fake_impl=kda_attention_fake,
)


class KimiDeltaAttention(nn.Module, MambaBase):
    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.GDN_ATTN

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size, self.num_heads, self.head_dim, conv_kernel_size=self.conv_size
        )

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        quant_config: QuantizationConfig | None = None,
        cache_config: CacheConfig | None = None,
        model_config: ModelConfig | None = None,
        rms_norm_eps: float = 1e-5,
        prefix: str = "",
        **kwargs,
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = hidden_size
        self.model_config = model_config
        self.cache_config = cache_config
        if model_config is None:
            raise ValueError("model_config must be provided")
        kda_config = model_config.linear_attn_config  # type: ignore[attr-defined]
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        self.layer_idx = layer_idx
        self.prefix = prefix
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        projection_size = self.head_dim * self.num_heads
        self.conv_size = kda_config["short_conv_kernel_size"]

        self.q_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj",
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj",
        )

        self.f_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_a_proj",
        )

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(divide(projection_size, self.tp_size), dtype=torch.float32)
        )

        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        self.b_proj = ColumnParallelLinear(
            self.hidden_size,
            self.num_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.b_proj",
        )

        self.q_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.q_conv1d",
        )
        self.k_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.k_conv1d",
        )
        self.v_conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.v_conv1d",
        )
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.q_conv1d.weight.data = self.q_conv1d.weight.data.unsqueeze(1)
        self.k_conv1d.weight.data = self.k_conv1d.weight.data.unsqueeze(1)
        self.v_conv1d.weight.data = self.v_conv1d.weight.data.unsqueeze(1)

        self.A_log = nn.Parameter(
            torch.empty(1, 1, self.local_num_heads, 1, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(2)})

        self.g_a_proj = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_a_proj",
        )
        self.g_b_proj = ColumnParallelLinear(
            self.head_dim,
            projection_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.g_b_proj",
        )
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=rms_norm_eps, activation="sigmoid"
        )
        self.o_proj = RowParallelLinear(
            projection_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        num_tokens = hidden_states.size(0)
        q = self.q_proj(hidden_states)[0]
        k = self.k_proj(hidden_states)[0]
        v = self.v_proj(hidden_states)[0]

        # NOTE: we pass the *raw* (pre-gate, pre-sigmoid) tensors through the
        # custom op. ``fused_kda_gate`` and ``.sigmoid()`` were previously
        # applied here, but the FlyDSL KDA decode kernel (opt-in, see
        # ``_maybe_get_flydsl_kda_kernel``) fuses the gate computation
        # internally and needs the raw pre-activation values. The FLA
        # (triton) fallback applies the gate+sigmoid inside ``_forward`` so
        # its behavior is unchanged.
        beta_raw = self.b_proj(hidden_states)[0].float()
        g1_raw = self.f_b_proj(self.f_a_proj(hidden_states)[0])[0]
        beta_raw = beta_raw.unsqueeze(0)
        g1_raw = g1_raw.unsqueeze(0)

        g_proj_states = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0]
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)

        core_attn_out = torch.zeros(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.kda_attention(
            q,
            k,
            v,
            g1_raw,
            beta_raw,
            core_attn_out,
            self.prefix,
        )
        core_attn_out = self.o_norm(core_attn_out, g2)
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        output[:] = self.o_proj(core_attn_out)[0]

    def _forward(
        self,
        q_proj_states: torch.Tensor,
        k_proj_states: torch.Tensor,
        v_proj_states: torch.Tensor,
        g1_raw: torch.Tensor,
        beta_raw: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata

        if attn_metadata_raw is None:
            #     # V1 profile run
            return

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        has_initial_state = attn_metadata_narrowed.has_initial_state
        non_spec_query_start_loc = attn_metadata_narrowed.non_spec_query_start_loc
        non_spec_state_indices_tensor = (
            attn_metadata_narrowed.non_spec_state_indices_tensor
        )  # noqa: E501
        num_actual_tokens = attn_metadata_narrowed.num_actual_tokens
        constant_caches = self.kv_cache

        q_proj_states = q_proj_states[:num_actual_tokens]
        k_proj_states = k_proj_states[:num_actual_tokens]
        v_proj_states = v_proj_states[:num_actual_tokens]
        # NOTE: g1_raw and beta_raw are pre-gate / pre-sigmoid. The gate
        # (``fused_kda_gate``) and sigmoid are applied below inside the
        # prefill / FLA-fallback decode branches; the FlyDSL decode branch
        # consumes the raw values directly. We preserve the previous slicing
        # semantics (slice on dim 0, which is size 1 after ``unsqueeze(0)``
        # in ``forward``) to keep behavior identical on the FLA path.
        g1_raw = g1_raw[:num_actual_tokens]
        beta_raw = beta_raw[:num_actual_tokens]

        (conv_state_q, conv_state_k, conv_state_v, recurrent_state) = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state_q = conv_state_q.transpose(-1, -2)
            conv_state_k = conv_state_k.transpose(-1, -2)
            conv_state_v = conv_state_v.transpose(-1, -2)

        q_conv_weights = self.q_conv1d.weight.view(
            self.q_conv1d.weight.size(0), self.q_conv1d.weight.size(2)
        )
        k_conv_weights = self.k_conv1d.weight.view(
            self.k_conv1d.weight.size(0), self.k_conv1d.weight.size(2)
        )
        v_conv_weights = self.v_conv1d.weight.view(
            self.v_conv1d.weight.size(0), self.v_conv1d.weight.size(2)
        )
        if attn_metadata_narrowed.num_prefills > 0:
            q_proj_states = q_proj_states.transpose(0, 1)
            k_proj_states = k_proj_states.transpose(0, 1)
            v_proj_states = v_proj_states.transpose(0, 1)
            q = causal_conv1d_fn(
                q_proj_states,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_states=conv_state_q,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            k = causal_conv1d_fn(
                k_proj_states,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_states=conv_state_k,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
            v = causal_conv1d_fn(
                v_proj_states,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_states=conv_state_v,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata_narrowed,
            ).transpose(0, 1)
        else:
            assert non_spec_state_indices_tensor is not None
            decode_conv_indices = non_spec_state_indices_tensor[
                : attn_metadata_narrowed.num_actual_tokens
            ]
            q = causal_conv1d_update(
                q_proj_states,
                conv_state_q,
                q_conv_weights,
                self.q_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            k = causal_conv1d_update(
                k_proj_states,
                conv_state_k,
                k_conv_weights,
                self.k_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )
            v = causal_conv1d_update(
                v_proj_states,
                conv_state_v,
                v_conv_weights,
                self.v_conv1d.bias,
                activation="silu",
                conv_state_indices=decode_conv_indices,
                validate_data=True,
            )

        q, k, v = map(
            lambda x: rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim), (q, k, v)
        )

        if attn_metadata_narrowed.num_prefills > 0:
            assert non_spec_state_indices_tensor is not None
            assert has_initial_state is not None
            # Prefill path: apply gate + sigmoid then run chunk_kda (triton).
            # Unchanged vs. upstream: identical numerics, just moved the
            # gate/sigmoid from ``forward()`` to here.
            g1 = fused_kda_gate(
                g1_raw, self.A_log, self.head_dim, g_bias=self.dt_bias
            )
            beta = beta_raw.sigmoid()
            zero_idx = non_spec_state_indices_tensor[~has_initial_state]
            recurrent_state[zero_idx] = 0
            initial_state = recurrent_state[non_spec_state_indices_tensor].contiguous()
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g1,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            # Init cache
            recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
        else:
            assert non_spec_query_start_loc is not None
            # Decode path: try FlyDSL first (opt-in, ROCm-only); otherwise
            # fall back to the FLA triton ``fused_recurrent_kda``.
            flydsl_kernel = _maybe_get_flydsl_kda_kernel()
            core_attn_out_non_spec = None
            if flydsl_kernel is not None:
                try:
                    core_attn_out_non_spec = _flydsl_kda_decode(
                        flydsl_kernel,
                        q=q,
                        k=k,
                        v=v,
                        g1_raw=g1_raw,
                        beta_raw=beta_raw,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        ssm_state_indices=non_spec_state_indices_tensor,
                        recurrent_state=recurrent_state,
                    )
                except Exception as e:
                    # Disable on first failure so we don't retry every step;
                    # emit a single warning per worker and fall back to FLA.
                    global _FLYDSL_KDA_KERNEL
                    _FLYDSL_KDA_KERNEL = None
                    logger.warning(
                        "FlyDSL KDA decode failed (%s: %s); falling back to "
                        "FLA triton fused_recurrent_kda for the remainder of "
                        "this run. Set VLLM_ROCM_USE_AITER_FLYDSL_KDA=0 to "
                        "silence this path.",
                        type(e).__name__,
                        e,
                    )
                    core_attn_out_non_spec = None
            if core_attn_out_non_spec is None:
                g1 = fused_kda_gate(
                    g1_raw, self.A_log, self.head_dim, g_bias=self.dt_bias
                )
                beta = beta_raw.sigmoid()
                (
                    core_attn_out_non_spec,
                    last_recurrent_state,
                ) = fused_recurrent_kda(
                    q=q,
                    k=k,
                    v=v,
                    g=g1,
                    beta=beta,
                    initial_state=recurrent_state,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=non_spec_query_start_loc[
                        : attn_metadata_narrowed.num_decodes + 1
                    ],
                    ssm_state_indices=non_spec_state_indices_tensor,
                )
        core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
            0, :num_actual_tokens
        ]
