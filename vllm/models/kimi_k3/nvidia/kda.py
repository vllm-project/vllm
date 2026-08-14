# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from einops import rearrange
from torch import nn
from torch.nn.parameter import Parameter

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_rank
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.gather_initial_states import (
    gather_initial_states,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.parameter import BasevLLMParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.kimi_k3.nvidia.kda_metadata import (
    KimiK3KDAAttentionBackend,
    KimiK3KDAMetadata,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backend import AttentionBackend

logger = init_logger(__name__)

_KDA_GATE_LOGBOUND_MIN = -5.0


def a_log_weight_loader(
    shard_axis: int,
) -> Callable[[torch.Tensor, torch.Tensor], None]:
    """Load KDA A_log stored as either old 4D or current 1D weights."""

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()
        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size

        if loaded_weight.dim() == 4:
            assert loaded_weight.shape[:2] == (1, 1), (
                f"Expected old A_log shape (1, 1, H, 1), got {loaded_weight.shape}"
            )
            assert loaded_weight.shape[-1] == 1, (
                f"Expected old A_log last dim to be 1, got {loaded_weight.shape}"
            )
            loaded_weight = loaded_weight.view(loaded_weight.shape[2])

        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)
        return default_weight_loader(param, loaded_weight)

    return loader


class _KimiGDNMergedColumnParallelLinear(MergedColumnParallelLinear):
    """Merged projection with one output replicated across TP ranks."""

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        replicated_shard_id: int,
        tp_size: int,
        **kwargs,
    ) -> None:
        self.replicated_shard_id = replicated_shard_id
        output_sizes = output_sizes.copy()
        output_sizes[replicated_shard_id] *= tp_size
        super().__init__(input_size, output_sizes, **kwargs)

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == self.replicated_shard_id:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank


def is_fused_kda_decode_supported(
    num_heads: int,
    head_dim: int,
    conv_width: int,
    num_spec: int,
    input_dtype: torch.dtype,
    conv_state_dtype: torch.dtype,
) -> bool:
    if (
        num_heads not in (12, 24, 48, 96)
        or head_dim != 128
        or conv_width != 4
        or num_spec != 0
        or input_dtype != torch.bfloat16
        or conv_state_dtype != torch.bfloat16
        or is_conv_state_dim_first()
        or not hasattr(torch.ops._C, "fused_kda_decode")
    ):
        return False
    # SM90 is architecture-specific; SM10x and SM12x use family binaries.
    return (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )


def is_flashkda_supported(
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> bool:
    if not current_platform.is_cuda():
        return False
    capability = current_platform.get_device_capability()
    return (
        capability is not None
        and capability.major in (9, 10, 12)
        and head_dim == 128
        and dtype == torch.bfloat16
        and lower_bound is not None
    )


def _flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    import vllm._flashkda_C  # noqa: F401

    out = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    final_state = torch.empty_like(initial_state)
    workspace = torch.empty(
        torch.ops._flashkda_C.get_workspace_size(
            q.shape[0] * q.shape[1],
            q.shape[2],
            cu_seqlens.numel() - 1,
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    # FlashKDA hardcodes dense Q/K/V/G strides. Beta may be row-strided because
    # FlashKDA materializes its transposed [H, T] layout internally.
    # TODO: Teach FlashKDA to consume beta in [T, H] layout directly instead
    # of transposing it to contiguous [H, T] storage internally.
    torch.ops._flashkda_C.fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta,
        q.shape[-1] ** -0.5,
        out,
        workspace,
        A_log.contiguous(),
        dt_bias.view(-1, q.shape[-1]).contiguous(),
        lower_bound,
        initial_state.contiguous(),
        final_state,
        cu_seqlens.contiguous(),
    )
    return out, final_state


def resolve_kda_prefill_backend(
    backend: str,
    head_dim: int,
    dtype: torch.dtype,
    lower_bound: float | None,
) -> str:
    if backend not in ("auto", "triton", "flashkda"):
        raise ValueError(f"Unsupported KDA prefill backend: {backend}")
    supported = is_flashkda_supported(head_dim, dtype, lower_bound)
    if backend == "flashkda" and not supported:
        raise RuntimeError(
            "FlashKDA requires CUDA SM90/SM10x/SM12x, bfloat16, "
            "head_dim=128, and a bounded KDA gate."
        )
    if supported and backend != "triton":
        logger.info_once("Using FlashKDA KDA prefill backend.")
        return "flashkda"
    return "triton"


def _make_decode_conv1d_weight_loader(
    dims: list[int],
    tp_size: int,
    tp_rank: int,
    decode_conv1d_weight: torch.Tensor | None,
) -> Callable[..., None]:
    sharded_dims = [dim // tp_size for dim in dims]

    def weight_loader(
        param: torch.Tensor,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)
        shard_size = sharded_dims[loaded_shard_id]
        source_start = tp_rank * shard_size
        target_start = sum(sharded_dims[:loaded_shard_id])
        loaded_shard = loaded_weight[source_start : source_start + shard_size]
        param.data[target_start : target_start + shard_size].copy_(loaded_shard)
        if decode_conv1d_weight is not None and not param.is_meta:
            decode_conv1d_weight[loaded_shard_id].copy_(
                loaded_shard.squeeze(1).transpose(0, 1)
            )

    return weight_loader


def _make_decode_norm_weight_loader(
    decode_norm_weight: torch.Tensor,
) -> Callable[..., None]:
    def weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        default_weight_loader(param, loaded_weight)
        if not param.is_meta:
            decode_norm_weight.copy_(param.data)

    return weight_loader


class KimiK3DeltaAttention(GatedDeltaNetAttention):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return KimiK3KDAAttentionBackend

    def get_state_dtype(
        self,
    ) -> tuple[torch.dtype, ...]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        base_dtypes = MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )
        if self.cache_config.use_kda_recoverssm:
            return MambaStateDtypeCalculator.append_kda_recoverssm_record(
                base_dtypes, self.model_config.dtype
            )
        return base_dtypes

    def get_state_shape(
        self,
    ) -> tuple[tuple[int, ...], ...]:
        base_shapes = MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )
        if self.cache_config.use_kda_recoverssm:
            return MambaStateShapeCalculator.append_kda_recoverssm_record(
                base_shapes,
                self.num_heads,
                self.head_dim,
                tp_world_size=self.tp_size,
                spec_query_len=1 + self.num_spec,
            )
        return base_shapes

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)
        self.use_recoverssm = self.cache_config.use_kda_recoverssm
        if self.cache_config.use_replayssm and not self.use_recoverssm:
            raise ValueError(
                "Kimi-K3 supports --use-replayssm only with speculative decoding"
            )
        self.spec_query_len = 1 + self.num_spec

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)
        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires a full-rank gate"
        )

        # Keep f_a before the narrow beta shard, then pad each TP-local row
        # to select the aligned BF16 GEMM path.
        qkvg_output_sizes = [self.projection_size] * 4
        in_proj_output_sizes = qkvg_output_sizes + [
            self.head_dim,
            self.num_heads,
        ]
        local_output_size = (
            4 * self.local_projection_size + self.head_dim + self.local_num_heads
        )
        self.in_proj_padding = -local_output_size % 16
        if self.in_proj_padding:
            in_proj_output_sizes.append(self.in_proj_padding * self.tp_size)
        self.in_proj_qkvgfab = _KimiGDNMergedColumnParallelLinear(
            self.hidden_size,
            in_proj_output_sizes,
            replicated_shard_id=4,
            tp_size=self.tp_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.in_proj_qkvgfab",
        )
        if self.in_proj_padding:
            self.in_proj_qkvgfab.weight.data[-self.in_proj_padding :].zero_()

        self.f_b_proj = ColumnParallelLinear(
            self.head_dim,
            self.projection_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.f_b_proj",
        )
        self.dt_bias = nn.Parameter(
            torch.empty(self.local_projection_size, dtype=torch.float32)
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # One packed parameter and cache let decode run a single conv update.
        # Prefill slices them back into Q/K/V to obtain dense outputs cheaply.
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_size,
            output_size=3 * self.projection_size,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        # Keep a width-major copy for fused decode without changing the layout
        # consumed by the prefill and fallback decode kernels.
        conv_state_dtype = self.get_state_dtype()[0]
        decode_conv1d_weight = None
        if is_fused_kda_decode_supported(
            self.local_num_heads,
            self.head_dim,
            self.conv_size,
            self.num_spec,
            vllm_config.model_config.dtype,
            conv_state_dtype,
        ):
            logger.info_once("Fused KDA decode kernel (conv+KDA+norm) is enabled.")
            decode_conv1d_weight = torch.empty(
                3,
                self.conv_size,
                self.local_projection_size,
                dtype=self.conv1d.weight.dtype,
                device=self.conv1d.weight.device,
            )
        self.register_buffer(
            "decode_conv1d_weight", decode_conv1d_weight, persistent=False
        )
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": _make_decode_conv1d_weight_loader(
                    [self.projection_size] * 3,
                    self.tp_size,
                    self.tp_rank,
                    decode_conv1d_weight,
                )
            },
        )

        self.A_log = nn.Parameter(
            torch.empty(self.local_num_heads, dtype=torch.float32)
        )
        set_weight_attrs(self.A_log, {"weight_loader": a_log_weight_loader(0)})

        self.gate_lower_bound: float | None = kda_config.get("gate_lower_bound", None)
        if self.gate_lower_bound is not None:
            assert _KDA_GATE_LOGBOUND_MIN <= self.gate_lower_bound < 0, (
                "KDA gate lower bound must be in "
                f"[{_KDA_GATE_LOGBOUND_MIN}, 0). "
                f"Got {self.gate_lower_bound}."
            )

        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        self.kda_prefill_backend = resolve_kda_prefill_backend(
            backend,
            self.head_dim,
            vllm_config.model_config.dtype,
            self.gate_lower_bound,
        )

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        decode_norm_weight = None
        if decode_conv1d_weight is not None:
            decode_norm_weight = torch.empty(
                self.head_dim,
                dtype=torch.float32,
                device=self.o_norm.weight.device,
            )
        self.register_buffer("decode_norm_weight", decode_norm_weight, persistent=False)
        if decode_norm_weight is not None:
            # Upcast once while loading; direct BF16 norm weights slow the
            # fully fused decode kernel.
            if hasattr(self.o_norm.weight, "weight_loader"):
                delattr(self.o_norm.weight, "weight_loader")
            set_weight_attrs(
                self.o_norm.weight,
                {"weight_loader": _make_decode_norm_weight_loader(decode_norm_weight)},
            )
        self.o_proj = RowParallelLinear(
            self.projection_size,
            self.hidden_size,
            bias=False,
            quant_config=self.quant_config,
            prefix=f"{prefix}.o_proj",
        )

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = hidden_states.size(0)
        projected_qkvgfab = self.in_proj_qkvgfab(hidden_states)[0]
        split_sizes = [
            3 * self.local_projection_size,
            self.local_projection_size,
            self.head_dim,
            self.local_num_heads,
        ]
        if self.in_proj_padding:
            split_sizes.append(self.in_proj_padding)
        projected = projected_qkvgfab.split(split_sizes, dim=-1)
        mixed_qkv, g_proj_states, f_a, beta = projected[:4]

        g1 = self.f_b_proj(f_a)[0]
        beta = beta.unsqueeze(0)
        g1 = rearrange(g1, "n (h d) -> 1 n h d", d=self.head_dim)
        g2 = rearrange(g_proj_states, "... (h d) -> ... h d", d=self.head_dim)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self._forward(
            mixed_qkv=mixed_qkv,
            g1=g1,
            g2=g2,
            beta=beta,
            core_attn_out=core_attn_out,
        )
        core_attn_out = rearrange(core_attn_out, "1 n h d -> n (h d)")
        return self.o_proj(core_attn_out)[0]

    @eager_break_during_capture
    def _forward(
        self,
        mixed_qkv: torch.Tensor,
        g1: torch.Tensor,
        g2: torch.Tensor,
        beta: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            return

        from vllm.models.kimi_k3.nvidia.ops.third_party.kda import (
            chunk_kda_with_fused_gate,
            fused_recurrent_kda,
            fused_recurrent_kda_packed_decode,
        )

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, KimiK3KDAMetadata)
        m = attn_metadata_narrowed
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        has_spec_decode = m.num_spec_decodes > 0
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        conv_state, recurrent_state, *recoverssm_records = self.kv_cache
        # The convolution kernels consume (..., dim, width - 1).
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        if (
            self.decode_conv1d_weight is not None
            and self.decode_norm_weight is not None
            and not has_spec_decode
            and m.num_prefills == 0
            and m.num_decodes > 0
        ):
            assert non_spec_state_indices_tensor is not None
            ops.fused_kda_decode(
                x=mixed_qkv,
                weight=self.decode_conv1d_weight,
                bias=self.conv1d.bias,
                conv_state=conv_state,
                raw_g=g1,
                raw_beta=beta,
                A_log=self.A_log,
                dt_bias=self.dt_bias,
                state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                state=recurrent_state,
                out=core_attn_out[:, :num_actual_tokens],
                lower_bound=self.gate_lower_bound,
                output_gate=g2[:num_actual_tokens],
                norm_weight=self.decode_norm_weight,
                norm_eps=self.o_norm.eps,
            )
            return

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        q_conv_weight, k_conv_weight, v_conv_weight = conv_weights.split(
            self.local_projection_size, dim=0
        )
        q_conv_state, k_conv_state, v_conv_state = conv_state.split(
            self.local_projection_size, dim=-2
        )

        # Separate multi-query speculative tokens from prefill/plain decode.
        if has_spec_decode:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

        # Spec-decode multi-query path.
        core_attn_out_spec = None
        if has_spec_decode:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = (
                self.spec_query_len
                if self.use_recoverssm
                else spec_state_indices_tensor.size(-1)
            )
            spec_conv_out = torch.empty_like(mixed_qkv_spec)
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                activation="silu",
                conv_state_indices=spec_conv_indices,
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_max_query_len,
                validate_data=False,
                out=spec_conv_out,
            )
            q_spec, k_spec, v_spec = (
                rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                for x in mixed_qkv_spec.split(self.local_projection_size, dim=-1)
            )
            spec_cu_seqlens = spec_query_start_loc[: m.num_spec_decodes + 1]
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
            if self.use_recoverssm:
                from vllm.models.kimi_k3.nvidia.ops.recoverssm import (
                    kda_recoverssm_verify,
                )

                if len(recoverssm_records) != 2:
                    raise ValueError(
                        "KDA RecoverSSM requires correction and key/gate buffers"
                    )
                core_attn_out_spec = kda_recoverssm_verify(
                    q=q_spec,
                    k=k_spec,
                    v=v_spec,
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    checkpoint_state=recurrent_state,
                    correction_cache=recoverssm_records[0],
                    kg_cache=recoverssm_records[1],
                    query_start_loc=spec_cu_seqlens,
                    state_indices=spec_state_indices_tensor[: m.num_spec_decodes, 0],
                    spec_query_len=self.spec_query_len,
                    out=spec_out,
                )
            else:
                core_attn_out_spec, _ = fused_recurrent_kda(
                    q=q_spec,
                    k=k_spec,
                    v=v_spec,
                    raw_g=g1_spec,
                    raw_beta=beta_spec,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    cu_seqlens=spec_cu_seqlens,
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    out=spec_out,
                )

        # Prefill or plain-decode path.
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                # Separate convolution calls accept row-strided packed inputs
                # and produce dense Q/K/V without an additional V copy.
                def _prefill_conv(
                    x: torch.Tensor,
                    state: torch.Tensor,
                    weight: torch.Tensor,
                ) -> torch.Tensor:
                    return causal_conv1d_fn(
                        x.transpose(0, 1),
                        weight,
                        None,
                        activation="silu",
                        conv_states=state,
                        has_initial_state=has_initial_state,
                        cache_indices=non_spec_state_indices_tensor,
                        query_start_loc=non_spec_query_start_loc,
                        metadata=m,
                    ).transpose(0, 1)

                q_ns = _prefill_conv(q_ns, q_conv_state, q_conv_weight)
                k_ns = _prefill_conv(k_ns, k_conv_state, k_conv_weight)
                v_ns = _prefill_conv(v_ns, v_conv_state, v_conv_weight)
                q_ns, k_ns, v_ns = (
                    rearrange(x, "n (h d) -> 1 n h d", d=self.head_dim)
                    for x in (q_ns, k_ns, v_ns)
                )

                assert non_spec_state_indices_tensor is not None
                assert has_initial_state is not None
                initial_state = gather_initial_states(
                    recurrent_state,
                    non_spec_state_indices_tensor,
                    has_initial_state,
                )
                if self.kda_prefill_backend == "flashkda":
                    assert self.gate_lower_bound is not None
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = _flashkda_prefill(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        g=g1_ns,
                        beta=beta_ns,
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                else:
                    (
                        core_attn_out_non_spec,
                        last_recurrent_state,
                    ) = chunk_kda_with_fused_gate(
                        q=q_ns,
                        k=k_ns,
                        v=v_ns,
                        raw_g=g1_ns,
                        raw_beta=beta_ns,
                        A_log=self.A_log,
                        g_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=initial_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                    )
                recurrent_state[non_spec_state_indices_tensor] = last_recurrent_state
            else:
                # Pure non-speculative decode.
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                packed_conv_out = torch.empty_like(mixed_qkv_ns)
                mixed_qkv_ns = causal_conv1d_update(
                    mixed_qkv_ns,
                    conv_state,
                    conv_weights,
                    self.conv1d.bias,
                    activation="silu",
                    conv_state_indices=decode_conv_indices,
                    validate_data=True,
                    out=packed_conv_out,
                )
                (
                    core_attn_out_non_spec,
                    _,
                ) = fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                )

        # Restore the scheduler's original token order for mixed batches.
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        elif core_attn_out_non_spec is not None:
            # TODO: prefill and decode kernels write directly to core_attn_out
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_spec is not None
        # Triton normalizes in place, so this is a self-copy with no device
        # work. Keep it for the out-of-place native implementation.
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))
