# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from einops import rearrange
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.distributed import divide
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention

# Generic KDA helpers, shared with Kimi-Linear. They are neither ROCm- nor
# K3-specific, so they are imported rather than duplicated. (nvidia/kda.py keeps
# its own copies; duplicating here instead would be a one-line change.)
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    _KDA_GATE_LOGBOUND_MIN,
    _KimiGDNMergedColumnParallelLinear,
    _make_fused_conv1d_weight_loader,
    a_log_weight_loader,
)
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
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.utils import set_weight_attrs
from vllm.models.kimi_k3.amd.kda_metadata import KimiK3ROCmKDABackend
from vllm.models.kimi_k3.amd.ops.kda_decode import (
    is_fused_kda_decode_supported,
    make_decode_conv1d_weight_loader,
    make_decode_norm_weight_loader,
)
from vllm.models.kimi_k3.amd.ops.third_party.kda import (
    chunk_kda_with_fused_gate,
    fused_recurrent_kda,
    fused_recurrent_kda_packed_decode,
)
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

logger = init_logger(__name__)


class KimiK3DeltaAttention(GatedDeltaNetAttention):
    def get_attn_backend(self) -> type[AttentionBackend]:
        return KimiK3ROCmKDABackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        if self.model_config is None or self.cache_config is None:
            raise ValueError("model_config and cache_config must be set")
        return MambaStateDtypeCalculator.kda_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.kda_state_shape(
            self.tp_size,
            self.num_heads,
            self.head_dim,
            conv_kernel_size=self.conv_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config: KimiLinearConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__(config, vllm_config, prefix)

        kda_config = config.linear_attn_config  # type: ignore[attr-defined]
        assert kda_config is not None, "linear_attn_config must be set"
        assert kda_config.get("use_full_rank_gate", False), (
            "KimiK3DeltaAttention requires use_full_rank_gate; the low-rank "
            "gate path belongs to the shared Kimi-Linear layer."
        )

        self.head_dim = kda_config["head_dim"]
        self.num_heads = kda_config["num_heads"]
        assert self.num_heads % self.tp_size == 0
        self.local_num_heads = divide(self.num_heads, self.tp_size)

        self.projection_size = self.head_dim * self.num_heads
        self.local_projection_size = divide(self.projection_size, self.tp_size)
        self.conv_size = kda_config["short_conv_kernel_size"]
        self.use_full_rank_gate = True

        # Keep f_a before the narrow beta shard, then pad each TP-local row to
        # select the aligned BF16 GEMM path. The padding also avoids an Inductor
        # correctness issue seen with the row-strided G view.
        qkvg_output_sizes = [self.projection_size] * 4
        in_proj_output_sizes = qkvg_output_sizes + [self.head_dim, self.num_heads]
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
        delattr(self.conv1d.weight, "weight_loader")
        # ROCm can fuse the whole decode step (conv + recurrence + gated norm)
        # into one kernel, which wants a width-major fp32 conv weight staged at
        # load time. Everything else keeps the [channel, width] layout.
        conv_state_dtype, _ = self.get_state_dtype()
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
        if decode_conv1d_weight is None:
            conv1d_weight_loader = _make_fused_conv1d_weight_loader(
                [self.projection_size] * 3,
                self.tp_size,
                self.tp_rank,
            )
        else:
            conv1d_weight_loader = make_decode_conv1d_weight_loader(
                [self.projection_size] * 3,
                self.tp_size,
                self.tp_rank,
                decode_conv1d_weight,
            )
        set_weight_attrs(self.conv1d.weight, {"weight_loader": conv1d_weight_loader})

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
        self.use_safe_gate = self.gate_lower_bound is not None

        additional_config = vllm_config.additional_config
        backend = (
            additional_config.get("kda_prefill_backend", "auto")
            if isinstance(additional_config, dict)
            else "auto"
        )
        backend = "triton" if backend == "auto" else backend
        assert backend == "triton", (
            "The ROCm Kimi-K3 KDA layer only supports the Triton KDA prefill "
            f"backend, got {backend!r}."
        )

        self.o_norm = FusedRMSNormGated(self.head_dim, activation="sigmoid")
        decode_norm_weight = None
        if decode_conv1d_weight is not None:
            # Upcast once at load time; a BF16 norm weight slows the fused
            # decode kernel's epilogue.
            decode_norm_weight = torch.empty(
                self.head_dim,
                dtype=torch.float32,
                device=self.o_norm.weight.device,
            )
            if hasattr(self.o_norm.weight, "weight_loader"):
                delattr(self.o_norm.weight, "weight_loader")
            set_weight_attrs(
                self.o_norm.weight,
                {"weight_loader": make_decode_norm_weight_loader(decode_norm_weight)},
            )
        self.register_buffer("decode_norm_weight", decode_norm_weight, persistent=False)
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

    def rearrange_mixed_qkv(
        self, mixed_qkv: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = mixed_qkv.shape[0]
        qkv = mixed_qkv.view(seq_len, 3, self.local_num_heads, self.head_dim)
        # Materialize all three row-strided inputs with one token-major to
        # QKV-major permutation. Each unbound tensor is then contiguous.
        qkv = qkv.permute(1, 0, 2, 3).contiguous().unsqueeze(1)
        return qkv.unbind(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
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
        output[:] = self.o_proj(core_attn_out)[0]

    @eager_break_during_capture(always_break=True)
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

        assert isinstance(attn_metadata_raw, dict)
        attn_metadata_narrowed = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata_narrowed, GDNAttentionMetadata)
        m = attn_metadata_narrowed
        has_initial_state = m.has_initial_state
        non_spec_query_start_loc = m.non_spec_query_start_loc
        non_spec_state_indices_tensor = m.non_spec_state_indices_tensor
        spec_sequence_masks = m.spec_sequence_masks
        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        spec_state_indices_tensor = m.spec_state_indices_tensor
        spec_query_start_loc = m.spec_query_start_loc
        num_accepted_tokens = m.num_accepted_tokens
        num_actual_tokens = m.num_actual_tokens
        mixed_qkv = mixed_qkv[:num_actual_tokens]
        g1 = g1[:, :num_actual_tokens]
        beta = beta[:, :num_actual_tokens]

        constant_caches = self.kv_cache

        conv_state, recurrent_state = constant_caches
        # conv_state must be (..., dim, width-1) for the conv kernels.
        # DS layout stores it that way directly; SD layout needs a transpose.
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)

        if (
            self.decode_conv1d_weight is not None
            and self.decode_norm_weight is not None
            and spec_sequence_masks is None
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

        # Split tokens into the multi-query spec-decode part and the remaining
        # (prefill / plain decode) part.
        if spec_sequence_masks is not None:
            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                g1_spec = g1.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                mixed_qkv_ns = mixed_qkv.index_select(0, non_spec_token_indx)
                g1_ns = g1.index_select(1, non_spec_token_indx)
                beta_ns = beta.index_select(1, non_spec_token_indx)
        else:
            mixed_qkv_spec = g1_spec = beta_spec = None
            mixed_qkv_ns, g1_ns, beta_ns = mixed_qkv, g1, beta

        # ---------- spec-decode multi-query path ----------
        core_attn_out_spec = None
        if spec_sequence_masks is not None:
            assert spec_state_indices_tensor is not None
            assert spec_query_start_loc is not None
            spec_conv_indices = spec_state_indices_tensor[:, 0][: m.num_spec_decodes]
            spec_max_query_len = spec_state_indices_tensor.size(-1)

            # Sibling beta and, for full-rank gates, output-gate views remain
            # live, so write the convolution output separately.
            spec_conv_out = torch.empty(
                mixed_qkv_spec.shape,
                dtype=mixed_qkv_spec.dtype,
                device=mixed_qkv_spec.device,
            )
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
            # Spec-only batches write directly into core_attn_out.
            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
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

        # ---------- non-spec path (prefill or plain decode) ----------
        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
                q_ns, k_ns, v_ns = mixed_qkv_ns.split(
                    self.local_projection_size, dim=-1
                )

                # Packed prefill conv would require copying V solely to make
                # it dense for KDA. Separate calls accept the strided inputs
                # and produce dense Q/K/V without that extra traffic.
                # TODO: Use packed conv once every KDA prefill backend accepts
                # row-strided Q/K/V directly.
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

                # A mixed non-spec batch is decode-first: the decodes are
                # length-1 sequences, and the chunk kernel returns NaN for those.
                # Send them to the recurrent kernel and give the chunk kernel the
                # prefill tail only.
                core_attn_out_decode = None
                split_non_spec = spec_sequence_masks is None and m.num_decodes > 0
                if split_non_spec:
                    assert non_spec_query_start_loc is not None
                    nd_tok = m.num_decode_tokens
                    core_attn_out_decode, _ = fused_recurrent_kda(
                        q=q_ns[:, :nd_tok],
                        k=k_ns[:, :nd_tok],
                        v=v_ns[:, :nd_tok],
                        raw_g=g1_ns[:, :nd_tok],
                        raw_beta=beta_ns[:, :nd_tok],
                        A_log=self.A_log,
                        dt_bias=self.dt_bias,
                        lower_bound=self.gate_lower_bound,
                        initial_state=recurrent_state,
                        cu_seqlens=non_spec_query_start_loc[: m.num_decodes + 1],
                        ssm_state_indices=non_spec_state_indices_tensor[
                            : m.num_decodes
                        ],
                    )
                    q_ns = q_ns[:, nd_tok:]
                    k_ns = k_ns[:, nd_tok:]
                    v_ns = v_ns[:, nd_tok:]
                    g1_ns = g1_ns[:, nd_tok:]
                    beta_ns = beta_ns[:, nd_tok:]
                    prefill_query_start_loc = m.prefill_query_start_loc
                    prefill_state_indices = m.prefill_state_indices
                    prefill_has_initial_state = m.prefill_has_initial_state
                    assert prefill_query_start_loc is not None
                    assert prefill_state_indices is not None
                    assert prefill_has_initial_state is not None
                else:
                    prefill_query_start_loc = non_spec_query_start_loc
                    prefill_state_indices = non_spec_state_indices_tensor
                    prefill_has_initial_state = has_initial_state

                initial_state = gather_initial_states(
                    recurrent_state,
                    prefill_state_indices,
                    prefill_has_initial_state,
                )
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
                    cu_seqlens=prefill_query_start_loc,
                    chunk_indices=m.chunk_indices,
                    chunk_offsets=m.chunk_offsets,
                )
                # Init cache
                recurrent_state[prefill_state_indices] = last_recurrent_state

                if split_non_spec:
                    # Restore decode-first token order for the merge below.
                    core_attn_out_non_spec = torch.cat(
                        [core_attn_out_decode, core_attn_out_non_spec], dim=1
                    )

            else:
                # pure-decode non-spec batch
                assert non_spec_state_indices_tensor is not None
                decode_conv_indices = non_spec_state_indices_tensor[
                    : mixed_qkv_ns.size(0)
                ]
                # Sibling beta and, for full-rank gates, output-gate views
                # remain live, so write the conv output separately.
                packed_conv_out = torch.empty(
                    mixed_qkv_ns.shape,
                    dtype=mixed_qkv_ns.dtype,
                    device=mixed_qkv_ns.device,
                )
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
                core_attn_out_non_spec, _ = fused_recurrent_kda_packed_decode(
                    mixed_qkv=mixed_qkv_ns,
                    raw_g=g1_ns,
                    raw_beta=beta_ns,
                    A_log=self.A_log,
                    dt_bias=self.dt_bias,
                    lower_bound=self.gate_lower_bound,
                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                )

        # ---------- merge spec and non-spec outputs ----------
        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            # Mixed batches require indexed placement in the original order.
            merged = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_spec.dtype,
                device=core_attn_out_spec.device,
            )
            merged.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[0, :num_actual_tokens] = merged[0, :num_actual_tokens]
        elif core_attn_out_non_spec is not None:
            core_attn_out[0, :num_actual_tokens] = core_attn_out_non_spec[
                0, :num_actual_tokens
            ]
        else:
            assert core_attn_out_spec is not None
        core_attn_out.copy_(self.o_norm(core_attn_out, g2))
